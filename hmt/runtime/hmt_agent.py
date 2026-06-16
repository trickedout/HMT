from __future__ import annotations

"""End-to-end HMT agent runtime.

The runtime implements the HMT inference path:

1. Normalize the incoming instruction into a canonical intent representation.
2. Retrieve candidate intent nodes from ``T_mem`` with local Qwen embeddings.
3. Retrieve stage nodes under those intents and combine semantic retrieval with
   observable pre/post-condition matching.
4. Use the Planner to select the active stage with a confidence-aware fallback.
5. Retrieve action patterns under the selected stage.
6. Use the Actor to ground the stored semantic description to the current page's
   candidate elements without copying source-site identifiers.

The runtime can be used in two modes.  ``deterministic`` uses the exact retrieval
and lexical matching implementation, which is useful for local runs and ablations.  ``llm_assisted`` additionally calls GPT-4 for stage selection and/or
action grounding after the candidate set has been constrained by HMT.  GPT-4 is
the only remote model; Qwen embedding/reranking remains local.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import time

from hmt.core.actor import ActorOutput, ground_action, llm_ground_action
from hmt.core.construction import LLMClient, normalize_instruction
from hmt.core.memory_tree import HMTConfig, MemoryTree, NormalizedIntent, StageNode
from hmt.core.planner import PlannerOutput, confidence_aware_select, llm_select_stage
from hmt.core.retrieval import ScoredNode, retrieve_actions, retrieve_all_actions, retrieve_intents, retrieve_stages
from hmt.models.embeddings import EmbeddingModel
from hmt.models.reranker import CrossEncoderReranker
from hmt.core.state_abstraction import abstract_state
from hmt.utils.io import read_yaml


@dataclass
class HMTAgentRuntimeConfig:
    config: HMTConfig
    use_llm_normalizer: bool = True
    use_llm_planner: bool = False
    use_llm_actor: bool = False
    use_base_policy: bool = True
    fallback_to_base_on_low_confidence: bool = True
    planner_expand_multiplier: int = 2
    action_expand_multiplier: int = 3
    stop_on_no_stage: bool = True
    stop_on_no_action: bool = True
    return_debug: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "HMTAgentRuntimeConfig":
        inference = data.get("inference", {})
        return cls(
            config=HMTConfig.from_mapping(data),
            use_llm_normalizer=bool(inference.get("use_llm_normalizer", True)),
            use_llm_planner=bool(inference.get("use_llm_planner", False)),
            use_llm_actor=bool(inference.get("use_llm_actor", False)),
            use_base_policy=bool(inference.get("use_base_policy", True)),
            fallback_to_base_on_low_confidence=bool(inference.get("fallback_to_base_on_low_confidence", True))
            and bool(data.get("ablation", {}).get("confidence_aware_fallback", True)),
            planner_expand_multiplier=int(inference.get("planner_expand_multiplier", 2)),
            action_expand_multiplier=int(inference.get("action_expand_multiplier", 3)),
            stop_on_no_stage=bool(inference.get("stop_on_no_stage", True)),
            stop_on_no_action=bool(inference.get("stop_on_no_action", True)),
            return_debug=bool(inference.get("return_debug", True)),
        )


@dataclass
class AgentStepInput:
    instruction: str
    observation: dict[str, Any] | str
    previous_actions: list[dict[str, Any]] = field(default_factory=list)
    candidate_elements: list[dict[str, Any]] | None = None
    task_id: str = ""
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HMTActionPrediction:
    operation: str
    target_element_id: str | None
    argument: str | None
    confidence: float
    selected_stage_id: str | None
    low_confidence: bool
    normalized_intent: dict[str, Any]
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "target_element_id": self.target_element_id,
            "argument": self.argument,
            "confidence": self.confidence,
            "selected_stage_id": self.selected_stage_id,
            "low_confidence": self.low_confidence,
            "normalized_intent": self.normalized_intent,
            "debug": self.debug,
        }


class HMTAgent:
    """Reusable inference object for Mind2Web and WebArena runners."""

    def __init__(
        self,
        memory: MemoryTree,
        runtime_config: HMTAgentRuntimeConfig,
        llm_client: LLMClient | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.memory = memory
        self.runtime_config = runtime_config
        self.config = runtime_config.config
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.reranker = reranker

    @classmethod
    def from_config_files(
        cls,
        config_path: str | Path,
        memory_path: str | Path | None = None,
        llm_client: LLMClient | None = None,
    ) -> "HMTAgent":
        data = read_yaml(config_path)
        runtime = HMTAgentRuntimeConfig.from_mapping(data)
        memory_cfg = data.get("memory", {})
        memory_source = memory_path or memory_cfg.get("path")
        if not memory_source:
            raise RuntimeError("A memory path must be supplied via --memory or memory.path in the config.")
        resolved_memory_path = Path(memory_source)
        memory = MemoryTree.load_jsonl(resolved_memory_path)
        embedding = EmbeddingModel.from_config(data)
        reranker = CrossEncoderReranker.from_config(data)
        return cls(memory=memory, runtime_config=runtime, llm_client=llm_client, embedding_model=embedding, reranker=reranker)

    def predict(self, step: AgentStepInput) -> HMTActionPrediction:
        started = time.perf_counter()
        normalized = self._normalize(step)
        state = abstract_state(
            self._merge_observation_and_candidates(step.observation, step.candidate_elements),
            instruction=step.instruction,
            recent_actions=step.previous_actions,
            max_salient_elements=self.config.max_salient_elements,
            history_truncation=self.config.history_truncation,
        )
        summary = state.to_summary_dict()
        query = self._query_text(step.instruction, normalized, summary["summary_text"])
        if self.config.use_intent_level:
            intent_hits = retrieve_intents(
                self.memory,
                query,
                top_k=self.config.task_top_k,
                config=self.config,
                embedding_model=self.embedding_model,
                reranker=self.reranker,
            )
        else:
            intent_hits = [ScoredNode(node=node, score=1.0, reason="no_intent_level_ablation") for node in self.memory.intents.values()]
        if not self.config.use_stage_level:
            return self._predict_without_stage_level(step, normalized, intent_hits, query, summary, started)
        stage_hits = retrieve_stages(
            self.memory,
            intent_hits,
            query=query,
            observation_summary=summary["summary_text"],
            config=self.config,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
        )
        expanded_stage_hits = self._expanded_stage_hits(intent_hits, query, summary["summary_text"], stage_hits)
        if self.config.use_planner:
            planner_output = confidence_aware_select(stage_hits, expanded_stage_hits, self.config)
        else:
            top_stage_id = stage_hits[0].node.stage_id if stage_hits else None
            top_score = float(stage_hits[0].score) if stage_hits else 0.0
            planner_output = PlannerOutput(top_stage_id, max(0.0, min(1.0, top_score)), False, [], [], "planner disabled: chose top retrieved stage")
        if self.runtime_config.use_llm_planner and self.llm_client and stage_hits:
            planner_output = llm_select_stage(
                instruction=step.instruction,
                intent_json=normalized.to_dict(),
                recent_actions=step.previous_actions,
                observation_summary=summary["summary_text"],
                stage_hits=stage_hits,
                client=self.llm_client,
            )
        selected_stage = self.memory.stages.get(planner_output.selected_stage_id or "")
        if selected_stage is None:
            base_prediction = self._base_policy_prediction(step, normalized, planner_output, intent_hits, stage_hits, summary, started, "no stage selected")
            if base_prediction is not None:
                return base_prediction
            return self._stop_prediction(step, normalized, planner_output, intent_hits, stage_hits, summary, started, "no stage selected")
        if planner_output.low_confidence and self.runtime_config.fallback_to_base_on_low_confidence:
            base_prediction = self._base_policy_prediction(step, normalized, planner_output, intent_hits, stage_hits, summary, started, "low-confidence stage selection")
            if base_prediction is not None:
                return base_prediction
        action_hits = retrieve_actions(
            self.memory,
            selected_stage.stage_id,
            query=query + "\n" + summary["summary_text"],
            top_k=max(self.config.action_top_k, self.config.action_top_k * self.runtime_config.action_expand_multiplier),
            config=self.config,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
        )
        if not action_hits:
            base_prediction = self._base_policy_prediction(step, normalized, planner_output, intent_hits, stage_hits, summary, started, "no actions for selected stage")
            if base_prediction is not None:
                return base_prediction
            return self._stop_prediction(step, normalized, planner_output, intent_hits, stage_hits, summary, started, "no actions for selected stage")
        actor_output = ground_action(action_hits[: self.config.action_top_k], summary["salient_elements"])
        if self.runtime_config.use_llm_actor and self.llm_client:
            actor_output = llm_ground_action(
                instruction=step.instruction,
                stage_json=selected_stage.to_record(),
                action_hits=action_hits[: self.config.action_top_k],
                candidate_elements=summary["salient_elements"],
                client=self.llm_client,
            )
        debug = self._debug_payload(
            step=step,
            started=started,
            summary=summary,
            intent_hits=intent_hits,
            stage_hits=stage_hits,
            expanded_stage_hits=expanded_stage_hits,
            selected_stage=selected_stage,
            action_hits=action_hits,
            planner_output=planner_output,
            actor_output=actor_output,
        )
        return HMTActionPrediction(
            operation=actor_output.operation,
            target_element_id=actor_output.target_element_id,
            argument=actor_output.argument,
            confidence=actor_output.confidence,
            selected_stage_id=planner_output.selected_stage_id,
            low_confidence=planner_output.low_confidence,
            normalized_intent=normalized.to_dict(),
            debug=debug if self.runtime_config.return_debug else {},
        )

    def _normalize(self, step: AgentStepInput) -> NormalizedIntent:
        client = self.llm_client if self.runtime_config.use_llm_normalizer else None
        return normalize_instruction(step.instruction, client=client, record_id=step.task_id or None)

    def _merge_observation_and_candidates(
        self,
        observation: dict[str, Any] | str,
        candidate_elements: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | str:
        if candidate_elements is None:
            return observation
        if isinstance(observation, dict):
            merged = dict(observation)
            merged["candidates"] = candidate_elements
            return merged
        return {"text": str(observation), "candidates": candidate_elements}

    def _query_text(self, instruction: str, normalized: NormalizedIntent, summary_text: str) -> str:
        intent = json.dumps(normalized.to_dict(), ensure_ascii=False, sort_keys=True)
        recent_summary = "\n".join(summary_text.splitlines()[: self.config.max_salient_elements + 2])
        return f"instruction: {instruction}\nnormalized_intent: {intent}\ncurrent_observation: {recent_summary}"

    def _expanded_stage_hits(
        self,
        intent_hits: list[ScoredNode],
        query: str,
        observation_summary: str,
        stage_hits: list[ScoredNode],
    ) -> list[ScoredNode] | None:
        if not stage_hits:
            return None
        original_top_k = self.config.stage_top_k
        self.config.stage_top_k = max(original_top_k, original_top_k * self.runtime_config.planner_expand_multiplier)
        try:
            expanded = retrieve_stages(
                self.memory,
                intent_hits,
                query=query,
                observation_summary=observation_summary,
                config=self.config,
                embedding_model=self.embedding_model,
                reranker=self.reranker,
            )
        finally:
            self.config.stage_top_k = original_top_k
        return expanded

    def _base_policy_prediction(
        self,
        step: AgentStepInput,
        normalized: NormalizedIntent,
        planner_output: PlannerOutput,
        intent_hits: list[ScoredNode],
        stage_hits: list[ScoredNode],
        summary: dict[str, Any],
        started: float,
        reason: str,
    ) -> HMTActionPrediction | None:
        if not self.runtime_config.use_base_policy or self.llm_client is None:
            return None
        data = self.llm_client.complete_json(
            "base_actor.txt",
            {
                "instruction": step.instruction,
                "recent_actions": step.previous_actions,
                "observation_summary": summary.get("summary_text", ""),
                "candidate_elements": summary.get("salient_elements", []),
                "record_id": f"{step.task_id}:base_policy:{step.step_index}",
            },
            "actor_output",
        )
        operation = str(data.get("operation", "stop"))
        target_element_id = data.get("target_element_id")
        argument = data.get("argument")
        confidence = float(data.get("confidence", 0.0))
        debug = {
            "reason": f"base policy fallback: {reason}",
            "latency_sec": round(time.perf_counter() - started, 4),
            "intent_hits": self._serialize_hits(intent_hits),
            "stage_hits": self._serialize_hits(stage_hits),
            "planner": planner_output.to_dict(),
            "base_actor": data,
            "observation_summary": summary.get("summary_text", ""),
            "task_id": step.task_id,
            "step_index": step.step_index,
        }
        return HMTActionPrediction(
            operation=operation,
            target_element_id=None if target_element_id is None else str(target_element_id),
            argument=None if argument is None else str(argument),
            confidence=confidence,
            selected_stage_id=planner_output.selected_stage_id,
            low_confidence=True,
            normalized_intent=normalized.to_dict(),
            debug=debug if self.runtime_config.return_debug else {},
        )

    def _stop_prediction(
        self,
        step: AgentStepInput,
        normalized: NormalizedIntent,
        planner_output: PlannerOutput,
        intent_hits: list[ScoredNode],
        stage_hits: list[ScoredNode],
        summary: dict[str, Any],
        started: float,
        reason: str,
    ) -> HMTActionPrediction:
        debug = {
            "reason": reason,
            "latency_sec": round(time.perf_counter() - started, 4),
            "intent_hits": self._serialize_hits(intent_hits),
            "stage_hits": self._serialize_hits(stage_hits),
            "planner": planner_output.to_dict(),
            "observation_summary": summary.get("summary_text", ""),
            "task_id": step.task_id,
            "step_index": step.step_index,
        }
        return HMTActionPrediction(
            operation="stop",
            target_element_id=None,
            argument=None,
            confidence=0.0,
            selected_stage_id=planner_output.selected_stage_id,
            low_confidence=True,
            normalized_intent=normalized.to_dict(),
            debug=debug if self.runtime_config.return_debug else {},
        )

    def _debug_payload(
        self,
        step: AgentStepInput,
        started: float,
        summary: dict[str, Any],
        intent_hits: list[ScoredNode],
        stage_hits: list[ScoredNode],
        expanded_stage_hits: list[ScoredNode] | None,
        selected_stage: StageNode,
        action_hits: list[ScoredNode],
        planner_output: PlannerOutput,
        actor_output: ActorOutput,
    ) -> dict[str, Any]:
        return {
            "task_id": step.task_id,
            "step_index": step.step_index,
            "latency_sec": round(time.perf_counter() - started, 4),
            "observation_summary": summary.get("summary_text", ""),
            "num_candidate_elements": len(summary.get("salient_elements", [])),
            "intent_hits": self._serialize_hits(intent_hits),
            "stage_hits": self._serialize_hits(stage_hits),
            "expanded_stage_hits": self._serialize_hits(expanded_stage_hits or []),
            "selected_stage": selected_stage.to_record(),
            "planner": planner_output.to_dict(),
            "action_hits": self._serialize_hits(action_hits[: self.config.action_top_k]),
            "actor": actor_output.to_dict(),
        }

    def _serialize_hits(self, hits: Iterable[ScoredNode]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for hit in hits:
            node = hit.node
            if hasattr(node, "to_record"):
                payload = node.to_record()
            else:
                payload = {"repr": repr(node)}
            records.append({"score": hit.score, "reason": hit.reason, "node": payload})
        return records
