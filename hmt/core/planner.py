from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from hmt.core.memory_tree import HMTConfig, StageNode
from hmt.core.retrieval import ScoredNode


class PlannerLLMClient(Protocol):
    def complete_json(self, prompt_name: str, variables: dict[str, Any], format_name: str) -> dict[str, Any]:
        ...


@dataclass
class PlannerOutput:
    selected_stage_id: str | None
    confidence: float
    low_confidence: bool
    precondition_evidence: list[str]
    postcondition_evidence: list[str]
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "selected_stage_id": self.selected_stage_id,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
            "precondition_evidence": self.precondition_evidence,
            "postcondition_evidence": self.postcondition_evidence,
            "reason": self.reason,
        }


def is_low_confidence(scores: list[float], config: HMTConfig) -> bool:
    if not scores:
        return True
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2
    return top1 < config.fallback_abs_confidence_tau or margin < config.fallback_margin_delta


def select_stage(stage_hits: list[ScoredNode], config: HMTConfig) -> PlannerOutput:
    valid_hits = [hit for hit in stage_hits if hit.score >= 0]
    if not valid_hits:
        return PlannerOutput(None, 0.0, True, [], [], "no non-conflicting stage candidates")
    valid_hits.sort(key=lambda x: (-x.score, x.node.stage_id))
    scores = [hit.score for hit in valid_hits]
    low = is_low_confidence(scores, config)
    selected: StageNode = valid_hits[0].node
    return PlannerOutput(
        selected_stage_id=selected.stage_id,
        confidence=max(0.0, min(1.0, valid_hits[0].score)),
        low_confidence=low,
        precondition_evidence=selected.pre_conditions,
        postcondition_evidence=selected.post_conditions,
        reason=f"selected by combined retrieval score; condition decision={valid_hits[0].reason}",
    )


def confidence_aware_select(
    stage_hits: list[ScoredNode],
    expanded_stage_hits: list[ScoredNode] | None,
    config: HMTConfig,
    base_policy: Callable[[], PlannerOutput] | None = None,
) -> PlannerOutput:
    first = select_stage(stage_hits, config)
    if not first.low_confidence:
        return first
    if expanded_stage_hits is not None:
        second = select_stage(expanded_stage_hits, config)
        if not second.low_confidence:
            return second
    if base_policy:
        return base_policy()
    return first


def llm_select_stage(
    instruction: str,
    intent_json: dict[str, Any],
    recent_actions: list[dict[str, Any]],
    observation_summary: str,
    stage_hits: list[ScoredNode],
    client: PlannerLLMClient,
) -> PlannerOutput:
    data = client.complete_json(
        "planner.txt",
        {
            "instruction": instruction,
            "intent_json": intent_json,
            "recent_actions": recent_actions,
            "observation_summary": observation_summary,
            "candidate_stage_jsonl": "\n".join(
                json.dumps(
                    {
                        "stage_id": hit.node.stage_id,
                        "name": hit.node.name,
                        "pre_conditions": hit.node.pre_conditions,
                        "post_conditions": hit.node.post_conditions,
                        "retrieval_score": hit.score,
                    },
                    ensure_ascii=False,
                )
                for hit in stage_hits
            ),
        },
        "planner_output",
    )
    return PlannerOutput(
        selected_stage_id=data.get("selected_stage_id"),
        confidence=float(data.get("confidence", 0.0)),
        low_confidence=bool(data.get("low_confidence", False)),
        precondition_evidence=[str(x) for x in data.get("precondition_evidence", [])],
        postcondition_evidence=[str(x) for x in data.get("postcondition_evidence", [])],
        reason=str(data.get("reason", "")),
    )
