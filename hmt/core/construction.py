from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hmt.core.memory_tree import (
    ActionNode,
    ConsistencyReport,
    Constraint,
    Episode,
    HMTConfig,
    IntentNode,
    MemoryTree,
    NormalizedIntent,
    FORMAT_VERSION,
    SemanticDescription,
    SourceMetadata,
    Span,
    StageDraft,
    StageNode,
    StageSpan,
    Trajectory,
    TrajectoryStep,
    TRANSFER_FORBIDDEN_KEYS,
)
from hmt.core.structured_output import StructuredOutputError, validate_data, validate_memory_record
from hmt.utils.io import repository_root, write_jsonl
from hmt.utils.json_utils import extract_json_object
from hmt.utils.logging import get_logger
from hmt.core.semantic_abstraction import abstract_action_from_step
from hmt.core.stage_reasoning import stage_drafts_from_trajectory

LOGGER = get_logger(__name__)


class LLMClient(Protocol):
    def complete_json(self, prompt_name: str, variables: dict[str, Any], format_name: str) -> dict[str, Any]:
        ...


def stable_id(prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def load_prompt(name: str) -> str:
    path = repository_root() / "prompts" / name
    return path.read_text(encoding="utf-8")


def _heuristic_normalize(raw_instruction: str) -> NormalizedIntent:
    text = raw_instruction.strip()
    lowered = text.lower()
    if any(word in lowered for word in ["buy", "shop", "product", "filter", "cart"]):
        domain = "shopping"
    elif any(word in lowered for word in ["flight", "hotel", "travel"]):
        domain = "travel"
    elif any(word in lowered for word in ["map", "route", "nearby"]):
        domain = "map"
    elif any(word in lowered for word in ["issue", "merge", "repository", "gitlab"]):
        domain = "code"
    elif any(word in lowered for word in ["post", "comment", "reddit"]):
        domain = "forum"
    else:
        domain = "other"
    canonical = re.sub(r"\s+", " ", lowered.rstrip("."))
    constraints: list[Constraint] = []
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw_instruction)
    for index, groups in enumerate(quoted):
        value = next((g for g in groups if g), "")
        if value:
            constraints.append(Constraint(slot=f"quoted_value_{index + 1}", value=value, evidence=value))
    return NormalizedIntent(intent=canonical, constraints=constraints, domain_hint=domain)


def normalize_instruction(raw_instruction: str, client: LLMClient | None, record_id: str | None = None) -> NormalizedIntent:
    if client is None:
        intent = _heuristic_normalize(raw_instruction)
    else:
        data = client.complete_json(
            "normalize.txt",
            {"raw_instruction": raw_instruction, "record_id": record_id},
            "normalized_intent",
        )
        intent = NormalizedIntent.from_dict(data)
    validate_data(intent.to_dict(), "normalized_intent")
    return intent


def _heuristic_segment(trajectory: Trajectory) -> list[StageDraft]:
    return stage_drafts_from_trajectory(trajectory)


def segment_trajectory(
    trajectory: Trajectory,
    client: LLMClient | None,
    raw_instruction: str = "",
    intent_json: dict[str, Any] | None = None,
    record_id: str | None = None,
) -> list[StageDraft]:
    if client is None:
        return _heuristic_segment(trajectory)
    summaries = "\n".join(step.summary() for step in trajectory.steps)
    data = client.complete_json(
        "segment.txt",
        {
            "raw_instruction": raw_instruction,
            "intent_json": intent_json or {},
            "step_summaries_with_indices": summaries,
            "record_id": record_id,
        },
        "segment_output",
    )
    drafts = [StageDraft.from_dict(item) for item in data.get("subgoals", [])]
    return drafts


def _stage_text(stage_span: StageSpan) -> str:
    return " ".join(step.summary() for step in stage_span.steps)


def describe_stage(stage_span: StageSpan, client: LLMClient | None, record_id: str | None = None) -> StageNode:
    if client is None:
        draft = stage_span.draft
        node = StageNode(
            stage_id=stage_span.stage_id or stable_id("stage", stage_span.parent_intent_id, draft.name, draft.span.to_list()),
            parent_intent_id=stage_span.parent_intent_id,
            name=draft.name,
            span=draft.span,
            pre_conditions=draft.pre_conditions or ["task-relevant controls are visible"],
            post_conditions=draft.post_conditions or ["stage outcome is visible"],
            source=stage_span.source,
            extra=dict(draft.extra),
        )
    else:
        data = client.complete_json(
            "describe.txt",
            {
                "stage_span": _stage_text(stage_span),
                "stage_draft_json": {
                    "name": stage_span.draft.name,
                    "span": stage_span.draft.span.to_list(),
                    "pre_conditions": stage_span.draft.pre_conditions,
                    "post_conditions": stage_span.draft.post_conditions,
                },
                "record_id": record_id,
            },
            "stage_node",
        )
        data["stage_id"] = data.get("stage_id") or stage_span.stage_id or stable_id("stage", stage_span.parent_intent_id, data.get("name", ""))
        data["parent_intent_id"] = stage_span.parent_intent_id
        data["span"] = data.get("span") or stage_span.draft.span.to_list()
        node = StageNode.from_record({"record_type": "stage", "format_version": FORMAT_VERSION, **data})
    validate_memory_record(node.to_record())
    return node


def _clean_semantic_target(target: dict[str, Any]) -> SemanticDescription:
    clean = {k: v for k, v in target.items() if k not in TRANSFER_FORBIDDEN_KEYS}
    return SemanticDescription.from_dict(clean)


def abstract_step(step: TrajectoryStep, client: LLMClient | None, record_id: str | None = None) -> ActionNode:
    if client is None:
        node = abstract_action_from_step(step)
    else:
        data = client.complete_json(
            "abstract_step.txt",
            {
                "step_json": {
                    "step_index": step.step_index,
                    "operation": step.operation,
                    "argument": step.argument,
                    "target": step.target,
                    "observation": step.observation,
                },
                "record_id": record_id,
            },
            "action_node",
        )
        data.setdefault("action_id", stable_id("act", step.step_index, step.operation, step.target))
        data.setdefault("parent_stage_id", "")
        data.setdefault("operation", step.operation)
        node = ActionNode.from_record({"record_type": "action", "format_version": FORMAT_VERSION, **data})
    return node


def _step_indices(trajectory: Trajectory) -> list[int]:
    return [step.step_index for step in trajectory.steps]


def consistency_check(
    stages: list[StageNode],
    actions: list[ActionNode],
    trajectory: Trajectory | None = None,
) -> ConsistencyReport:
    report = ConsistencyReport(ok=True)
    sorted_stages = sorted(stages, key=lambda s: s.span.start)
    if [s.stage_id for s in stages] != [s.stage_id for s in sorted_stages]:
        report.add_error("stage spans are not ordered")
    for previous, current in zip(sorted_stages, sorted_stages[1:]):
        if current.span.start != previous.span.end + 1:
            report.add_error(f"stage spans are not contiguous: {previous.stage_id} -> {current.stage_id}")
        if current.span.start <= previous.span.end:
            report.add_error(f"stage spans overlap: {previous.stage_id} -> {current.stage_id}")
    covered: list[int] = []
    for stage in sorted_stages:
        if not stage.pre_conditions and not stage.post_conditions:
            report.add_error(f"stage {stage.stage_id} has no observable pre/post conditions")
        covered.extend(stage.span.indices())
    if trajectory is not None:
        expected = _step_indices(trajectory)
        if sorted(covered) != sorted(expected):
            report.add_error(f"stage spans do not cover trajectory exactly: expected {expected}, got {sorted(covered)}")
    if len(covered) != len(set(covered)):
        report.add_error("one or more action steps are covered more than once")
    action_steps = [a.source.step_index for a in actions if a.source and a.source.step_index is not None]
    if trajectory is not None and sorted(action_steps) != sorted(_step_indices(trajectory)):
        report.add_error("action nodes do not cover trajectory steps exactly once")
    for action in actions:
        semantic = action.semantic_description.to_dict()
        for key in TRANSFER_FORBIDDEN_KEYS:
            if key in semantic:
                report.add_error(f"action {action.action_id} exposes forbidden transferable field {key}")
        try:
            validate_memory_record(action.to_record())
        except StructuredOutputError as exc:
            report.add_error(str(exc))
    for stage in stages:
        try:
            validate_memory_record(stage.to_record())
        except StructuredOutputError as exc:
            report.add_error(str(exc))
    return report


def repair_or_merge_segments(
    drafts: list[StageDraft],
    trajectory: Trajectory,
    *_: Any,
    **__: Any,
) -> list[StageDraft]:
    if not trajectory.steps:
        return []
    ordered = sorted(drafts, key=lambda d: d.span.start)
    expected = _step_indices(trajectory)
    if not ordered:
        return _heuristic_segment(trajectory)
    repaired: list[StageDraft] = []
    cursor = expected[0]
    last_step = expected[-1]
    for index, draft in enumerate(ordered):
        start = max(cursor, expected[0])
        end = min(max(draft.span.end, start), last_step)
        if index == len(ordered) - 1:
            end = last_step
        if start > last_step:
            break
        repaired.append(
            StageDraft(
                name=draft.name or "merged stage",
                span=Span(start, end),
                pre_conditions=draft.pre_conditions or ["task-relevant page state is visible"],
                post_conditions=draft.post_conditions or ["stage outcome is visible"],
                extra=dict(draft.extra),
            )
        )
        cursor = end + 1
    if cursor <= last_step and repaired:
        last = repaired[-1]
        repaired[-1] = StageDraft(
            name=last.name,
            span=Span(last.span.start, last_step),
            pre_conditions=last.pre_conditions,
            post_conditions=last.post_conditions,
            extra=dict(last.extra),
        )
    return repaired


def _draft_to_dict(draft: StageDraft) -> dict[str, Any]:
    return {
        "name": draft.name,
        "span": draft.span.to_list(),
        "pre_conditions": draft.pre_conditions,
        "post_conditions": draft.post_conditions,
        **draft.extra,
    }


def repair_segments_with_client(
    drafts: list[StageDraft],
    trajectory: Trajectory,
    report: ConsistencyReport,
    client: LLMClient | None,
    record_id: str | None = None,
) -> list[StageDraft] | None:
    if client is None:
        return None
    try:
        data = client.complete_json(
            "consistency_repair.txt",
            {
                "invalid_json": {"subgoals": [_draft_to_dict(draft) for draft in drafts]},
                "validation_errors": report.errors,
                "trajectory_step_indices": _step_indices(trajectory),
                "record_id": record_id,
            },
            "segment_output",
        )
        validate_data(data, "segment_output")
        return [StageDraft.from_dict(item) for item in data.get("subgoals", [])]
    except Exception as exc:  # pragma: no cover - defensive fallback path
        LOGGER.warning("LLM consistency repair failed for %s: %s", record_id or "episode", exc)
        return None


def _steps_for_span(trajectory: Trajectory, span: Span) -> list[TrajectoryStep]:
    return [step for step in trajectory.steps if span.start <= step.step_index <= span.end]


def _materialize_nodes_from_drafts(
    drafts: list[StageDraft],
    episode: Episode,
    intent_id: str,
    source: SourceMetadata,
    client: LLMClient | None,
    repair_label: str | None = None,
) -> tuple[list[StageNode], list[ActionNode]]:
    stage_nodes: list[StageNode] = []
    action_nodes: list[ActionNode] = []
    record_suffix = f":{repair_label}" if repair_label else ""
    for draft_index, draft in enumerate(drafts, start=1):
        stable_parts: list[Any] = [episode.episode_id, intent_id]
        if repair_label:
            stable_parts.append(repair_label)
        stable_parts.extend([draft_index, draft.name, draft.span.to_list()])
        stage_id = stable_id("stage", *stable_parts)
        span = StageSpan(
            draft=draft,
            parent_intent_id=intent_id,
            source=source,
            steps=_steps_for_span(episode.trajectory, draft.span),
            stage_id=stage_id,
        )
        stage = describe_stage(
            span,
            client=client,
            record_id=f"{episode.dataset}:{episode.episode_id}:describe{record_suffix}:{draft_index}",
        )
        stage.source = source
        validate_memory_record(stage.to_record())
        stage_nodes.append(stage)
        for step in span.steps:
            action = abstract_step(
                step,
                client=client,
                record_id=f"{episode.dataset}:{episode.episode_id}:abstract{record_suffix}:{step.step_index}",
            )
            action.parent_stage_id = stage.stage_id
            action.source = SourceMetadata(dataset=episode.dataset, episode_id=episode.episode_id, step_index=step.step_index)
            action.action_id = stable_id("act", episode.episode_id, stage.stage_id, step.step_index, step.operation, step.target)
            validate_memory_record(action.to_record())
            action_nodes.append(action)
    return stage_nodes, action_nodes


def build_memory(
    episodes: list[Episode],
    config: HMTConfig | dict[str, Any] | None = None,
    client: LLMClient | None = None,
) -> MemoryTree:
    cfg = config if isinstance(config, HMTConfig) else HMTConfig.from_mapping(config)
    _ = cfg
    tree = MemoryTree()
    skipped: list[dict[str, Any]] = []
    for episode in episodes:
        if not episode.success:
            continue
        try:
            normalized = normalize_instruction(episode.raw_instruction, client=client, record_id=f"{episode.dataset}:{episode.episode_id}:normalize")
            intent_id = stable_id("intent", normalized.intent, normalized.domain_hint, [c.to_dict() for c in normalized.constraints])
            source = SourceMetadata(dataset=episode.dataset, episode_id=episode.episode_id)
            intent_node = IntentNode(
                intent_id=intent_id,
                canonical_intent=normalized.intent,
                constraints=normalized.constraints,
                domain_hint=normalized.domain_hint,
                source=source,
            )
            validate_memory_record(intent_node.to_record())
            tree.add_intent(intent_node)

            drafts = segment_trajectory(
                episode.trajectory,
                client=client,
                raw_instruction=episode.raw_instruction,
                intent_json=normalized.to_dict(),
                record_id=f"{episode.dataset}:{episode.episode_id}:segment",
            )
            stage_nodes, action_nodes = _materialize_nodes_from_drafts(drafts, episode, intent_id, source, client)
            report = consistency_check(stage_nodes, action_nodes, episode.trajectory)
            if not report.ok:
                llm_repaired = repair_segments_with_client(
                    drafts,
                    episode.trajectory,
                    report,
                    client,
                    record_id=f"{episode.dataset}:{episode.episode_id}:consistency_repair",
                )
                if llm_repaired is not None:
                    try:
                        candidate_stages, candidate_actions = _materialize_nodes_from_drafts(
                            llm_repaired,
                            episode,
                            intent_id,
                            source,
                            client,
                            repair_label="repair",
                        )
                        candidate_report = consistency_check(candidate_stages, candidate_actions, episode.trajectory)
                        if candidate_report.ok:
                            stage_nodes = candidate_stages
                            action_nodes = candidate_actions
                            report = candidate_report
                        else:
                            LOGGER.warning(
                                "LLM consistency repair remained invalid for %s: %s",
                                episode.episode_id,
                                candidate_report.errors,
                            )
                    except Exception as exc:  # pragma: no cover - defensive fallback path
                        LOGGER.warning("LLM consistency repair materialization failed for %s: %s", episode.episode_id, exc)
                if not report.ok:
                    repaired = repair_or_merge_segments(drafts, episode.trajectory)
                    stage_nodes, action_nodes = _materialize_nodes_from_drafts(
                        repaired,
                        episode,
                        intent_id,
                        source,
                        client,
                        repair_label="repair",
                    )
                    report = consistency_check(stage_nodes, action_nodes, episode.trajectory)
            if not report.ok:
                skipped.append({"episode_id": episode.episode_id, "errors": report.errors})
                continue
            for stage in stage_nodes:
                tree.add_stage(stage)
            for action in action_nodes:
                tree.add_action(action)
        except Exception as exc:  # pragma: no cover - defensive log path
            LOGGER.exception("Skipping episode %s", episode.episode_id)
            skipped.append({"episode_id": episode.episode_id, "errors": [str(exc)]})
    if skipped:
        write_jsonl(repository_root() / "outputs" / "logs" / "construction_failures.jsonl", skipped)
    return tree


@dataclass
class PromptedLLMClient:
    """Tiny helper for tests and offline prompt rendering.

    It does not call an external service; it parses a provided JSON response map.
    Real provider clients should implement the `LLMClient` protocol.
    """

    responses: dict[str, str | dict[str, Any]]

    def complete_json(self, prompt_name: str, variables: dict[str, Any], format_name: str) -> dict[str, Any]:
        _ = load_prompt(prompt_name)
        response = self.responses[prompt_name]
        data = response if isinstance(response, dict) else extract_json_object(response)
        validate_data(data, format_name)
        return data
