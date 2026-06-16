from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from hmt.utils.io import read_jsonl, write_jsonl

FORMAT_VERSION = "1.0"
TRANSFER_FORBIDDEN_KEYS = {
    "backend_node_id",
    "node_id",
    "raw_node_id",
    "css_selector",
    "selector",
    "xpath",
    "absolute_coordinates",
    "coordinates",
    "bbox",
    "dom_path",
    "candidate_id",
    "data_reactid",
}


@dataclass
class Constraint:
    slot: str
    value: str
    evidence: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Constraint":
        return cls(
            slot=str(data.get("slot", "")),
            value=str(data.get("value", "")),
            evidence=str(data.get("evidence", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"slot": self.slot, "value": self.value, "evidence": self.evidence}


@dataclass
class NormalizedIntent:
    intent: str
    constraints: list[Constraint] = field(default_factory=list)
    domain_hint: str = "other"
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedIntent":
        return cls(
            intent=str(data.get("intent", "")),
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            domain_hint=str(data.get("domain_hint", "other")),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "intent": self.intent,
            "constraints": [c.to_dict() for c in self.constraints],
            "domain_hint": self.domain_hint,
        }
        if self.notes:
            result["notes"] = self.notes
        return result


@dataclass
class SourceMetadata:
    dataset: str
    episode_id: str
    step_index: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceMetadata":
        step_index = data.get("step_index")
        return cls(
            dataset=str(data.get("dataset", "")),
            episode_id=str(data.get("episode_id", "")),
            step_index=None if step_index is None else int(step_index),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dataset": self.dataset,
            "episode_id": self.episode_id,
        }
        if self.step_index is not None:
            result["step_index"] = self.step_index
        return result


@dataclass
class IntentNode:
    intent_id: str
    canonical_intent: str
    constraints: list[Constraint] = field(default_factory=list)
    domain_hint: str = "other"
    source: SourceMetadata | None = None
    format_version: str = FORMAT_VERSION

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "IntentNode":
        return cls(
            intent_id=str(record["intent_id"]),
            canonical_intent=str(record["canonical_intent"]),
            constraints=[Constraint.from_dict(c) for c in record.get("constraints", [])],
            domain_hint=str(record.get("domain_hint", "other")),
            source=SourceMetadata.from_dict(record["source"]) if record.get("source") else None,
            format_version=str(record.get("format_version", FORMAT_VERSION)),
        )

    def to_record(self) -> dict[str, Any]:
        record = {
            "record_type": "intent",
            "format_version": self.format_version,
            "intent_id": self.intent_id,
            "canonical_intent": self.canonical_intent,
            "constraints": [c.to_dict() for c in self.constraints],
            "domain_hint": self.domain_hint,
        }
        if self.source:
            record["source"] = self.source.to_dict()
        return record


@dataclass
class Span:
    start: int
    end: int

    @classmethod
    def from_any(cls, value: Any) -> "Span":
        if isinstance(value, dict):
            return cls(start=int(value["start"]), end=int(value["end"]))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(start=int(value[0]), end=int(value[1]))
        raise ValueError(f"Invalid span: {value!r}")

    def to_list(self) -> list[int]:
        return [self.start, self.end]

    def indices(self) -> list[int]:
        return list(range(self.start, self.end + 1))


@dataclass
class StageDraft:
    name: str
    span: Span
    pre_conditions: list[str] = field(default_factory=list)
    post_conditions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageDraft":
        known = {"name", "span", "pre_conditions", "post_conditions"}
        return cls(
            name=str(data.get("name", "stage")),
            span=Span.from_any(data.get("span", [0, 0])),
            pre_conditions=[str(x) for x in data.get("pre_conditions", [])],
            post_conditions=[str(x) for x in data.get("post_conditions", [])],
            extra={k: v for k, v in data.items() if k not in known},
        )


@dataclass
class StageSpan:
    draft: StageDraft
    parent_intent_id: str
    source: SourceMetadata
    steps: list["TrajectoryStep"] = field(default_factory=list)
    stage_id: str | None = None


@dataclass
class StageNode:
    stage_id: str
    parent_intent_id: str
    name: str
    span: Span
    pre_conditions: list[str] = field(default_factory=list)
    post_conditions: list[str] = field(default_factory=list)
    source: SourceMetadata | None = None
    format_version: str = FORMAT_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "StageNode":
        known = {
            "record_type",
            "format_version",
            "stage_id",
            "parent_intent_id",
            "name",
            "span",
            "pre_conditions",
            "post_conditions",
            "source",
        }
        return cls(
            stage_id=str(record["stage_id"]),
            parent_intent_id=str(record["parent_intent_id"]),
            name=str(record["name"]),
            span=Span.from_any(record["span"]),
            pre_conditions=[str(x) for x in record.get("pre_conditions", [])],
            post_conditions=[str(x) for x in record.get("post_conditions", [])],
            source=SourceMetadata.from_dict(record["source"]) if record.get("source") else None,
            format_version=str(record.get("format_version", FORMAT_VERSION)),
            extra={k: v for k, v in record.items() if k not in known},
        )

    def to_record(self, include_debug: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": "stage",
            "format_version": self.format_version,
            "stage_id": self.stage_id,
            "parent_intent_id": self.parent_intent_id,
            "name": self.name,
            "span": self.span.to_list(),
            "pre_conditions": self.pre_conditions,
            "post_conditions": self.post_conditions,
        }
        for key, value in self.extra.items():
            if key not in record:
                record[key] = value
        if self.source:
            record["source"] = self.source.to_dict()
        return record


@dataclass
class SemanticDescription:
    role: str = ""
    label_or_text: str = ""
    accessible_name: str = ""
    parent_context: str = ""
    sibling_context: str = ""
    relative_position: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticDescription":
        clean = {k: v for k, v in data.items() if k not in TRANSFER_FORBIDDEN_KEYS}
        known = {
            "role",
            "label_or_text",
            "text",
            "accessible_name",
            "parent_context",
            "context",
            "sibling_context",
            "relative_position",
        }
        return cls(
            role=str(clean.get("role", "")),
            label_or_text=str(clean.get("label_or_text", clean.get("text", ""))),
            accessible_name=str(clean.get("accessible_name", "")),
            parent_context=str(clean.get("parent_context", clean.get("context", ""))),
            sibling_context=str(clean.get("sibling_context", "")),
            relative_position=str(clean.get("relative_position", "")),
            extra={k: v for k, v in clean.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "role": self.role,
            "label_or_text": self.label_or_text,
            "accessible_name": self.accessible_name,
            "parent_context": self.parent_context,
            "sibling_context": self.sibling_context,
            "relative_position": self.relative_position,
        }
        for key, value in self.extra.items():
            if key not in TRANSFER_FORBIDDEN_KEYS and key not in base:
                base[key] = value
        return base

    def text(self) -> str:
        def flatten(value: Any) -> str:
            if isinstance(value, (list, tuple, set)):
                return " ".join(flatten(v) for v in value)
            if isinstance(value, dict):
                return " ".join(flatten(v) for v in value.values())
            return str(value)

        return " ".join(flatten(v) for v in self.to_dict().values() if v)


@dataclass
class ActionNode:
    action_id: str
    parent_stage_id: str
    operation: str
    argument_template: str | None = None
    semantic_description: SemanticDescription = field(default_factory=SemanticDescription)
    source: SourceMetadata | None = None
    source_debug: dict[str, Any] = field(default_factory=dict)
    format_version: str = FORMAT_VERSION

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ActionNode":
        return cls(
            action_id=str(record["action_id"]),
            parent_stage_id=str(record["parent_stage_id"]),
            operation=str(record["operation"]),
            argument_template=record.get("argument_template"),
            semantic_description=SemanticDescription.from_dict(record.get("semantic_description", {})),
            source=SourceMetadata.from_dict(record["source"]) if record.get("source") else None,
            source_debug=dict(record.get("source_debug", {})),
            format_version=str(record.get("format_version", FORMAT_VERSION)),
        )

    def to_record(self, include_debug: bool = False) -> dict[str, Any]:
        record = {
            "record_type": "action",
            "format_version": self.format_version,
            "action_id": self.action_id,
            "parent_stage_id": self.parent_stage_id,
            "operation": self.operation,
            "argument_template": self.argument_template,
            "semantic_description": self.semantic_description.to_dict(),
        }
        if self.source:
            record["source"] = self.source.to_dict()
        if include_debug and self.source_debug:
            record["source_debug"] = self.source_debug
        return record


@dataclass
class TrajectoryStep:
    step_index: int
    observation: dict[str, Any] | str
    operation: str
    argument: str | None = None
    target: dict[str, Any] = field(default_factory=dict)
    action_text: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryStep":
        return cls(
            step_index=int(data.get("step_index", data.get("index", 0))),
            observation=data.get("observation", ""),
            operation=str(data.get("operation", data.get("action", "click"))),
            argument=data.get("argument"),
            target=dict(data.get("target", {})),
            action_text=str(data.get("action_text", "")),
        )

    def summary(self) -> str:
        label = self.target.get("label_or_text") or self.target.get("text") or self.target.get("accessible_name") or ""
        return f"{self.step_index}: {self.operation} {label} {self.argument or ''}".strip()


@dataclass
class Trajectory:
    steps: list[TrajectoryStep]

    @classmethod
    def from_dicts(cls, steps: Iterable[dict[str, Any]]) -> "Trajectory":
        return cls([TrajectoryStep.from_dict(step) for step in steps])


@dataclass
class Episode:
    raw_instruction: str
    trajectory: Trajectory
    dataset: str = "unknown"
    episode_id: str = ""
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        return cls(
            raw_instruction=str(data["raw_instruction"]),
            trajectory=Trajectory.from_dicts(data.get("trajectory", [])),
            dataset=str(data.get("dataset", "unknown")),
            episode_id=str(data.get("episode_id", "")),
            success=bool(data.get("success", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ConsistencyReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.ok = False


@dataclass
class HMTConfig:
    random_seed: int = 3407
    task_top_k: int = 5
    stage_top_k: int = 8
    action_top_k: int = 5
    condition_weight_lambda: float = 0.3
    fallback_margin_delta: float = 0.1
    fallback_abs_confidence_tau: float = 0.15
    history_truncation: int = 6
    max_salient_elements: int = 30
    theta_pre: float = 0.20
    theta_post_done: float = 0.25
    theta_conflict: float = 0.10
    retrieval_backend: str = "qwen"
    allow_model_fallback: bool = False
    inference_mode: str = "deterministic_fallback"
    use_intent_level: bool = True
    use_stage_level: bool = True
    use_action_level: bool = True
    use_pre_conditions: bool = True
    use_post_conditions: bool = True
    use_planner: bool = True
    use_raw_element_ids: bool = False
    confidence_aware_fallback: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "HMTConfig":
        if not data:
            return cls()
        retrieval = data.get("retrieval", {})
        condition = data.get("condition_matching", {})
        ablation = data.get("ablation", {})
        return cls(
            random_seed=int(data.get("random_seed", 3407)),
            task_top_k=int(retrieval.get("task_top_k", 5)),
            stage_top_k=int(retrieval.get("stage_top_k", 8)),
            action_top_k=int(retrieval.get("action_top_k", 5)),
            condition_weight_lambda=float(retrieval.get("condition_weight_lambda", 0.3)),
            fallback_margin_delta=float(retrieval.get("fallback_margin_delta", 0.1)),
            fallback_abs_confidence_tau=float(retrieval.get("fallback_abs_confidence_tau", 0.15)),
            history_truncation=int(retrieval.get("history_truncation", 6)),
            max_salient_elements=int(retrieval.get("max_salient_elements", 30)),
            theta_pre=float(condition.get("theta_pre", 0.20)),
            theta_post_done=float(condition.get("theta_post_done", 0.25)),
            theta_conflict=float(condition.get("theta_conflict", 0.10)),
            retrieval_backend=str(retrieval.get("backend", "qwen")),
            allow_model_fallback=bool(retrieval.get("allow_model_fallback", False)),
            inference_mode=str(data.get("inference", {}).get("mode", "deterministic_fallback")),
            use_intent_level=bool(ablation.get("use_intent_level", True)),
            use_stage_level=bool(ablation.get("use_stage_level", True)),
            use_action_level=bool(ablation.get("use_action_level", True)),
            use_pre_conditions=bool(ablation.get("use_pre_conditions", True)),
            use_post_conditions=bool(ablation.get("use_post_conditions", True)),
            use_planner=bool(ablation.get("use_planner", True)),
            use_raw_element_ids=bool(ablation.get("use_raw_element_ids", False)),
            confidence_aware_fallback=bool(ablation.get("confidence_aware_fallback", True)),
        )


class MemoryTree:
    def __init__(self) -> None:
        self.intents: dict[str, IntentNode] = {}
        self.stages: dict[str, StageNode] = {}
        self.actions: dict[str, ActionNode] = {}

    def add_intent(self, node: IntentNode) -> None:
        self.intents[node.intent_id] = node

    def add_stage(self, node: StageNode) -> None:
        if node.parent_intent_id not in self.intents:
            raise KeyError(f"Missing parent intent: {node.parent_intent_id}")
        self.stages[node.stage_id] = node

    def add_action(self, node: ActionNode) -> None:
        if node.parent_stage_id not in self.stages:
            raise KeyError(f"Missing parent stage: {node.parent_stage_id}")
        self.actions[node.action_id] = node

    def stages_for_intents(self, intent_ids: Iterable[str]) -> list[StageNode]:
        ids = set(intent_ids)
        return [stage for stage in self.stages.values() if stage.parent_intent_id in ids]

    def actions_for_stage(self, stage_id: str) -> list[ActionNode]:
        return [action for action in self.actions.values() if action.parent_stage_id == stage_id]

    def iter_records(self) -> Iterable[dict[str, Any]]:
        for node in self.intents.values():
            yield node.to_record()
        for node in self.stages.values():
            yield node.to_record()
        for node in self.actions.values():
            yield node.to_record()

    def to_records(self) -> list[dict[str, Any]]:
        return list(self.iter_records())

    def save_jsonl(self, path: str | Path) -> None:
        write_jsonl(path, self.iter_records())

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> "MemoryTree":
        tree = cls()
        pending_stages: list[StageNode] = []
        pending_actions: list[ActionNode] = []
        for record in records:
            record_type = record.get("record_type")
            if record_type == "intent":
                tree.add_intent(IntentNode.from_record(record))
            elif record_type == "stage":
                pending_stages.append(StageNode.from_record(record))
            elif record_type == "action":
                pending_actions.append(ActionNode.from_record(record))
            else:
                raise ValueError(f"Unknown memory record type: {record_type}")
        for stage in pending_stages:
            tree.add_stage(stage)
        for action in pending_actions:
            tree.add_action(action)
        return tree

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "MemoryTree":
        return cls.from_records(read_jsonl(path))


def has_forbidden_transfer_fields(action: ActionNode) -> bool:
    data = action.semantic_description.to_dict()
    return any(key in data for key in TRANSFER_FORBIDDEN_KEYS)
