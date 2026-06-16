from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import collections
import json
import time

from hmt.core.memory_tree import ActionNode, IntentNode, MemoryTree, StageNode, has_forbidden_transfer_fields
from hmt.core.semantic_abstraction import descriptor_similarity, verify_transferable_action
from hmt.utils.io import write_json, write_jsonl


@dataclass
class MemoryIssue:
    severity: str
    record_type: str
    record_id: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "record_type": self.record_type,
            "record_id": self.record_id,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class MemoryStats:
    num_intents: int
    num_stages: int
    num_actions: int
    stages_per_intent: dict[str, int]
    actions_per_stage: dict[str, int]
    operation_counts: dict[str, int]
    domain_counts: dict[str, int]
    dataset_counts: dict[str, int]
    average_actions_per_stage: float
    average_stages_per_intent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_intents": self.num_intents,
            "num_stages": self.num_stages,
            "num_actions": self.num_actions,
            "stages_per_intent": self.stages_per_intent,
            "actions_per_stage": self.actions_per_stage,
            "operation_counts": self.operation_counts,
            "domain_counts": self.domain_counts,
            "dataset_counts": self.dataset_counts,
            "average_actions_per_stage": self.average_actions_per_stage,
            "average_stages_per_intent": self.average_stages_per_intent,
        }


@dataclass
class MaintenanceReport:
    created_at: float = field(default_factory=time.time)
    issues: list[MemoryIssue] = field(default_factory=list)
    stats_before: dict[str, Any] = field(default_factory=dict)
    stats_after: dict[str, Any] = field(default_factory=dict)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)

    def add_issue(self, severity: str, record_type: str, record_id: str, message: str, **details: Any) -> None:
        self.issues.append(MemoryIssue(severity, record_type, record_id, message, details))

    def add_action(self, action: str, **details: Any) -> None:
        self.actions_taken.append({"action": action, **details})

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "issues": [issue.to_dict() for issue in self.issues],
            "stats_before": self.stats_before,
            "stats_after": self.stats_after,
            "actions_taken": self.actions_taken,
        }


@dataclass
class MaintenanceConfig:
    duplicate_intent_threshold: float = 0.93
    duplicate_stage_threshold: float = 0.90
    duplicate_action_threshold: float = 0.92
    max_actions_per_stage: int = 50
    remove_orphan_records: bool = True
    merge_duplicates: bool = False
    keep_source_debug: bool = False
    min_stage_conditions: int = 1
    min_action_grounding_fields: int = 2


class MemoryInspector:
    def __init__(self, tree: MemoryTree) -> None:
        self.tree = tree

    def stats(self) -> MemoryStats:
        stages_per_intent: dict[str, int] = collections.Counter(stage.parent_intent_id for stage in self.tree.stages.values())
        actions_per_stage: dict[str, int] = collections.Counter(action.parent_stage_id for action in self.tree.actions.values())
        operation_counts: dict[str, int] = collections.Counter(action.operation for action in self.tree.actions.values())
        domain_counts: dict[str, int] = collections.Counter(intent.domain_hint for intent in self.tree.intents.values())
        dataset_counts: dict[str, int] = collections.Counter(
            node.source.dataset for node in list(self.tree.intents.values()) + list(self.tree.stages.values()) + list(self.tree.actions.values()) if node.source
        )
        return MemoryStats(
            num_intents=len(self.tree.intents),
            num_stages=len(self.tree.stages),
            num_actions=len(self.tree.actions),
            stages_per_intent=dict(stages_per_intent),
            actions_per_stage=dict(actions_per_stage),
            operation_counts=dict(operation_counts),
            domain_counts=dict(domain_counts),
            dataset_counts=dict(dataset_counts),
            average_actions_per_stage=(sum(actions_per_stage.values()) / len(actions_per_stage) if actions_per_stage else 0.0),
            average_stages_per_intent=(sum(stages_per_intent.values()) / len(stages_per_intent) if stages_per_intent else 0.0),
        )

    def inspect(self, config: MaintenanceConfig | None = None) -> MaintenanceReport:
        cfg = config or MaintenanceConfig()
        report = MaintenanceReport(stats_before=self.stats().to_dict())
        self._inspect_parent_links(report)
        self._inspect_stage_conditions(report, cfg)
        self._inspect_actions(report, cfg)
        self._inspect_duplicates(report, cfg)
        report.stats_after = self.stats().to_dict()
        return report

    def _inspect_parent_links(self, report: MaintenanceReport) -> None:
        for stage in self.tree.stages.values():
            if stage.parent_intent_id not in self.tree.intents:
                report.add_issue("error", "stage", stage.stage_id, "missing parent intent", parent_intent_id=stage.parent_intent_id)
        for action in self.tree.actions.values():
            if action.parent_stage_id not in self.tree.stages:
                report.add_issue("error", "action", action.action_id, "missing parent stage", parent_stage_id=action.parent_stage_id)

    def _inspect_stage_conditions(self, report: MaintenanceReport, config: MaintenanceConfig) -> None:
        for stage in self.tree.stages.values():
            if len(stage.pre_conditions) < config.min_stage_conditions:
                report.add_issue("warning", "stage", stage.stage_id, "too few pre-conditions")
            if len(stage.post_conditions) < config.min_stage_conditions:
                report.add_issue("warning", "stage", stage.stage_id, "too few post-conditions")
            if not stage.name.strip():
                report.add_issue("error", "stage", stage.stage_id, "empty stage name")

    def _inspect_actions(self, report: MaintenanceReport, config: MaintenanceConfig) -> None:
        for action in self.tree.actions.values():
            if has_forbidden_transfer_fields(action):
                report.add_issue("error", "action", action.action_id, "source-specific field is present in transferable semantic descriptor")
            problems = verify_transferable_action(action)
            for problem in problems:
                report.add_issue("warning", "action", action.action_id, problem)
            semantic = action.semantic_description.to_dict()
            grounding_fields = [k for k in ["label_or_text", "accessible_name", "element_purpose", "parent_context", "form_section", "field_slot", "nearby_text"] if semantic.get(k)]
            if len(grounding_fields) < config.min_action_grounding_fields:
                report.add_issue("warning", "action", action.action_id, "semantic descriptor has few grounding fields", grounding_fields=grounding_fields)
        counts = collections.Counter(action.parent_stage_id for action in self.tree.actions.values())
        for stage_id, count in counts.items():
            if count > config.max_actions_per_stage:
                report.add_issue("warning", "stage", stage_id, "stage has many action nodes", count=count)

    def _inspect_duplicates(self, report: MaintenanceReport, config: MaintenanceConfig) -> None:
        for left, right, score in duplicate_intents(self.tree, config.duplicate_intent_threshold):
            report.add_issue("info", "intent", left.intent_id, "near-duplicate intent", duplicate_of=right.intent_id, score=score)
        for left, right, score in duplicate_stages(self.tree, config.duplicate_stage_threshold):
            report.add_issue("info", "stage", left.stage_id, "near-duplicate stage", duplicate_of=right.stage_id, score=score)
        for left, right, score in duplicate_actions(self.tree, config.duplicate_action_threshold):
            report.add_issue("info", "action", left.action_id, "near-duplicate action", duplicate_of=right.action_id, score=score)


def _token_similarity(left: str, right: str) -> float:
    a = {x for x in left.lower().split() if x}
    b = {x for x in right.lower().split() if x}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _intent_text(intent: IntentNode) -> str:
    return " ".join([intent.canonical_intent, intent.domain_hint, " ".join(f"{c.slot}:{c.value}" for c in intent.constraints)])


def _stage_text(stage: StageNode) -> str:
    return " ".join([stage.name, " ".join(stage.pre_conditions), " ".join(stage.post_conditions), json.dumps(stage.extra, ensure_ascii=False, sort_keys=True)])


def duplicate_intents(tree: MemoryTree, threshold: float) -> list[tuple[IntentNode, IntentNode, float]]:
    intents = sorted(tree.intents.values(), key=lambda node: node.intent_id)
    result: list[tuple[IntentNode, IntentNode, float]] = []
    for i, left in enumerate(intents):
        for right in intents[i + 1 :]:
            score = _token_similarity(_intent_text(left), _intent_text(right))
            if score >= threshold:
                result.append((left, right, score))
    return result


def duplicate_stages(tree: MemoryTree, threshold: float) -> list[tuple[StageNode, StageNode, float]]:
    stages = sorted(tree.stages.values(), key=lambda node: node.stage_id)
    result: list[tuple[StageNode, StageNode, float]] = []
    for i, left in enumerate(stages):
        for right in stages[i + 1 :]:
            if left.parent_intent_id != right.parent_intent_id:
                continue
            score = _token_similarity(_stage_text(left), _stage_text(right))
            if score >= threshold:
                result.append((left, right, score))
    return result


def duplicate_actions(tree: MemoryTree, threshold: float) -> list[tuple[ActionNode, ActionNode, float]]:
    actions = sorted(tree.actions.values(), key=lambda node: node.action_id)
    result: list[tuple[ActionNode, ActionNode, float]] = []
    for i, left in enumerate(actions):
        for right in actions[i + 1 :]:
            if left.parent_stage_id != right.parent_stage_id or left.operation != right.operation:
                continue
            score = descriptor_similarity(left.semantic_description, right.semantic_description)
            if score >= threshold:
                result.append((left, right, score))
    return result


class MemoryMaintainer:
    def __init__(self, tree: MemoryTree, config: MaintenanceConfig | None = None) -> None:
        self.tree = tree
        self.config = config or MaintenanceConfig()
        self.report = MaintenanceReport(stats_before=MemoryInspector(tree).stats().to_dict())

    def run(self) -> tuple[MemoryTree, MaintenanceReport]:
        if self.config.remove_orphan_records:
            self.remove_orphans()
        if self.config.merge_duplicates:
            self.merge_duplicate_intents()
            self.merge_duplicate_stages()
            self.merge_duplicate_actions()
        self.strip_debug_if_needed()
        self.report.stats_after = MemoryInspector(self.tree).stats().to_dict()
        self.report.issues.extend(MemoryInspector(self.tree).inspect(self.config).issues)
        return self.tree, self.report

    def remove_orphans(self) -> None:
        bad_stages = [sid for sid, stage in self.tree.stages.items() if stage.parent_intent_id not in self.tree.intents]
        for stage_id in bad_stages:
            self.tree.stages.pop(stage_id, None)
            self.report.add_action("remove_orphan_stage", stage_id=stage_id)
        valid_stages = set(self.tree.stages)
        bad_actions = [aid for aid, action in self.tree.actions.items() if action.parent_stage_id not in valid_stages]
        for action_id in bad_actions:
            self.tree.actions.pop(action_id, None)
            self.report.add_action("remove_orphan_action", action_id=action_id)

    def merge_duplicate_intents(self) -> None:
        for left, right, score in duplicate_intents(self.tree, self.config.duplicate_intent_threshold):
            if right.intent_id not in self.tree.intents or left.intent_id not in self.tree.intents:
                continue
            for stage in self.tree.stages.values():
                if stage.parent_intent_id == right.intent_id:
                    stage.parent_intent_id = left.intent_id
            self.tree.intents.pop(right.intent_id, None)
            self.report.add_action("merge_duplicate_intent", keep=left.intent_id, remove=right.intent_id, score=score)

    def merge_duplicate_stages(self) -> None:
        for left, right, score in duplicate_stages(self.tree, self.config.duplicate_stage_threshold):
            if right.stage_id not in self.tree.stages or left.stage_id not in self.tree.stages:
                continue
            for action in self.tree.actions.values():
                if action.parent_stage_id == right.stage_id:
                    action.parent_stage_id = left.stage_id
            self.tree.stages.pop(right.stage_id, None)
            self.report.add_action("merge_duplicate_stage", keep=left.stage_id, remove=right.stage_id, score=score)

    def merge_duplicate_actions(self) -> None:
        for left, right, score in duplicate_actions(self.tree, self.config.duplicate_action_threshold):
            if right.action_id not in self.tree.actions or left.action_id not in self.tree.actions:
                continue
            self.tree.actions.pop(right.action_id, None)
            self.report.add_action("merge_duplicate_action", keep=left.action_id, remove=right.action_id, score=score)

    def strip_debug_if_needed(self) -> None:
        if self.config.keep_source_debug:
            return
        for action in self.tree.actions.values():
            if action.source_debug:
                action.source_debug = {}
                self.report.add_action("strip_source_debug", action_id=action.action_id)


def save_memory_report(tree: MemoryTree, output_dir: str | Path, config: MaintenanceConfig | None = None) -> MaintenanceReport:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = MemoryInspector(tree).inspect(config)
    write_json(out / "memory_stats.json", report.stats_after or MemoryInspector(tree).stats().to_dict())
    write_jsonl(out / "memory_issues.jsonl", [issue.to_dict() for issue in report.issues])
    return report
