from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json
import time

from hmt.core.memory_tree import ActionNode, Episode, HMTConfig, IntentNode, MemoryTree, StageNode, Trajectory
from hmt.core.semantic_abstraction import descriptor_similarity, verify_transferable_action
from hmt.utils.io import read_jsonl, write_jsonl


@dataclass
class EpisodeTrace:
    task_id: str
    domain: str
    instruction: str
    trajectory: list[dict[str, Any]]
    success: bool
    score: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EpisodeTrace":
        return cls(
            task_id=str(data.get("task_id") or data.get("episode_id") or ""),
            domain=str(data.get("domain") or data.get("site") or ""),
            instruction=str(data.get("instruction") or data.get("raw_instruction") or ""),
            trajectory=list(data.get("trajectory") or data.get("steps") or []),
            success=bool(data.get("success", False)),
            score=None if data.get("score") is None else float(data.get("score")),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_episode(self, dataset: str = "webarena") -> Episode:
        return Episode(
            raw_instruction=self.instruction,
            trajectory=Trajectory.from_dicts(self.trajectory),
            dataset=dataset,
            episode_id=self.task_id,
            success=self.success,
            metadata={"domain": self.domain, **self.metadata},
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "instruction": self.instruction,
            "trajectory": self.trajectory,
            "success": self.success,
            "score": self.score,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
        }


@dataclass
class MemoryInsertionDecision:
    insert: bool
    reason: str
    quality_score: float = 0.0
    duplicate_of: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "insert": self.insert,
            "reason": self.reason,
            "quality_score": self.quality_score,
            "duplicate_of": self.duplicate_of,
            "warnings": self.warnings,
        }


@dataclass
class MemoryUpdateResult:
    task_id: str
    domain: str
    success: bool
    inserted: bool
    decision: MemoryInsertionDecision
    before_counts: dict[str, int]
    after_counts: dict[str, int]
    inserted_record_ids: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "success": self.success,
            "inserted": self.inserted,
            "decision": self.decision.to_dict(),
            "before_counts": self.before_counts,
            "after_counts": self.after_counts,
            "inserted_record_ids": self.inserted_record_ids,
            "timestamp": self.timestamp,
        }


@dataclass
class OnlineMemoryPolicy:
    reset_per_domain: bool = True
    insert_success_only: bool = True
    max_intents_per_domain: int = 500
    max_actions_per_stage: int = 40
    duplicate_threshold: float = 0.92
    min_quality_score: float = 0.45
    retain_failed_traces_for_logs: bool = True
    allow_cross_domain_bootstrap: bool = False
    insertion_dataset_name: str = "webarena_online"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "OnlineMemoryPolicy":
        if not data:
            return cls()
        return cls(
            reset_per_domain=bool(data.get("reset_per_domain", True)),
            insert_success_only=bool(data.get("insert_success_only", True)),
            max_intents_per_domain=int(data.get("max_intents_per_domain", 500)),
            max_actions_per_stage=int(data.get("max_actions_per_stage", 40)),
            duplicate_threshold=float(data.get("duplicate_threshold", 0.92)),
            min_quality_score=float(data.get("min_quality_score", 0.45)),
            retain_failed_traces_for_logs=bool(data.get("retain_failed_traces_for_logs", True)),
            allow_cross_domain_bootstrap=bool(data.get("allow_cross_domain_bootstrap", False)),
            insertion_dataset_name=str(data.get("insertion_dataset_name", "webarena_online")),
        )


class MemoryDuplicateIndex:
    def __init__(self, tree: MemoryTree, duplicate_threshold: float = 0.92) -> None:
        self.tree = tree
        self.duplicate_threshold = duplicate_threshold

    def find_duplicate_intent(self, intent: IntentNode) -> str | None:
        left = " ".join([intent.canonical_intent, intent.domain_hint, " ".join(f"{c.slot}:{c.value}" for c in intent.constraints)])
        for existing in self.tree.intents.values():
            right = " ".join([existing.canonical_intent, existing.domain_hint, " ".join(f"{c.slot}:{c.value}" for c in existing.constraints)])
            if self._text_similarity(left, right) >= self.duplicate_threshold:
                return existing.intent_id
        return None

    def find_duplicate_stage(self, stage: StageNode) -> str | None:
        left = " ".join([stage.name, " ".join(stage.pre_conditions), " ".join(stage.post_conditions), str(stage.extra)])
        for existing in self.tree.stages.values():
            if existing.parent_intent_id != stage.parent_intent_id:
                continue
            right = " ".join([existing.name, " ".join(existing.pre_conditions), " ".join(existing.post_conditions), str(existing.extra)])
            if self._text_similarity(left, right) >= self.duplicate_threshold:
                return existing.stage_id
        return None

    def find_duplicate_action(self, action: ActionNode) -> str | None:
        for existing in self.tree.actions.values():
            if existing.parent_stage_id != action.parent_stage_id:
                continue
            if action.operation != existing.operation:
                continue
            sim = descriptor_similarity(action.semantic_description, existing.semantic_description)
            same_argument = (action.argument_template or "") == (existing.argument_template or "")
            if sim >= self.duplicate_threshold and same_argument:
                return existing.action_id
        return None

    def _text_similarity(self, left: str, right: str) -> float:
        a = {tok for tok in left.lower().split() if tok}
        b = {tok for tok in right.lower().split() if tok}
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


class MemoryQualityGate:
    def __init__(self, policy: OnlineMemoryPolicy) -> None:
        self.policy = policy

    def evaluate(self, trace: EpisodeTrace, new_tree: MemoryTree, current_tree: MemoryTree) -> MemoryInsertionDecision:
        if self.policy.insert_success_only and not trace.success:
            return MemoryInsertionDecision(False, "failed episode excluded by success-only policy", 0.0)
        if not trace.trajectory:
            return MemoryInsertionDecision(False, "empty trajectory", 0.0)
        warnings: list[str] = []
        quality = 0.0
        if new_tree.intents:
            quality += 0.20
        if new_tree.stages:
            quality += 0.25
        if new_tree.actions:
            quality += 0.25
        action_quality = []
        for action in new_tree.actions.values():
            problems = verify_transferable_action(action)
            if problems:
                warnings.extend([f"{action.action_id}: {p}" for p in problems])
            action_quality.append(max(0.0, 1.0 - 0.18 * len(problems)))
        if action_quality:
            quality += 0.30 * (sum(action_quality) / len(action_quality))
        duplicate = self._find_tree_duplicate(new_tree, current_tree)
        if duplicate:
            quality -= 0.15
            warnings.append(f"near-duplicate memory already exists: {duplicate}")
        quality = max(0.0, min(1.0, quality))
        if quality < self.policy.min_quality_score:
            return MemoryInsertionDecision(False, "memory quality below insertion threshold", quality, duplicate_of=duplicate, warnings=warnings)
        if duplicate and quality < self.policy.min_quality_score + 0.20:
            return MemoryInsertionDecision(False, "near-duplicate with insufficient additional value", quality, duplicate_of=duplicate, warnings=warnings)
        return MemoryInsertionDecision(True, "successful trace passed HMT quality gate", quality, duplicate_of=duplicate, warnings=warnings)

    def _find_tree_duplicate(self, new_tree: MemoryTree, current_tree: MemoryTree) -> str | None:
        index = MemoryDuplicateIndex(current_tree, self.policy.duplicate_threshold)
        for intent in new_tree.intents.values():
            duplicate = index.find_duplicate_intent(intent)
            if duplicate:
                return duplicate
        return None


@dataclass
class DomainMemorySession:
    domain: str
    initial_memory: MemoryTree
    policy: OnlineMemoryPolicy
    active_memory: MemoryTree = field(default_factory=MemoryTree)
    update_trace: list[MemoryUpdateResult] = field(default_factory=list)
    failed_traces: list[EpisodeTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.active_memory = clone_memory_tree(self.initial_memory)

    def counts(self) -> dict[str, int]:
        return {
            "intent": len(self.active_memory.intents),
            "stage": len(self.active_memory.stages),
            "action": len(self.active_memory.actions),
        }

    def reset(self) -> None:
        self.active_memory = clone_memory_tree(self.initial_memory)
        self.update_trace.clear()
        self.failed_traces.clear()

    def update_from_trace(self, trace: EpisodeTrace, config: HMTConfig | None = None, client: Any = None) -> MemoryUpdateResult:
        before = self.counts()
        if self.policy.insert_success_only and not trace.success:
            if self.policy.retain_failed_traces_for_logs:
                self.failed_traces.append(trace)
            decision = MemoryInsertionDecision(False, "failed episode excluded by success-only policy", 0.0)
            result = MemoryUpdateResult(trace.task_id, self.domain, trace.success, False, decision, before, before, [])
            self.update_trace.append(result)
            return result
        from hmt.core.construction import build_memory
        episode = trace.to_episode(dataset=self.policy.insertion_dataset_name)
        new_tree = build_memory([episode], config=config, client=client)
        decision = MemoryQualityGate(self.policy).evaluate(trace, new_tree, self.active_memory)
        inserted_ids: list[str] = []
        if decision.insert:
            inserted_ids = self._merge_tree(new_tree)
            self._prune_if_needed()
        after = self.counts()
        result = MemoryUpdateResult(trace.task_id, self.domain, trace.success, decision.insert, decision, before, after, inserted_ids)
        self.update_trace.append(result)
        return result

    def _merge_tree(self, new_tree: MemoryTree) -> list[str]:
        inserted: list[str] = []
        duplicate = MemoryDuplicateIndex(self.active_memory, self.policy.duplicate_threshold)
        intent_id_map: dict[str, str] = {}
        stage_id_map: dict[str, str] = {}
        for intent in new_tree.intents.values():
            existing = duplicate.find_duplicate_intent(intent)
            if existing:
                intent_id_map[intent.intent_id] = existing
            else:
                self.active_memory.add_intent(intent)
                intent_id_map[intent.intent_id] = intent.intent_id
                inserted.append(intent.intent_id)
        for stage in new_tree.stages.values():
            stage.parent_intent_id = intent_id_map.get(stage.parent_intent_id, stage.parent_intent_id)
            existing = duplicate.find_duplicate_stage(stage)
            if existing:
                stage_id_map[stage.stage_id] = existing
            else:
                self.active_memory.add_stage(stage)
                stage_id_map[stage.stage_id] = stage.stage_id
                inserted.append(stage.stage_id)
        for action in new_tree.actions.values():
            action.parent_stage_id = stage_id_map.get(action.parent_stage_id, action.parent_stage_id)
            existing = duplicate.find_duplicate_action(action)
            if existing:
                continue
            self.active_memory.add_action(action)
            inserted.append(action.action_id)
        return inserted

    def _prune_if_needed(self) -> None:
        if len(self.active_memory.intents) <= self.policy.max_intents_per_domain:
            return
        protected = set(self.initial_memory.intents.keys())
        removable = [intent_id for intent_id in self.active_memory.intents if intent_id not in protected]
        overflow = len(self.active_memory.intents) - self.policy.max_intents_per_domain
        for intent_id in removable[:overflow]:
            self._remove_intent(intent_id)

    def _remove_intent(self, intent_id: str) -> None:
        stage_ids = [sid for sid, stage in self.active_memory.stages.items() if stage.parent_intent_id == intent_id]
        action_ids = [aid for aid, action in self.active_memory.actions.items() if action.parent_stage_id in set(stage_ids)]
        for action_id in action_ids:
            self.active_memory.actions.pop(action_id, None)
        for stage_id in stage_ids:
            self.active_memory.stages.pop(stage_id, None)
        self.active_memory.intents.pop(intent_id, None)

    def save(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.active_memory.save_jsonl(out / "T_mem_online.jsonl")
        write_jsonl(out / "online_memory_trace.jsonl", [item.to_record() for item in self.update_trace])
        if self.failed_traces:
            write_jsonl(out / "failed_episode_traces.jsonl", [trace.to_record() for trace in self.failed_traces])


class WebArenaOnlineMemoryManager:
    def __init__(self, base_memory: MemoryTree | None = None, policy: OnlineMemoryPolicy | None = None) -> None:
        self.base_memory = base_memory or MemoryTree()
        self.policy = policy or OnlineMemoryPolicy()
        self.sessions: dict[str, DomainMemorySession] = {}

    def session_for_domain(self, domain: str) -> DomainMemorySession:
        if self.policy.reset_per_domain or domain not in self.sessions:
            if domain not in self.sessions:
                initial = self.base_memory if self.policy.allow_cross_domain_bootstrap else MemoryTree()
                self.sessions[domain] = DomainMemorySession(domain=domain, initial_memory=initial, policy=self.policy)
            return self.sessions[domain]
        return self.sessions[domain]

    def reset_domain(self, domain: str) -> None:
        if domain in self.sessions:
            self.sessions[domain].reset()
        else:
            self.sessions[domain] = DomainMemorySession(domain=domain, initial_memory=MemoryTree(), policy=self.policy)

    def update(self, trace: EpisodeTrace | dict[str, Any], config: HMTConfig | None = None, client: Any = None) -> MemoryUpdateResult:
        episode_trace = trace if isinstance(trace, EpisodeTrace) else EpisodeTrace.from_mapping(trace)
        session = self.session_for_domain(episode_trace.domain)
        return session.update_from_trace(episode_trace, config=config, client=client)

    def save_all(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for domain, session in self.sessions.items():
            session.save(root / domain)


def clone_memory_tree(tree: MemoryTree) -> MemoryTree:
    return MemoryTree.from_records([json.loads(json.dumps(record, ensure_ascii=False)) for record in tree.iter_records()])


def load_episode_traces(path: str | Path) -> list[EpisodeTrace]:
    return [EpisodeTrace.from_mapping(row) for row in read_jsonl(path)]
