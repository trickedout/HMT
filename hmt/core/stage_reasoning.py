from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable
import math
import re

from hmt.core.condition_match import ConditionMatchResult, condition_overlap, match_stage_conditions
from hmt.core.memory_tree import HMTConfig, Span, StageDraft, StageNode, Trajectory, TrajectoryStep
from hmt.preprocess.page_snapshot import PageSnapshot, normalize_key, normalize_space


class StageKind(str, Enum):
    NAVIGATION = "navigation"
    SEARCH_FORM = "search_form"
    FILTERING = "filtering"
    ITEM_SELECTION = "item_selection"
    CONFIGURATION = "configuration"
    CHECKOUT_OR_CONFIRMATION = "checkout_or_confirmation"
    CONTENT_READING = "content_reading"
    FORM_FILLING = "form_filling"
    SUBMISSION = "submission"
    ANSWER_EXTRACTION = "answer_extraction"
    GENERIC = "generic"


class StageStatus(str, Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


@dataclass
class StepEvidence:
    step_index: int
    operation: str
    label: str = ""
    role: str = ""
    argument: str | None = None
    page_title: str = ""
    observation_summary: str = ""
    tokens: set[str] = field(default_factory=set)
    kind_scores: dict[StageKind, float] = field(default_factory=dict)

    @classmethod
    def from_step(cls, step: TrajectoryStep) -> "StepEvidence":
        snapshot = PageSnapshot.from_observation(step.observation)
        target = step.target or {}
        label = normalize_space(
            target.get("label_or_text") or target.get("visible_text") or target.get("text") or target.get("accessible_name") or target.get("aria_label")
        )
        role = normalize_key(target.get("role") or target.get("tag") or "")
        text = " ".join([step.operation, label, role, step.argument or "", snapshot.summary(max_chars=1000, max_elements=10)])
        evidence = cls(
            step_index=step.step_index,
            operation=normalize_key(step.operation),
            label=label,
            role=role,
            argument=step.argument,
            page_title=snapshot.title,
            observation_summary=snapshot.summary(max_chars=1200, max_elements=12),
            tokens={t for t in re.findall(r"[a-z0-9_./-]+", normalize_key(text)) if len(t) > 1},
        )
        evidence.kind_scores = score_stage_kinds(evidence)
        return evidence

    def text(self) -> str:
        return normalize_space(" ".join([str(self.step_index), self.operation, self.role, self.label, self.argument or "", self.page_title, self.observation_summary]))


@dataclass
class BoundaryCandidate:
    start: int
    end: int
    kind: StageKind
    confidence: float
    evidence: list[StepEvidence] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_span(self) -> Span:
        return Span(self.start, self.end)

    def length(self) -> int:
        return max(0, self.end - self.start + 1)

    def label(self) -> str:
        words = [self.kind.value.replace("_", " ")]
        labels = [e.label for e in self.evidence if e.label]
        if labels:
            words.append(f"around {labels[-1]}")
        return normalize_space(" ".join(words)) or "task stage"


@dataclass
class StageEvaluation:
    stage: StageNode
    status: StageStatus
    score: float
    precondition_score: float = 0.0
    postcondition_score: float = 0.0
    conflict_score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage.stage_id,
            "name": self.stage.name,
            "status": self.status.value,
            "score": self.score,
            "precondition_score": self.precondition_score,
            "postcondition_score": self.postcondition_score,
            "conflict_score": self.conflict_score,
            "evidence": self.evidence,
            "missing": self.missing,
            "decision": self.decision,
        }


@dataclass
class StageGraphNode:
    stage_id: str
    name: str
    span: Span
    kind: StageKind
    pre_conditions: list[str]
    post_conditions: list[str]
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    evidence_steps: list[int] = field(default_factory=list)

    def to_stage_draft(self) -> StageDraft:
        return StageDraft(
            name=self.name,
            span=self.span,
            pre_conditions=self.pre_conditions,
            post_conditions=self.post_conditions,
            extra={"stage_type": self.kind.value, "predecessors": self.predecessors, "successors": self.successors, "evidence_steps": self.evidence_steps},
        )


@dataclass
class StageGraph:
    nodes: list[StageGraphNode]

    def to_drafts(self) -> list[StageDraft]:
        return [node.to_stage_draft() for node in self.nodes]

    def check_contiguity(self, trajectory: Trajectory) -> list[str]:
        errors: list[str] = []
        if not self.nodes and trajectory.steps:
            return ["stage graph is empty"]
        expected = [s.step_index for s in trajectory.steps]
        covered: list[int] = []
        for node in self.nodes:
            covered.extend(node.span.indices())
        if sorted(covered) != sorted(expected):
            errors.append(f"stages cover {sorted(covered)}, expected {expected}")
        if len(covered) != len(set(covered)):
            errors.append("one or more steps are covered by multiple stages")
        for left, right in zip(self.nodes, self.nodes[1:]):
            if right.span.start != left.span.end + 1:
                errors.append(f"non-contiguous transition: {left.name} -> {right.name}")
        return errors


@dataclass
class StageReasoningConfig:
    min_stage_len: int = 1
    max_stage_len: int = 6
    force_contiguous: bool = True
    prefer_semantic_boundaries: bool = True
    merge_low_confidence: bool = True
    low_confidence_threshold: float = 0.35
    max_stages: int = 8

    @classmethod
    def from_hmt_config(cls, config: HMTConfig | None) -> "StageReasoningConfig":
        if config is None:
            return cls()
        return cls(max_stage_len=max(2, config.history_truncation), max_stages=max(3, config.stage_top_k))


def score_stage_kinds(evidence: StepEvidence) -> dict[StageKind, float]:
    tokens = evidence.tokens
    text = " ".join(tokens)
    scores = {kind: 0.0 for kind in StageKind}
    if any(t in tokens for t in ["home", "menu", "section", "category", "subreddit", "repository", "classification", "open"]):
        scores[StageKind.NAVIGATION] += 1.0
    if any(t in tokens for t in ["search", "find", "query", "from", "to", "date", "flight", "hotel"]):
        scores[StageKind.SEARCH_FORM] += 1.0
    if any(t in tokens for t in ["filter", "sort", "brand", "price", "rating", "refine"]):
        scores[StageKind.FILTERING] += 1.0
    if any(t in tokens for t in ["select", "choose", "product", "item", "option", "configuration", "color", "size"]):
        scores[StageKind.ITEM_SELECTION] += 0.8
    if any(t in tokens for t in ["configure", "storage", "protection", "financing", "plan", "option"]):
        scores[StageKind.CONFIGURATION] += 1.0
    if any(t in tokens for t in ["cart", "checkout", "continue", "confirm", "payment", "shipping"]):
        scores[StageKind.CHECKOUT_OR_CONFIRMATION] += 1.0
    if any(t in tokens for t in ["read", "answer", "classification", "section", "content", "list", "subreddit"]):
        scores[StageKind.CONTENT_READING] += 0.8
    if evidence.operation in {"type", "input", "fill"} or any(t in tokens for t in ["field", "textbox", "input", "enter"]):
        scores[StageKind.FORM_FILLING] += 1.0
    if any(t in tokens for t in ["submit", "apply", "go", "done", "save", "search"]):
        scores[StageKind.SUBMISSION] += 0.8
    if evidence.operation in {"stop", "answer", "finish"} or any(t in tokens for t in ["final", "answer", "done"]):
        scores[StageKind.ANSWER_EXTRACTION] += 1.0
    if max(scores.values()) == 0:
        scores[StageKind.GENERIC] = 0.5
    normalized: dict[StageKind, float] = {}
    total = sum(scores.values()) or 1.0
    for kind, score in scores.items():
        normalized[kind] = score / total
    return normalized


def dominant_kind(evidences: Iterable[StepEvidence]) -> StageKind:
    totals = {kind: 0.0 for kind in StageKind}
    for ev in evidences:
        for kind, score in ev.kind_scores.items():
            totals[kind] += score
    return max(totals.items(), key=lambda item: (item[1], item[0].value))[0]


def semantic_distance(left: StepEvidence, right: StepEvidence) -> float:
    a = left.tokens
    b = right.tokens
    lexical = 1.0 if not a or not b else 1.0 - (len(a & b) / len(a | b))
    kind_change = 0.0 if dominant_kind([left]) == dominant_kind([right]) else 0.35
    op_change = 0.0 if left.operation == right.operation else 0.15
    return min(1.0, lexical * 0.5 + kind_change + op_change)


def propose_boundaries(trajectory: Trajectory, config: StageReasoningConfig | None = None) -> list[BoundaryCandidate]:
    cfg = config or StageReasoningConfig()
    if not trajectory.steps:
        return []
    evidences = [StepEvidence.from_step(step) for step in trajectory.steps]
    if len(evidences) == 1:
        kind = dominant_kind(evidences)
        return [BoundaryCandidate(evidences[0].step_index, evidences[0].step_index, kind, 0.8, evidences, ["single-step task"])]
    boundaries: list[int] = [0]
    reasons_by_index: dict[int, list[str]] = {}
    for i in range(1, len(evidences)):
        previous = evidences[i - 1]
        current = evidences[i]
        distance = semantic_distance(previous, current)
        previous_kind = dominant_kind([previous])
        current_kind = dominant_kind([current])
        op_transition = f"{previous.operation}->{current.operation}"
        split = False
        reasons: list[str] = []
        if current_kind != previous_kind and distance >= 0.35:
            split = True
            reasons.append(f"dominant stage kind changes from {previous_kind.value} to {current_kind.value}")
        if op_transition in {"click->type", "click->select", "submit->click", "click->stop", "type->click"} and distance >= 0.25:
            split = True
            reasons.append(f"operation transition {op_transition} suggests a new subgoal")
        if i - boundaries[-1] >= cfg.max_stage_len:
            split = True
            reasons.append("maximum stage length reached")
        if split:
            boundaries.append(i)
            reasons_by_index[i] = reasons
    boundaries.append(len(evidences))
    candidates: list[BoundaryCandidate] = []
    for left, right in zip(boundaries, boundaries[1:]):
        chunk = evidences[left:right]
        if not chunk:
            continue
        kind = dominant_kind(chunk)
        confidence = _boundary_confidence(chunk)
        candidates.append(
            BoundaryCandidate(
                start=chunk[0].step_index,
                end=chunk[-1].step_index,
                kind=kind,
                confidence=confidence,
                evidence=chunk,
                reasons=reasons_by_index.get(left, []) or [f"coherent {kind.value} evidence"],
            )
        )
    return repair_boundary_candidates(candidates, trajectory, cfg)


def _boundary_confidence(evidences: list[StepEvidence]) -> float:
    if not evidences:
        return 0.0
    kind = dominant_kind(evidences)
    kind_scores = [ev.kind_scores.get(kind, 0.0) for ev in evidences]
    avg_kind = sum(kind_scores) / len(kind_scores)
    lexical_coherence = 1.0
    if len(evidences) > 1:
        distances = [semantic_distance(a, b) for a, b in zip(evidences, evidences[1:])]
        lexical_coherence = 1.0 - min(1.0, sum(distances) / len(distances))
    length_prior = min(1.0, math.sqrt(len(evidences)) / 2.5)
    return max(0.0, min(1.0, 0.45 * avg_kind + 0.35 * lexical_coherence + 0.20 * length_prior))


def repair_boundary_candidates(candidates: list[BoundaryCandidate], trajectory: Trajectory, config: StageReasoningConfig) -> list[BoundaryCandidate]:
    if not candidates:
        return candidates
    steps = [step.step_index for step in trajectory.steps]
    by_step: dict[int, StepEvidence] = {ev.step_index: ev for cand in candidates for ev in cand.evidence}
    repaired: list[BoundaryCandidate] = []
    cursor = steps[0]
    for cand in sorted(candidates, key=lambda c: c.start):
        start = max(cursor, cand.start)
        end = min(cand.end, steps[-1])
        if start > end:
            continue
        evidence = [by_step.get(idx) for idx in range(start, end + 1) if by_step.get(idx)]
        if not evidence:
            evidence = [StepEvidence.from_step(step) for step in trajectory.steps if start <= step.step_index <= end]
        repaired.append(BoundaryCandidate(start, end, dominant_kind(evidence), cand.confidence, evidence, list(cand.reasons)))
        cursor = end + 1
    if cursor <= steps[-1]:
        tail = [StepEvidence.from_step(step) for step in trajectory.steps if cursor <= step.step_index <= steps[-1]]
        repaired.append(BoundaryCandidate(cursor, steps[-1], dominant_kind(tail), _boundary_confidence(tail), tail, ["tail repair"] ))
    if config.merge_low_confidence:
        repaired = merge_low_confidence_stages(repaired, config.low_confidence_threshold)
    return repaired[: config.max_stages]


def merge_low_confidence_stages(candidates: list[BoundaryCandidate], threshold: float) -> list[BoundaryCandidate]:
    if len(candidates) <= 1:
        return candidates
    merged: list[BoundaryCandidate] = []
    i = 0
    while i < len(candidates):
        current = candidates[i]
        if current.confidence >= threshold or i == len(candidates) - 1:
            merged.append(current)
            i += 1
            continue
        nxt = candidates[i + 1]
        evidence = current.evidence + nxt.evidence
        merged.append(
            BoundaryCandidate(
                start=current.start,
                end=nxt.end,
                kind=dominant_kind(evidence),
                confidence=max(current.confidence, nxt.confidence, _boundary_confidence(evidence)),
                evidence=evidence,
                reasons=current.reasons + nxt.reasons + ["merged low-confidence boundary"],
            )
        )
        i += 2
    return merged


def condition_from_evidence(kind: StageKind, evidence: list[StepEvidence], is_pre: bool) -> list[str]:
    labels = [e.label for e in evidence if e.label]
    operations = [e.operation for e in evidence]
    text = normalize_key(" ".join(labels + operations + [kind.value]))
    if is_pre:
        if kind == StageKind.NAVIGATION:
            return ["task-relevant site or landing page is visible", "navigation controls or content links are available"]
        if kind == StageKind.SEARCH_FORM:
            return ["task search form or relevant input fields are visible", "task constraints can be entered or selected"]
        if kind == StageKind.FILTERING:
            return ["result list is visible", "filter, sort, or refinement controls are available"]
        if kind == StageKind.ITEM_SELECTION:
            return ["candidate items or options are visible", "the desired item can be identified by task constraints"]
        if kind == StageKind.CONFIGURATION:
            return ["product or workflow configuration options are visible"]
        if kind == StageKind.CHECKOUT_OR_CONFIRMATION:
            return ["the selected item or completed form is ready to continue"]
        if kind == StageKind.CONTENT_READING:
            return ["the relevant content page or section can be opened or read"]
        if kind == StageKind.FORM_FILLING:
            return ["input fields relevant to the current task stage are visible"]
        if kind == StageKind.SUBMISSION:
            return ["all required fields for the current form or filter are filled"]
        if kind == StageKind.ANSWER_EXTRACTION:
            return ["the requested answer or final state is visible"]
        return ["task-relevant page state is visible"]
    else:
        if kind == StageKind.NAVIGATION:
            return ["the next task-relevant page, section, or listing is visible"]
        if kind == StageKind.SEARCH_FORM:
            return ["the search query or form constraints are reflected in the page state"]
        if kind == StageKind.FILTERING:
            return ["the visible results reflect the applied filter or sort option"]
        if kind == StageKind.ITEM_SELECTION:
            return ["the intended item or option is selected or opened"]
        if kind == StageKind.CONFIGURATION:
            return ["the chosen configuration options are visible as selected"]
        if kind == StageKind.CHECKOUT_OR_CONFIRMATION:
            return ["the workflow advances to checkout, confirmation, or the next step"]
        if kind == StageKind.CONTENT_READING:
            return ["the requested content is visible for reading or answering"]
        if kind == StageKind.FORM_FILLING:
            return ["the task-provided values are present in the appropriate fields"]
        if kind == StageKind.SUBMISSION:
            return ["the page updates to results, saved state, or submitted state"]
        if kind == StageKind.ANSWER_EXTRACTION:
            return ["the task can be stopped with the extracted answer"]
        return ["the stage outcome is visible"]


def candidate_to_graph_node(index: int, candidate: BoundaryCandidate, predecessor: str | None, successor: str | None) -> StageGraphNode:
    stage_id = f"stage_auto_{index:02d}_{candidate.kind.value}"
    evidence_labels = [e.label for e in candidate.evidence if e.label]
    if evidence_labels:
        name = f"{candidate.kind.value.replace('_', ' ')}: {evidence_labels[-1][:48]}"
    else:
        name = candidate.kind.value.replace("_", " ")
    return StageGraphNode(
        stage_id=stage_id,
        name=normalize_space(name),
        span=candidate.to_span(),
        kind=candidate.kind,
        pre_conditions=condition_from_evidence(candidate.kind, candidate.evidence, is_pre=True),
        post_conditions=condition_from_evidence(candidate.kind, candidate.evidence, is_pre=False),
        predecessors=[predecessor] if predecessor else [],
        successors=[successor] if successor else [],
        evidence_steps=[ev.step_index for ev in candidate.evidence],
    )


def build_stage_graph(trajectory: Trajectory, config: StageReasoningConfig | None = None) -> StageGraph:
    candidates = propose_boundaries(trajectory, config)
    nodes: list[StageGraphNode] = []
    temp_ids = [f"stage_auto_{i + 1:02d}_{cand.kind.value}" for i, cand in enumerate(candidates)]
    for index, candidate in enumerate(candidates):
        predecessor = temp_ids[index - 1] if index > 0 else None
        successor = temp_ids[index + 1] if index + 1 < len(temp_ids) else None
        node = candidate_to_graph_node(index + 1, candidate, predecessor, successor)
        node.stage_id = temp_ids[index]
        nodes.append(node)
    return StageGraph(nodes)


def stage_drafts_from_trajectory(trajectory: Trajectory, config: HMTConfig | None = None) -> list[StageDraft]:
    graph = build_stage_graph(trajectory, StageReasoningConfig.from_hmt_config(config))
    drafts = graph.to_drafts()
    if not drafts and trajectory.steps:
        return [StageDraft("complete task", Span(trajectory.steps[0].step_index, trajectory.steps[-1].step_index), ["task page is visible"], ["task goal is completed"])]
    return drafts


def evaluate_stage_status(stage: StageNode, observation: dict[str, Any] | str, config: HMTConfig | None = None) -> StageEvaluation:
    cfg = config or HMTConfig()
    summary = PageSnapshot.from_observation(observation).summary(max_chars=2200, max_elements=25)
    match: ConditionMatchResult = match_stage_conditions(stage, summary, cfg.theta_pre, cfg.theta_post_done, cfg.theta_conflict)
    status = StageStatus.UNKNOWN
    if match.has_conflict:
        status = StageStatus.CONFLICT
    elif match.already_completed:
        status = StageStatus.COMPLETED
    elif bool(match.satisfied_preconditions):
        status = StageStatus.READY
    elif match.score > 0:
        status = StageStatus.IN_PROGRESS
    else:
        status = StageStatus.NOT_READY
    evidence = []
    missing = []
    for cond in stage.pre_conditions:
        score = condition_overlap(cond, summary)
        (evidence if score >= cfg.theta_pre else missing).append(cond)
    return StageEvaluation(
        stage=stage,
        status=status,
        score=match.score,
        precondition_score=max([condition_overlap(c, summary) for c in stage.pre_conditions] or [0.0]),
        postcondition_score=max([condition_overlap(c, summary) for c in stage.post_conditions] or [0.0]),
        conflict_score=1.0 if match.has_conflict else 0.0,
        evidence=evidence,
        missing=missing,
        decision=match.decision,
    )


def order_stage_candidates(stages: Iterable[StageNode], observation: dict[str, Any] | str, config: HMTConfig | None = None) -> list[StageEvaluation]:
    evaluations = [evaluate_stage_status(stage, observation, config) for stage in stages]
    status_priority = {
        StageStatus.READY: 4,
        StageStatus.IN_PROGRESS: 3,
        StageStatus.UNKNOWN: 2,
        StageStatus.NOT_READY: 1,
        StageStatus.COMPLETED: 0,
        StageStatus.CONFLICT: -1,
    }
    evaluations.sort(key=lambda item: (-status_priority[item.status], -item.score, item.stage.span.start, item.stage.stage_id))
    return evaluations


def explain_stage_transition(previous: StageNode | None, current: StageNode, next_stage: StageNode | None = None) -> dict[str, Any]:
    return {
        "previous_stage": None if previous is None else {"stage_id": previous.stage_id, "name": previous.name, "post_conditions": previous.post_conditions},
        "current_stage": {"stage_id": current.stage_id, "name": current.name, "pre_conditions": current.pre_conditions, "post_conditions": current.post_conditions},
        "next_stage": None if next_stage is None else {"stage_id": next_stage.stage_id, "name": next_stage.name, "pre_conditions": next_stage.pre_conditions},
        "handoff_logic": "select the first non-completed stage whose pre-conditions are satisfied; skip completed stages and avoid conflicting contexts",
    }
