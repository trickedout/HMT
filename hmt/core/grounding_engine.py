from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import math
import re

from hmt.core.condition_match import condition_overlap
from hmt.core.memory_tree import ActionNode, SemanticDescription
from hmt.preprocess.page_snapshot import PageSnapshot, normalize_candidate_elements, normalize_key, normalize_space

OPERATION_ROLE_COMPATIBILITY = {
    "click": {"button", "link", "tab", "menuitem", "option", "checkbox", "radio", "switch", "summary", "generic"},
    "type": {"textbox", "searchbox", "combobox", "spinbutton", "input", "textarea", "generic"},
    "select": {"combobox", "listbox", "option", "radio", "checkbox", "switch", "generic"},
    "hover": {"link", "button", "menuitem", "generic"},
    "scroll": {"main", "region", "generic"},
    "stop": {"generic"},
}
FIELD_WEIGHTS = {
    "role": 0.12,
    "label_or_text": 0.20,
    "accessible_name": 0.18,
    "element_purpose": 0.12,
    "parent_context": 0.12,
    "sibling_context": 0.06,
    "nearby_text": 0.08,
    "form_section": 0.08,
    "region": 0.05,
    "field_slot": 0.08,
    "expected_state_change": 0.04,
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [normalize_space(v) for v in value if normalize_space(v)]
    return [normalize_space(value)] if normalize_space(value) else []


def _candidate_text(candidate: dict[str, Any]) -> str:
    fields = [
        "role", "visible_text", "text", "label_or_text", "accessible_name", "name", "aria_label",
        "placeholder", "value", "parent_context", "ancestor_text", "form_section", "region", "nearby_text",
        "sibling_text", "sibling_context", "relative_position", "state", "title", "alt",
    ]
    parts = []
    for key in fields:
        value = candidate.get(key)
        if isinstance(value, dict):
            parts.extend(f"{k}:{v}" for k, v in value.items())
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(v) for v in value)
        elif value:
            parts.append(str(value))
    return normalize_space(" ".join(parts))


def _candidate_field(candidate: dict[str, Any], *keys: str) -> str:
    return normalize_space(" ".join(str(candidate.get(k, "")) for k in keys if candidate.get(k)))


def _tokens(text: Any) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9_./-]+", normalize_key(text)) if len(x) > 1}


@dataclass
class CandidateScoreBreakdown:
    element_id: str
    total: float = 0.0
    operation_score: float = 0.0
    field_scores: dict[str, float] = field(default_factory=dict)
    disambiguator_score: float = 0.0
    negative_penalty: float = 0.0
    state_penalty: float = 0.0
    tie_break_score: float = 0.0
    matched_fields: list[str] = field(default_factory=list)
    rejected_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "total": round(self.total, 6),
            "operation_score": round(self.operation_score, 6),
            "field_scores": {k: round(v, 6) for k, v in self.field_scores.items()},
            "disambiguator_score": round(self.disambiguator_score, 6),
            "negative_penalty": round(self.negative_penalty, 6),
            "state_penalty": round(self.state_penalty, 6),
            "tie_break_score": round(self.tie_break_score, 6),
            "matched_fields": self.matched_fields,
            "rejected_reasons": self.rejected_reasons,
        }


@dataclass
class GroundingDecision:
    operation: str
    target_element_id: str | None
    argument: str | None
    confidence: float
    matched_descriptor_fields: list[str]
    selected_action_id: str | None = None
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    grounding_trace: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "target_element_id": self.target_element_id,
            "argument": self.argument,
            "confidence": self.confidence,
            "matched_descriptor_fields": self.matched_descriptor_fields,
            "selected_action_id": self.selected_action_id,
            "rejected_candidates": self.rejected_candidates,
            "grounding_trace": self.grounding_trace,
            "reason": self.reason,
        }


@dataclass
class GroundingConfig:
    min_confidence: float = 0.05
    role_mismatch_penalty: float = 0.25
    disabled_penalty: float = 0.45
    negative_constraint_penalty: float = 0.35
    disambiguator_weight: float = 0.18
    prefer_interactive: bool = True
    max_trace_candidates: int = 12
    use_candidate_salience: bool = True


class CandidateGrounder:
    def __init__(self, config: GroundingConfig | None = None) -> None:
        self.config = config or GroundingConfig()

    def ground(self, action: ActionNode, candidate_elements: Iterable[dict[str, Any]], observation: dict[str, Any] | str | None = None) -> GroundingDecision:
        candidates = self._normalize_candidates(candidate_elements, observation)
        if action.operation == "stop":
            return GroundingDecision("stop", None, action.argument_template, 1.0, [], action.action_id, reason="stored action is stop")
        if not candidates:
            return GroundingDecision(action.operation, None, action.argument_template, 0.0, [], action.action_id, reason="no candidate elements available")
        descriptor = action.semantic_description.to_dict()
        scored = [self.score_candidate(action.operation, descriptor, candidate, index) for index, candidate in enumerate(candidates)]
        scored.sort(key=lambda item: (-item.total, item.element_id))
        best = scored[0]
        best_candidate = next((c for c in candidates if str(c.get("element_id")) == best.element_id), candidates[0])
        confidence = self._calibrate_confidence(best, scored[1] if len(scored) > 1 else None)
        rejected = [s.to_dict() for s in scored[1 : self.config.max_trace_candidates + 1]]
        trace = [s.to_dict() for s in scored[: self.config.max_trace_candidates]]
        if confidence < self.config.min_confidence:
            return GroundingDecision(
                action.operation,
                None,
                action.argument_template,
                confidence,
                best.matched_fields,
                action.action_id,
                rejected_candidates=rejected,
                grounding_trace=trace,
                reason="best candidate below grounding confidence threshold",
            )
        return GroundingDecision(
            action.operation,
            str(best_candidate.get("element_id")),
            self._resolve_argument(action, best_candidate),
            confidence,
            best.matched_fields,
            action.action_id,
            rejected_candidates=rejected,
            grounding_trace=trace,
            reason="selected candidate with highest semantic grounding score",
        )

    def _normalize_candidates(self, candidate_elements: Iterable[dict[str, Any]], observation: dict[str, Any] | str | None = None) -> list[dict[str, Any]]:
        raw = [dict(c) for c in candidate_elements if isinstance(c, dict)]
        if observation is not None:
            snapshot = PageSnapshot.from_observation(observation, raw)
            return snapshot.candidate_dicts(max_elements=max(len(raw), 30), interactive_only=False)
        return normalize_candidate_elements(raw, max_elements=len(raw) or 0)

    def score_candidate(self, operation: str, descriptor: dict[str, Any], candidate: dict[str, Any], index: int = 0) -> CandidateScoreBreakdown:
        element_id = str(candidate.get("element_id", index))
        score = CandidateScoreBreakdown(element_id=element_id)
        score.operation_score = self._operation_compatibility(operation, descriptor, candidate)
        weighted = 0.0
        weight_sum = 0.0
        for field, weight in self._field_weight_order(descriptor):
            dvalue = descriptor.get(field)
            if not dvalue:
                continue
            cvalue = self._candidate_value_for_field(field, candidate)
            if not cvalue:
                continue
            if field == "role":
                field_score = 1.0 if normalize_key(dvalue) == normalize_key(cvalue) else 0.0
            elif isinstance(dvalue, list):
                field_score = max([condition_overlap(str(item), cvalue) for item in dvalue] or [0.0])
            else:
                field_score = condition_overlap(str(dvalue), cvalue)
            score.field_scores[field] = field_score
            if field_score > 0:
                score.matched_fields.append(field)
            weighted += weight * field_score
            weight_sum += weight
        if weight_sum:
            weighted /= weight_sum
        score.disambiguator_score = self._score_disambiguators(descriptor, candidate)
        score.negative_penalty = self._score_negative_constraints(descriptor, candidate)
        score.state_penalty = self._state_penalty(candidate)
        score.tie_break_score = self._tie_break_score(candidate, index)
        total = (
            0.22 * score.operation_score
            + 0.58 * weighted
            + self.config.disambiguator_weight * score.disambiguator_score
            + score.tie_break_score
            - score.negative_penalty
            - score.state_penalty
        )
        score.total = max(0.0, min(1.0, total))
        if score.operation_score == 0:
            score.rejected_reasons.append("operation/role mismatch")
        if score.state_penalty > 0:
            score.rejected_reasons.append("candidate appears disabled or hidden")
        if score.negative_penalty > 0:
            score.rejected_reasons.append("negative constraint matched")
        return score

    def _field_weight_order(self, descriptor: dict[str, Any]) -> list[tuple[str, float]]:
        priority = descriptor.get("grounding_priority") or []
        ordered: list[tuple[str, float]] = []
        used: set[str] = set()
        for field in priority:
            if field in FIELD_WEIGHTS:
                ordered.append((field, FIELD_WEIGHTS[field] * 1.25))
                used.add(field)
        for field, weight in FIELD_WEIGHTS.items():
            if field not in used:
                ordered.append((field, weight))
        return ordered

    def _candidate_value_for_field(self, field: str, candidate: dict[str, Any]) -> str:
        if field == "role":
            return _candidate_field(candidate, "role", "tag")
        if field == "label_or_text":
            return _candidate_field(candidate, "label_or_text", "visible_text", "text", "value", "title")
        if field == "accessible_name":
            return _candidate_field(candidate, "accessible_name", "name", "aria_label", "label_or_text")
        if field == "element_purpose":
            return _candidate_text(candidate)
        if field == "parent_context":
            return _candidate_field(candidate, "parent_context", "ancestor_text", "form_section", "region")
        if field == "sibling_context":
            return _candidate_field(candidate, "sibling_context", "sibling_text", "nearby_text")
        if field == "nearby_text":
            return _candidate_field(candidate, "nearby_text", "sibling_text", "ancestor_text")
        if field == "form_section":
            return _candidate_field(candidate, "form_section", "parent_context", "region")
        if field == "region":
            return _candidate_field(candidate, "region", "parent_context", "form_section")
        if field == "field_slot":
            return _candidate_text(candidate)
        if field == "expected_state_change":
            return _candidate_text(candidate)
        return _candidate_text(candidate)

    def _operation_compatibility(self, operation: str, descriptor: dict[str, Any], candidate: dict[str, Any]) -> float:
        op = normalize_key(operation)
        role = normalize_key(candidate.get("role") or candidate.get("tag") or "generic")
        expected = OPERATION_ROLE_COMPATIBILITY.get(op, {"generic"})
        if role in expected:
            base = 1.0
        elif role == "generic" or "generic" in expected:
            base = 0.65
        else:
            base = max(0.0, 1.0 - self.config.role_mismatch_penalty)
        if self.config.prefer_interactive and candidate.get("salience"):
            base += min(0.08, float(candidate.get("salience", 0)) / 50.0)
        if op == "type" and candidate.get("state", {}).get("readonly"):
            base *= 0.5
        return max(0.0, min(1.0, base))

    def _score_disambiguators(self, descriptor: dict[str, Any], candidate: dict[str, Any]) -> float:
        disambiguators = _as_list(descriptor.get("disambiguators"))
        if not disambiguators:
            return 0.0
        text = _candidate_text(candidate)
        scores = [condition_overlap(item, text) for item in disambiguators]
        return sum(scores) / len(scores)

    def _score_negative_constraints(self, descriptor: dict[str, Any], candidate: dict[str, Any]) -> float:
        constraints = _as_list(descriptor.get("negative_constraints"))
        if not constraints:
            return 0.0
        text = _candidate_text(candidate)
        max_overlap = max([condition_overlap(item, text) for item in constraints] or [0.0])
        return self.config.negative_constraint_penalty * max_overlap

    def _state_penalty(self, candidate: dict[str, Any]) -> float:
        state = candidate.get("state") or {}
        penalty = 0.0
        if state.get("disabled") or normalize_key(candidate.get("disabled")) == "true":
            penalty += self.config.disabled_penalty
        if state.get("visible") is False:
            penalty += 0.35
        return penalty

    def _tie_break_score(self, candidate: dict[str, Any], index: int) -> float:
        salience = candidate.get("salience")
        salience_bonus = min(0.04, float(salience) / 100.0) if isinstance(salience, (int, float)) else 0.0
        position_bonus = max(0.0, 0.03 - index * 0.001)
        return salience_bonus + position_bonus

    def _calibrate_confidence(self, best: CandidateScoreBreakdown, second: CandidateScoreBreakdown | None) -> float:
        margin = best.total - (second.total if second else 0.0)
        matched_bonus = min(0.15, 0.03 * len(best.matched_fields))
        confidence = best.total * 0.78 + max(0.0, margin) * 0.15 + matched_bonus
        if best.rejected_reasons:
            confidence *= 0.92
        return max(0.0, min(1.0, confidence))

    def _resolve_argument(self, action: ActionNode, candidate: dict[str, Any]) -> str | None:
        if action.argument_template:
            return action.argument_template
        descriptor = action.semantic_description.to_dict()
        source = descriptor.get("value_source")
        if source and isinstance(source, str) and ":" in source:
            return source.split(":", 1)[1].strip()
        return None


def ground_action_patterns(action_hits: Iterable[Any], candidate_elements: Iterable[dict[str, Any]], observation: dict[str, Any] | str | None = None) -> GroundingDecision:
    hits = list(action_hits)
    if not hits:
        return GroundingDecision("stop", None, None, 0.0, [], None, reason="no action hits")
    action = hits[0].node if hasattr(hits[0], "node") else hits[0]
    return CandidateGrounder().ground(action, candidate_elements, observation=observation)


def descriptor_to_candidate_query(description: SemanticDescription | dict[str, Any]) -> str:
    data = description.to_dict() if isinstance(description, SemanticDescription) else description
    fields = ["role", "label_or_text", "accessible_name", "element_purpose", "parent_context", "form_section", "field_slot", "nearby_text"]
    return normalize_space(" ".join(str(data.get(field, "")) for field in fields if data.get(field)))
