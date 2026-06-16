from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from hmt.core.memory_tree import StageNode

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

LAYOUT_ONLY_TOKENS = {
    "area",
    "box",
    "container",
    "content",
    "div",
    "element",
    "footer",
    "header",
    "layout",
    "main",
    "nav",
    "panel",
    "section",
    "span",
    "wrapper",
}

CONTRADICTION_PAIRS = {
    "visible": {"hidden", "invisible", "missing", "absent"},
    "enabled": {"disabled", "inactive"},
    "open": {"closed"},
    "opened": {"closed"},
    "selected": {"unselected", "deselected"},
    "checked": {"unchecked"},
    "available": {"unavailable"},
    "success": {"error", "failed", "failure"},
    "completed": {"incomplete", "pending"},
    "logged-in": {"logged-out"},
    "logged": {"logged-out"},
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_/:][a-z0-9]+)*")


@dataclass
class ConditionMatchResult:
    score: float
    pre_score: float
    post_score: float
    conflict_score: float
    decision: str
    satisfied_preconditions: list[str] = field(default_factory=list)
    satisfied_postconditions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def already_completed(self) -> bool:
        return self.decision == "already_completed"

    @property
    def has_conflict(self) -> bool:
        return self.decision == "conflict"


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    allowed = set("-_/:|")
    translation = {ord(ch): " " for ch in string.punctuation if ch not in allowed}
    normalized = normalized.translate(translation)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str) -> set[str]:
    normalized = normalize_text(text)
    tokens = set(TOKEN_RE.findall(normalized))
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            expanded.add(token[:-1])
        for sep in ("-", "_", "/", ":"):
            if sep in token:
                for part in token.split(sep):
                    if part:
                        expanded.add(part)
                        if len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
                            expanded.add(part[:-1])
    return {
        token
        for token in expanded
        if token and token not in STOP_WORDS and token not in LAYOUT_ONLY_TOKENS
    }


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def condition_overlap(condition: str, observation_summary: str) -> float:
    return jaccard(tokenize(condition), tokenize(observation_summary))


def _negated_phrase(token: str) -> str:
    return f"not {token.replace('-', ' ')}"


def contradiction_score(condition: str, observation_summary: str) -> tuple[float, list[str]]:
    condition_norm = normalize_text(condition).replace("-", " ")
    observation_norm = normalize_text(observation_summary).replace("-", " ")
    condition_tokens = tokenize(condition)
    observation_tokens = tokenize(observation_summary)
    conflicts: list[str] = []
    for token, opposites in CONTRADICTION_PAIRS.items():
        token_plain = token.replace("-", " ")
        token_in_condition = token in condition_tokens or token_plain in condition_norm
        if not token_in_condition:
            continue
        for opposite in opposites:
            opposite_plain = opposite.replace("-", " ")
            if (
                opposite in observation_tokens
                or opposite_plain in observation_norm
                or _negated_phrase(token) in observation_norm
            ):
                conflicts.append(f"{token} contradicted by {opposite}")
    if not conflicts:
        return 0.0, []
    return min(1.0, len(conflicts) / max(1, len(condition_tokens))), conflicts


def match_stage_conditions(
    stage: StageNode,
    observation_summary: str,
    theta_pre: float = 0.20,
    theta_post_done: float = 0.25,
    theta_conflict: float = 0.10,
) -> ConditionMatchResult:
    pre_scores = [(condition, condition_overlap(condition, observation_summary)) for condition in stage.pre_conditions]
    post_scores = [(condition, condition_overlap(condition, observation_summary)) for condition in stage.post_conditions]
    all_conditions = list(stage.pre_conditions) + list(stage.post_conditions)
    conflicts: list[str] = []
    conflict_score = 0.0
    for condition in all_conditions:
        score, condition_conflicts = contradiction_score(condition, observation_summary)
        conflict_score = max(conflict_score, score)
        conflicts.extend(condition_conflicts)
    pre_score = max((score for _, score in pre_scores), default=0.0)
    post_score = max((score for _, score in post_scores), default=0.0)
    satisfied_pre = [condition for condition, score in pre_scores if score >= theta_pre]
    satisfied_post = [condition for condition, score in post_scores if score >= theta_post_done]
    if conflict_score >= theta_conflict:
        decision = "conflict"
        score = 0.0
    elif satisfied_post:
        decision = "already_completed"
        score = post_score
    elif satisfied_pre:
        decision = "satisfied"
        score = pre_score
    elif pre_score > 0:
        decision = "partial"
        score = pre_score
    else:
        decision = "low_confidence"
        score = 0.0
    return ConditionMatchResult(
        score=score,
        pre_score=pre_score,
        post_score=post_score,
        conflict_score=conflict_score,
        decision=decision,
        satisfied_preconditions=satisfied_pre,
        satisfied_postconditions=satisfied_post,
        conflicts=conflicts,
    )
