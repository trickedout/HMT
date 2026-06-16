from __future__ import annotations

"""Observable state abstraction used by the HMT Planner.

HMT uses this step before stage selection.  This module keeps the abstraction explicit instead of burying it inside prompt construction.  The abstraction is intentionally lightweight and benchmark-agnostic: it takes an HTML/accessibility/screenshot-derived observation
plus recent action history and emits a textual state record that can be
matched against stage pre-conditions and post-conditions.

The representation keeps:

* scripts/styles/invisible/layout-only clutter is removed upstream by
  ``hmt.preprocess.dom`` and ``hmt.preprocess.accessibility``;
* the latest ``N_h=6`` actions are retained;
* up to ``N_e=30`` salient elements are retained;
* the Planner receives URL category, state flags, salient controls, and recent
  history rather than raw source-site identifiers.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse
import re

from hmt.preprocess.observation_summary import summarize_observation


_STATE_KEYWORDS = {
    "search": ["search", "query", "find", "lookup"],
    "results": ["result", "results", "listing", "list", "items"],
    "detail": ["detail", "description", "overview", "profile"],
    "cart": ["cart", "basket", "checkout", "order"],
    "form": ["form", "textbox", "input", "field", "dropdown", "select"],
    "login": ["login", "log in", "sign in", "username", "password"],
    "confirmation": ["success", "confirmed", "created", "submitted", "done"],
    "error": ["error", "failed", "invalid", "required", "try again"],
}


@dataclass(frozen=True)
class StateFlag:
    """An observable predicate used for condition matching."""

    name: str
    value: bool
    evidence: str = ""

    def to_text(self) -> str:
        state = "true" if self.value else "false"
        return f"{self.name}={state}" + (f" evidence={self.evidence}" if self.evidence else "")


@dataclass(frozen=True)
class AbstractedElement:
    """Current-page candidate element with transferable, non-source fields."""

    element_id: str
    role: str = ""
    label_or_text: str = ""
    accessible_name: str = ""
    parent_context: str = ""
    sibling_context: str = ""
    relative_position: str = ""
    clickable: bool = False
    editable: bool = False

    @classmethod
    def from_candidate(cls, candidate: dict[str, Any]) -> "AbstractedElement":
        return cls(
            element_id=str(candidate.get("element_id", candidate.get("id", ""))),
            role=str(candidate.get("role", "")),
            label_or_text=str(candidate.get("label_or_text") or candidate.get("visible_text") or candidate.get("text") or ""),
            accessible_name=str(candidate.get("accessible_name") or candidate.get("name") or ""),
            parent_context=str(candidate.get("parent_context") or candidate.get("context") or ""),
            sibling_context=str(candidate.get("sibling_context") or candidate.get("sibling_text") or ""),
            relative_position=str(candidate.get("relative_position") or candidate.get("position") or ""),
            clickable=bool(candidate.get("clickable", False)),
            editable=bool(candidate.get("editable", False)),
        )

    def descriptor_text(self) -> str:
        pieces = [self.role, self.label_or_text, self.accessible_name, self.parent_context, self.sibling_context, self.relative_position]
        return " ".join(piece for piece in pieces if piece).strip()

    def to_planner_line(self) -> str:
        label = self.accessible_name or self.label_or_text
        attrs = []
        if self.clickable:
            attrs.append("clickable")
        if self.editable:
            attrs.append("editable")
        attr_text = ",".join(attrs) if attrs else "static"
        return (
            f"{self.element_id} role={self.role} label={label} "
            f"parent={self.parent_context[:120]} sibling={self.sibling_context[:80]} pos={self.relative_position} attrs={attr_text}"
        ).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AbstractState:
    """Planner-facing state abstraction for one agent step."""

    url: str = ""
    url_domain: str = ""
    url_category: str = "unknown"
    visible_text_excerpt: str = ""
    recent_actions: list[dict[str, Any]] = field(default_factory=list)
    salient_elements: list[AbstractedElement] = field(default_factory=list)
    flags: list[StateFlag] = field(default_factory=list)
    raw_summary_text: str = ""

    def to_summary_text(self) -> str:
        lines: list[str] = []
        if self.url:
            lines.append(f"url: {self.url}")
        lines.append(f"url_category: {self.url_category}")
        if self.visible_text_excerpt:
            lines.append(f"visible_text: {self.visible_text_excerpt}")
        if self.recent_actions:
            lines.append("recent_actions:")
            for action in self.recent_actions:
                op = action.get("operation", "")
                target = action.get("target_element_id", "")
                arg = action.get("argument", "")
                success = action.get("success", action.get("reward", ""))
                lines.append(f"- op={op} target={target} arg={arg} success_or_reward={success}")
        if self.flags:
            lines.append("state_flags:")
            for flag in self.flags:
                lines.append(f"- {flag.to_text()}")
        if self.salient_elements:
            lines.append("salient_elements:")
            for element in self.salient_elements:
                lines.append(f"- {element.to_planner_line()}")
        return "\n".join(lines)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "url_domain": self.url_domain,
            "url_category": self.url_category,
            "recent_actions": self.recent_actions,
            "salient_elements": [element.to_dict() for element in self.salient_elements],
            "flags": [asdict(flag) for flag in self.flags],
            "summary_text": self.to_summary_text(),
            "raw_summary_text": self.raw_summary_text,
        }


def _clean_text(text: Any, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _url_category(url: str) -> str:
    path = urlparse(url).path.lower()
    if not url:
        return "unknown"
    if any(token in path for token in ["search", "results", "listing", "list"]):
        return "search_or_results"
    if any(token in path for token in ["cart", "checkout", "order"]):
        return "cart_or_checkout"
    if any(token in path for token in ["login", "signin", "auth"]):
        return "authentication"
    if any(token in path for token in ["issue", "merge", "repo", "project"]):
        return "code_management"
    if any(token in path for token in ["map", "route", "place"]):
        return "maps"
    if path in {"", "/"}:
        return "home"
    return "detail_or_form"


def _flag_evidence(keyword_group: Iterable[str], text: str) -> tuple[bool, str]:
    lowered = text.lower()
    for word in keyword_group:
        if word in lowered:
            return True, word
    return False, ""


def infer_state_flags(summary_text: str, elements: list[AbstractedElement]) -> list[StateFlag]:
    element_text = " ".join(element.descriptor_text() for element in elements)
    combined = f"{summary_text} {element_text}".lower()
    flags: list[StateFlag] = []
    for name, keywords in _STATE_KEYWORDS.items():
        present, evidence = _flag_evidence(keywords, combined)
        flags.append(StateFlag(name=f"has_{name}_evidence", value=present, evidence=evidence))
    interactive_count = sum(1 for element in elements if element.clickable or element.editable)
    flags.append(StateFlag("has_interactive_candidates", interactive_count > 0, str(interactive_count)))
    flags.append(StateFlag("has_editable_field", any(element.editable for element in elements), "textbox/input candidate"))
    flags.append(StateFlag("has_clickable_control", any(element.clickable for element in elements), "button/link candidate"))
    return flags


def abstract_state(
    observation: dict[str, Any] | str,
    *,
    instruction: str = "",
    recent_actions: list[dict[str, Any]] | None = None,
    max_salient_elements: int = 30,
    history_truncation: int = 6,
) -> AbstractState:
    """Return the explicit ``AbstractState(obs_t)`` record used by HMT-Plan."""

    summary = summarize_observation(
        observation,
        instruction=instruction,
        recent_actions=recent_actions,
        max_salient_elements=max_salient_elements,
        history_truncation=history_truncation,
    )
    url = str(summary.get("url", ""))
    parsed = urlparse(url)
    elements = [AbstractedElement.from_candidate(candidate) for candidate in summary.get("salient_elements", [])]
    if isinstance(observation, dict):
        visible_text = _clean_text(observation.get("text", ""), 500)
    else:
        visible_text = _clean_text(observation, 500)
    flags = infer_state_flags(summary.get("summary_text", ""), elements)
    return AbstractState(
        url=url,
        url_domain=parsed.netloc,
        url_category=_url_category(url),
        visible_text_excerpt=visible_text,
        recent_actions=list(summary.get("recent_actions", [])),
        salient_elements=elements,
        flags=flags,
        raw_summary_text=str(summary.get("summary_text", "")),
    )


__all__ = [
    "AbstractState",
    "AbstractedElement",
    "StateFlag",
    "abstract_state",
    "infer_state_flags",
]
