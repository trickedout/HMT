from __future__ import annotations

from typing import Any

from hmt.core.condition_match import tokenize


def _bbox_area(bbox: dict[str, Any] | None) -> float:
    if not bbox:
        return 1.0
    width = float(bbox.get("width", bbox.get("w", 1)) or 1)
    height = float(bbox.get("height", bbox.get("h", 1)) or 1)
    return max(1.0, width * height)


def _candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key, ""))
        for key in ["role", "visible_text", "accessible_name", "value", "parent_context", "sibling_text"]
        if candidate.get(key)
    )


def rank_salient_elements(
    candidates: list[dict[str, Any]],
    query: str = "",
    previous_element_ids: list[str] | None = None,
    max_elements: int = 30,
) -> list[dict[str, Any]]:
    previous = set(previous_element_ids or [])
    query_tokens = tokenize(query)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for candidate in candidates:
        text_tokens = tokenize(_candidate_text(candidate))
        overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens | text_tokens))
        interactivity = 1.0 if candidate.get("clickable") or candidate.get("editable") else 0.0
        visibility = 1.0 if candidate.get("visible", True) else 0.0
        area_bonus = min(0.2, _bbox_area(candidate.get("bbox")) / 1_000_000)
        proximity = 0.2 if str(candidate.get("element_id")) in previous else 0.0
        score = visibility + interactivity + overlap + proximity + area_bonus
        ranked.append((score, str(candidate.get("element_id", "")), candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in ranked[:max_elements]]
