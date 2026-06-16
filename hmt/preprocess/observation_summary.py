from __future__ import annotations

from typing import Any

from hmt.preprocess.candidates import rank_salient_elements
from hmt.preprocess.dom import extract_dom_candidates


def summarize_observation(
    observation: dict[str, Any] | str,
    instruction: str = "",
    recent_actions: list[dict[str, Any]] | None = None,
    max_salient_elements: int = 30,
    history_truncation: int = 6,
) -> dict[str, Any]:
    if isinstance(observation, str):
        candidates = extract_dom_candidates(observation) if "<" in observation and ">" in observation else []
        raw_text = observation
        url = ""
    else:
        candidates = list(observation.get("candidates", []))
        raw_text = str(observation.get("text", ""))
        url = str(observation.get("url", ""))
    recent = (recent_actions or [])[-history_truncation:]
    previous_ids = [str(action.get("target_element_id", "")) for action in recent if action.get("target_element_id")]
    salient = rank_salient_elements(candidates, query=f"{instruction} {raw_text}", previous_element_ids=previous_ids, max_elements=max_salient_elements)
    lines = []
    if url:
        lines.append(f"url: {url}")
    if raw_text:
        lines.append(f"text: {raw_text[:500]}")
    for element in salient:
        label = element.get("accessible_name") or element.get("visible_text") or element.get("value") or ""
        lines.append(
            f"{element.get('element_id')} role={element.get('role')} label={label} "
            f"parent={element.get('parent_context', '')[:80]}"
        )
    return {
        "url": url,
        "recent_actions": recent,
        "salient_elements": salient,
        "summary_text": "\n".join(lines),
    }
