from __future__ import annotations

import json
from typing import Any


def dumps_compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON, accepting surrounding prose only for repair workflows."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
