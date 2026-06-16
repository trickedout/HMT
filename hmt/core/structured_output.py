from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from hmt.utils.io import read_jsonl


class StructuredOutputError(ValueError):
    pass


def _strip_name(name: str) -> str:
    base = Path(str(name)).name
    for suffix in (".format.json", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


@lru_cache(maxsize=32)
def load_format_schema(format_name: str) -> dict[str, Any] | None:
    """Load the JSON format specification used to constrain LLM outputs.

    The files under ``schemas/`` are not data examples. They describe the
    structured objects that Normalize, Segment, Describe, Planner, and Actor are
    expected to return. Runtime validation below remains intentionally light so
    that the code can be used without an extra jsonschema dependency.
    """
    name = _strip_name(format_name)
    root = Path(__file__).resolve().parents[2]
    path = root / "schemas" / f"{name}.schema.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_schema_instructions(format_name: str) -> str:
    schema = load_format_schema(format_name)
    if schema is None:
        return ""
    compact = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return (
        f"Output format constraint for `{_strip_name(format_name)}`. "
        "Return exactly one JSON object and follow this schema:\n"
        f"{compact}"
    )


_REQUIRED: dict[str, tuple[str, ...]] = {
    "normalized_intent": ("intent",),
    "segment_output": ("subgoals",),
    "stage_node": ("name",),
    "action_node": ("operation", "semantic_description"),
    "planner_output": ("confidence",),
    "actor_output": ("operation", "confidence"),
    "memory_record": ("record_type",),
}


def validate_data(data: dict[str, Any], format_name: str) -> None:
    if not isinstance(data, dict):
        raise StructuredOutputError(f"{format_name}: expected JSON object")
    name = _strip_name(format_name)
    required = _REQUIRED.get(name, ())
    missing = [key for key in required if key not in data]
    if missing:
        raise StructuredOutputError(f"{name}: missing field(s): {', '.join(missing)}")
    if name == "segment_output" and not isinstance(data.get("subgoals"), list):
        raise StructuredOutputError("segment_output.subgoals must be a list")
    if name in {"planner_output", "actor_output"}:
        try:
            float(data.get("confidence", 0.0))
        except Exception as exc:
            raise StructuredOutputError(f"{name}.confidence must be numeric") from exc


def validate_memory_record(record: dict[str, Any]) -> None:
    validate_data(record, "memory_record")
    record_type = record.get("record_type")
    if record_type == "intent":
        for key in ["intent_id", "canonical_intent"]:
            if key not in record:
                raise StructuredOutputError(f"intent record missing {key}")
    elif record_type == "stage":
        for key in ["stage_id", "parent_intent_id", "name", "span"]:
            if key not in record:
                raise StructuredOutputError(f"stage record missing {key}")
    elif record_type == "action":
        for key in ["action_id", "parent_stage_id", "operation", "semantic_description"]:
            if key not in record:
                raise StructuredOutputError(f"action record missing {key}")
    else:
        raise StructuredOutputError(f"unsupported record_type: {record_type}")


def validate_memory_file(path: str | Path) -> None:
    for line_no, record in enumerate(read_jsonl(path), start=1):
        try:
            validate_memory_record(record)
        except StructuredOutputError as exc:
            raise StructuredOutputError(f"{path}:{line_no}: {exc}") from exc
