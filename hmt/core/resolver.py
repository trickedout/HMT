from __future__ import annotations

"""Environment-specific action resolver for HMT.

HMT keeps memory in an abstract, transferable form and uses a lightweight resolver to map the Actor's semantic decision to the action representation required by a benchmark.  Mind2Web scoring uses candidate IDs, WebArena browser
environments use ID/string commands, and coordinate-based environments can use a
bounding-box center.  This module keeps that layer separate from memory retrieval.
"""

from dataclasses import dataclass, field
from typing import Any, Literal
import math

from hmt.core.actor import ActorOutput

ActionBackend = Literal["mind2web_id", "webarena_id", "browsergym", "coordinate", "abstract"]


@dataclass(frozen=True)
class ResolvedAction:
    operation: str
    backend: ActionBackend
    element_id: str | None = None
    argument: str | None = None
    coordinate: tuple[float, float] | None = None
    serialized: str | None = None
    confidence: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "backend": self.backend,
            "element_id": self.element_id,
            "argument": self.argument,
            "coordinate": list(self.coordinate) if self.coordinate is not None else None,
            "serialized": self.serialized,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": self.metadata,
        }


def candidate_identifier(candidate: dict[str, Any]) -> str | None:
    for key in ["element_id", "backend_node_id", "node_id", "uid", "id", "candidate_id", "ref"]:
        value = candidate.get(key)
        if value not in [None, ""]:
            return str(value)
    raw = candidate.get("raw_candidate")
    if isinstance(raw, dict):
        for key in ["backend_node_id", "node_id", "uid", "id", "candidate_id", "ref"]:
            value = raw.get(key)
            if value not in [None, ""]:
                return str(value)
    return None


def find_candidate_by_id(candidates: list[dict[str, Any]], element_id: str | None) -> dict[str, Any] | None:
    if element_id is None:
        return None
    target = str(element_id)
    for candidate in candidates:
        if candidate_identifier(candidate) == target:
            return candidate
    return None


def bbox_center(candidate: dict[str, Any]) -> tuple[float, float] | None:
    bbox = candidate.get("bbox") or candidate.get("bounding_box") or candidate.get("rect")
    if not bbox and isinstance(candidate.get("raw_candidate"), dict):
        raw = candidate["raw_candidate"]
        bbox = raw.get("bbox") or raw.get("bounding_box") or raw.get("rect")
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        x = bbox.get("x", bbox.get("left"))
        y = bbox.get("y", bbox.get("top"))
        w = bbox.get("width", bbox.get("w"))
        h = bbox.get("height", bbox.get("h"))
        if None not in [x, y, w, h]:
            return (float(x) + float(w) / 2.0, float(y) + float(h) / 2.0)
        if "center" in bbox and isinstance(bbox["center"], (list, tuple)) and len(bbox["center"]) >= 2:
            return (float(bbox["center"][0]), float(bbox["center"][1]))
    if isinstance(bbox, (list, tuple)):
        values = [float(v) for v in bbox[:4]]
        if len(values) == 4:
            x1, y1, x2, y2 = values
            if x2 >= x1 and y2 >= y1:
                return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            return (x1 + x2 / 2.0, y1 + y2 / 2.0)
    return None


def _quote_browsergym(text: str | None) -> str:
    return (text or "").replace("\\", "\\\\").replace("'", "\\'")


def serialize_webarena_id(operation: str, element_id: str | None, argument: str | None) -> str:
    op = operation.lower()
    if op == "click" and element_id:
        return f"click [{element_id}]"
    if op in {"type", "input", "fill"} and element_id:
        return f"type [{element_id}] [{argument or ''}]"
    if op == "select" and element_id:
        return f"select [{element_id}] [{argument or ''}]"
    if op == "hover" and element_id:
        return f"hover [{element_id}]"
    if op == "press":
        return f"press [{argument or 'ENTER'}]"
    if op == "scroll":
        return f"scroll [{argument or 'down'}]"
    return "stop"


def serialize_browsergym(operation: str, element_id: str | None, argument: str | None) -> str:
    op = operation.lower()
    eid = _quote_browsergym(element_id)
    arg = _quote_browsergym(argument)
    if op == "click" and element_id:
        return f"click('{eid}')"
    if op in {"type", "input", "fill"} and element_id:
        return f"fill('{eid}', '{arg}')"
    if op == "select" and element_id:
        return f"select_option('{eid}', '{arg}')"
    if op == "hover" and element_id:
        return f"hover('{eid}')"
    if op == "press":
        return f"press('{arg or 'Enter'}')"
    if op == "scroll":
        return f"scroll('{arg or 'down'}')"
    return "stop()"


def serialize_coordinate(operation: str, coordinate: tuple[float, float] | None, argument: str | None) -> str:
    op = operation.lower()
    if coordinate is None:
        return "stop"
    x, y = coordinate
    if op == "click":
        return f"click [{x:.1f}] [{y:.1f}]"
    if op in {"type", "input", "fill"}:
        return f"type [{x:.1f}] [{y:.1f}] [{argument or ''}]"
    if op == "hover":
        return f"hover [{x:.1f}] [{y:.1f}]"
    return f"{op} [{x:.1f}] [{y:.1f}]"


def resolve_actor_output(
    actor_output: ActorOutput,
    candidates: list[dict[str, Any]] | None = None,
    *,
    backend: ActionBackend = "abstract",
) -> ResolvedAction:
    """Resolve an ActorOutput to a benchmark-specific action representation."""

    candidates = candidates or []
    candidate = find_candidate_by_id(candidates, actor_output.target_element_id)
    element_id = actor_output.target_element_id
    if candidate is not None:
        element_id = candidate_identifier(candidate) or element_id
    coord = bbox_center(candidate) if candidate is not None else None
    serialized: str | None = None
    reason = actor_output.reason
    if backend in {"mind2web_id", "abstract"}:
        serialized = None
    elif backend == "webarena_id":
        serialized = serialize_webarena_id(actor_output.operation, element_id, actor_output.argument)
    elif backend == "browsergym":
        serialized = serialize_browsergym(actor_output.operation, element_id, actor_output.argument)
    elif backend == "coordinate":
        serialized = serialize_coordinate(actor_output.operation, coord, actor_output.argument)
        if coord is None:
            reason = (reason + "; " if reason else "") + "no bounding box available for coordinate resolver"
    return ResolvedAction(
        operation=actor_output.operation,
        backend=backend,
        element_id=element_id,
        argument=actor_output.argument,
        coordinate=coord,
        serialized=serialized,
        confidence=actor_output.confidence,
        reason=reason,
        metadata={
            "matched_descriptor_fields": list(actor_output.matched_descriptor_fields),
            "candidate_found": candidate is not None,
            "coordinate_is_finite": coord is not None and all(math.isfinite(v) for v in coord),
        },
    )


__all__ = [
    "ActionBackend",
    "ResolvedAction",
    "bbox_center",
    "candidate_identifier",
    "find_candidate_by_id",
    "resolve_actor_output",
    "serialize_browsergym",
    "serialize_coordinate",
    "serialize_webarena_id",
]
