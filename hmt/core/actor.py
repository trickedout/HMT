from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol

from hmt.core.condition_match import condition_overlap
from hmt.core.retrieval import ScoredNode
from hmt.core.grounding_engine import CandidateGrounder


class ActorLLMClient(Protocol):
    def complete_json(self, prompt_name: str, variables: dict[str, Any], format_name: str) -> dict[str, Any]:
        ...


@dataclass
class ActorOutput:
    operation: str
    target_element_id: str | None
    argument: str | None
    confidence: float
    matched_descriptor_fields: list[str]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "target_element_id": self.target_element_id,
            "argument": self.argument,
            "confidence": self.confidence,
            "matched_descriptor_fields": self.matched_descriptor_fields,
            "reason": self.reason,
        }


def _candidate_text(candidate: dict[str, Any]) -> str:
    def flatten(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            return " ".join(flatten(v) for v in value)
        if isinstance(value, dict):
            return " ".join(flatten(v) for v in value.values())
        return str(value)

    priority_fields = [
        "role",
        "visible_text",
        "text",
        "label_or_text",
        "accessible_name",
        "name",
        "aria_label",
        "placeholder",
        "value",
        "parent_context",
        "ancestor_text",
        "form_section",
        "region",
        "nearby_text",
        "sibling_text",
        "sibling_context",
        "relative_position",
        "state",
        "checked",
        "selected",
        "disabled",
    ]
    fields = [candidate.get(key, "") for key in priority_fields]
    # Include remaining semantic metadata as weak lexical evidence, while leaving raw IDs unusable.
    for key, value in candidate.items():
        if key not in priority_fields and key not in {"backend_node_id", "node_id", "raw_node_id", "css_selector", "selector", "xpath", "coordinates", "bbox", "element_id"}:
            fields.append(value)
    return " ".join(flatten(field) for field in fields if field)


def ground_action(action_hits: list[ScoredNode], candidate_elements: list[dict[str, Any]]) -> ActorOutput:
    if not action_hits:
        return ActorOutput("stop", None, None, 0.0, [], "no action patterns available")
    decision = CandidateGrounder().ground(action_hits[0].node, candidate_elements)
    return ActorOutput(
        operation=decision.operation,
        target_element_id=decision.target_element_id,
        argument=decision.argument,
        confidence=decision.confidence,
        matched_descriptor_fields=decision.matched_descriptor_fields,
        reason=decision.reason,
    )


def llm_ground_action(
    instruction: str,
    stage_json: dict[str, Any],
    action_hits: list[ScoredNode],
    candidate_elements: list[dict[str, Any]],
    client: ActorLLMClient,
) -> ActorOutput:
    data = client.complete_json(
        "actor.txt",
        {
            "instruction": instruction,
            "stage_json": stage_json,
            "action_pattern_jsonl": "\n".join(json.dumps(hit.node.to_record(), ensure_ascii=False) for hit in action_hits),
            "candidate_element_jsonl": "\n".join(json.dumps(candidate, ensure_ascii=False) for candidate in candidate_elements),
        },
        "actor_output",
    )
    return ActorOutput(
        operation=str(data["operation"]),
        target_element_id=data.get("target_element_id"),
        argument=data.get("argument"),
        confidence=float(data["confidence"]),
        matched_descriptor_fields=[str(x) for x in data.get("matched_descriptor_fields", [])],
        reason=str(data.get("reason", "")),
    )
