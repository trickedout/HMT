from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import re

from hmt.core.memory_tree import ActionNode, SemanticDescription, SourceMetadata, TrajectoryStep, TRANSFER_FORBIDDEN_KEYS
import hashlib
import json


def stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return f"{prefix}_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
from hmt.preprocess.page_snapshot import ElementSnapshot, PageSnapshot, normalize_candidate_elements, normalize_key, normalize_space

FIELD_SLOT_WORDS = {
    "origin": ["from", "origin", "departure", "pickup", "source"],
    "destination": ["to", "destination", "arrival", "dropoff"],
    "date": ["date", "depart", "return", "calendar", "day"],
    "search_query": ["search", "query", "keyword", "find"],
    "quantity": ["quantity", "qty", "count", "number"],
    "price": ["price", "budget", "cost"],
    "color": ["color", "colour"],
    "size": ["size"],
    "account": ["username", "email", "account", "login"],
}
OPERATION_ALIASES = {
    "click": {"click", "press", "tap", "select", "choose", "open", "submit", "check", "uncheck"},
    "type": {"type", "input", "fill", "enter", "write", "set_value", "send_keys"},
    "select": {"select", "choose_option", "dropdown", "pick"},
    "hover": {"hover", "move"},
    "scroll": {"scroll"},
    "stop": {"stop", "finish", "answer", "done"},
}
SEMANTIC_FORBIDDEN = set(TRANSFER_FORBIDDEN_KEYS) | {"id", "backend_id", "path", "x", "y", "width", "height", "left", "top", "right", "bottom"}


def canonical_operation(operation: str, target: dict[str, Any] | None = None) -> str:
    op = normalize_key(operation)
    for canonical, aliases in OPERATION_ALIASES.items():
        if op in aliases or any(alias in op.split() for alias in aliases):
            if canonical == "click" and target:
                role = normalize_key(target.get("role") or target.get("tag"))
                typ = normalize_key((target.get("attributes") or {}).get("type") or target.get("type"))
                if role in {"textbox", "searchbox", "combobox"} and typ not in {"submit", "button"} and operation in {"fill", "type", "input"}:
                    return "type"
            return canonical
    if "click" in op or "press" in op:
        return "click"
    if "type" in op or "fill" in op or "input" in op:
        return "type"
    if "select" in op or "option" in op:
        return "select"
    return op or "click"


def _token_set(text: Any) -> set[str]:
    text = normalize_key(text)
    return {tok for tok in re.findall(r"[\w.-]+", text) if tok and len(tok) > 1}


def _overlap(left: Any, right: Any) -> float:
    a = _token_set(left)
    b = _token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class CandidateSemantics:
    element_id: str = ""
    role: str = ""
    label_or_text: str = ""
    accessible_name: str = ""
    element_purpose: str = ""
    parent_context: str = ""
    sibling_context: str = ""
    nearby_text: str = ""
    form_section: str = ""
    region: str = ""
    relative_position: str = ""
    field_slot: str = ""
    value_source: str = ""
    expected_state_change: str = ""
    disambiguators: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    grounding_priority: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_target(cls, target: dict[str, Any], operation: str, argument: str | None = None, observation: Any = None) -> "CandidateSemantics":
        clean = {k: v for k, v in (target or {}).items() if k not in SEMANTIC_FORBIDDEN}
        normalized_candidates = normalize_candidate_elements([clean], max_elements=1)
        candidate = normalized_candidates[0] if normalized_candidates else dict(clean)
        role = normalize_key(candidate.get("role") or clean.get("role") or clean.get("tag") or "")
        label = normalize_space(candidate.get("label_or_text") or candidate.get("visible_text") or clean.get("label_or_text") or clean.get("text"))
        accessible = normalize_space(candidate.get("accessible_name") or clean.get("accessible_name") or clean.get("aria_label"))
        parent_context = normalize_space(candidate.get("parent_context") or clean.get("parent_context") or clean.get("ancestor_text"))
        sibling = normalize_space(candidate.get("sibling_text") or candidate.get("sibling_context") or clean.get("sibling_context"))
        nearby = normalize_space(candidate.get("nearby_text") or clean.get("nearby_text"))
        section = normalize_space(candidate.get("form_section") or clean.get("form_section") or infer_form_section(parent_context, nearby, label))
        region = normalize_space(candidate.get("region") or infer_region(parent_context, nearby, section, label))
        field_slot = infer_field_slot(" ".join([label, accessible, parent_context, nearby, section]), operation, argument)
        purpose = infer_element_purpose(role, label or accessible, operation, field_slot, section, region)
        expected = infer_expected_state_change(canonical_operation(operation, target), role, label or accessible, argument, field_slot, section, region)
        return cls(
            element_id=str(candidate.get("element_id", "")),
            role=role or "generic",
            label_or_text=label,
            accessible_name=accessible,
            element_purpose=purpose,
            parent_context=parent_context,
            sibling_context=sibling,
            nearby_text=nearby,
            form_section=section,
            region=region,
            relative_position=normalize_space(candidate.get("relative_position") or clean.get("relative_position")),
            field_slot=field_slot,
            value_source=infer_value_source(argument, field_slot),
            expected_state_change=expected,
            disambiguators=derive_disambiguators(role, label or accessible, parent_context, sibling, nearby, section, region),
            negative_constraints=derive_negative_constraints(role, label or accessible, operation, section, region),
            grounding_priority=derive_grounding_priority(role, operation, field_slot),
            state=dict(candidate.get("state") or clean.get("state") or {}),
        )

    def to_semantic_description(self) -> SemanticDescription:
        base = {
            "role": self.role,
            "label_or_text": self.label_or_text,
            "accessible_name": self.accessible_name,
            "element_purpose": self.element_purpose,
            "parent_context": self.parent_context,
            "sibling_context": self.sibling_context,
            "nearby_text": self.nearby_text,
            "form_section": self.form_section,
            "region": self.region,
            "relative_position": self.relative_position,
            "field_slot": self.field_slot,
            "value_source": self.value_source,
            "expected_state_change": self.expected_state_change,
            "disambiguators": self.disambiguators,
            "negative_constraints": self.negative_constraints,
            "grounding_priority": self.grounding_priority,
            "state": self.state,
        }
        return SemanticDescription.from_dict({k: v for k, v in base.items() if v not in ("", None, [], {})})


@dataclass
class SemanticActionDraft:
    operation: str
    argument_template: str | None
    semantics: CandidateSemantics
    source_step_index: int
    raw_action_text: str = ""
    quality_warnings: list[str] = field(default_factory=list)

    def to_action_node(self, parent_stage_id: str = "", dataset: str = "", episode_id: str = "") -> ActionNode:
        action = ActionNode(
            action_id=stable_id("act", episode_id, parent_stage_id, self.source_step_index, self.operation, self.semantics.to_semantic_description().to_dict()),
            parent_stage_id=parent_stage_id,
            operation=self.operation,
            argument_template=self.argument_template,
            semantic_description=self.semantics.to_semantic_description(),
            source=SourceMetadata(dataset=dataset, episode_id=episode_id, step_index=self.source_step_index),
            source_debug={"raw_action_text": self.raw_action_text, "quality_warnings": self.quality_warnings},
        )
        return action


def infer_form_section(parent_context: str, nearby_text: str, label: str) -> str:
    text = normalize_key(" ".join([parent_context, nearby_text, label]))
    if any(x in text for x in ["flight", "depart", "return", "passenger"]):
        return "travel search form"
    if any(x in text for x in ["filter", "sort", "brand", "price", "rating"]):
        return "result filtering panel"
    if any(x in text for x in ["cart", "checkout", "shipping", "payment"]):
        return "checkout form"
    if any(x in text for x in ["comment", "post", "reply", "subreddit", "forum"]):
        return "forum content area"
    if any(x in text for x in ["repository", "issue", "merge", "gitlab"]):
        return "repository workflow area"
    if any(x in text for x in ["navigation", "menu", "section", "toc"]):
        return "navigation menu"
    if "search" in text:
        return "search form"
    return "task-relevant content area"


def infer_region(parent_context: str, nearby_text: str, section: str, label: str) -> str:
    text = normalize_key(" ".join([parent_context, nearby_text, section, label]))
    if any(x in text for x in ["header", "top nav", "global"]):
        return "global header"
    if any(x in text for x in ["sidebar", "filter", "refine"]):
        return "left/sidebar control area"
    if any(x in text for x in ["modal", "dialog", "popup"]):
        return "modal dialog"
    if any(x in text for x in ["footer"]):
        return "footer area"
    if any(x in text for x in ["form", "field", "search", "checkout"]):
        return "form area"
    return "main content"


def infer_field_slot(text: str, operation: str, argument: str | None = None) -> str:
    normalized = normalize_key(" ".join([text, argument or "", operation]))
    best_slot = ""
    best_hits = 0
    for slot, words in FIELD_SLOT_WORDS.items():
        hits = sum(1 for word in words if word in normalized)
        if hits > best_hits:
            best_slot = slot
            best_hits = hits
    if best_slot:
        return best_slot
    if canonical_operation(operation) == "type":
        return "input_value"
    if canonical_operation(operation) == "select":
        return "selected_option"
    return ""


def infer_value_source(argument: str | None, field_slot: str) -> str:
    if not argument:
        return "no argument"
    arg = normalize_space(argument)
    if field_slot:
        return f"copy task constraint for {field_slot}: {arg}"
    if re.search(r"\d", arg):
        return f"numeric or date value from task: {arg}"
    return f"task-provided value: {arg}"


def infer_element_purpose(role: str, label: str, operation: str, field_slot: str, section: str, region: str) -> str:
    op = canonical_operation(operation)
    label_text = label or field_slot or role
    if op == "type":
        if field_slot:
            return f"enter the task-provided {field_slot} into the {label_text} field"
        return f"enter task-provided text into the {label_text} field"
    if op == "select":
        if field_slot:
            return f"choose the task-specified {field_slot} option in {section or region}"
        return f"choose the intended option for {label_text}"
    if op == "click":
        if role in {"button", "link", "tab", "menuitem"}:
            return f"activate the {label_text} control in {section or region}"
        return f"activate the task-relevant element labeled {label_text}"
    if op == "scroll":
        return f"scroll within {section or region} to reveal the next relevant controls"
    if op == "stop":
        return "finish because the task outcome is already visible"
    return f"perform {op} on the task-relevant {label_text} element"


def infer_expected_state_change(operation: str, role: str, label: str, argument: str | None, field_slot: str, section: str, region: str) -> str:
    label = label or field_slot or role
    if operation == "type":
        return f"the {label} field contains the task-provided value"
    if operation == "select":
        return f"the chosen {label} option is selected and reflected in the form state"
    if operation == "click":
        text = normalize_key(" ".join([label, section, region]))
        if any(x in text for x in ["search", "submit", "apply", "go"]):
            return "the page updates to results, loading, or a submitted state"
        if any(x in text for x in ["filter", "sort"]):
            return "the result list reflects the selected filter or sort option"
        if any(x in text for x in ["next", "continue", "checkout", "cart"]):
            return "the workflow advances to the next step or confirmation state"
        if any(x in text for x in ["section", "classification", "anchor", "link"]):
            return "the requested section or linked page content becomes visible"
        return "the clicked control causes the expected task-relevant page update"
    if operation == "scroll":
        return "previously hidden task-relevant controls become visible"
    return "the observation changes consistently with the selected operation"


def derive_disambiguators(role: str, label: str, parent_context: str, sibling: str, nearby: str, section: str, region: str) -> list[str]:
    candidates = []
    if role:
        candidates.append(f"role is {role}")
    if label:
        candidates.append(f"primary text/name matches {label}")
    if section:
        candidates.append(f"belongs to {section}")
    if region:
        candidates.append(f"located in {region}")
    if parent_context:
        candidates.append(f"parent context mentions {parent_context[:80]}")
    if sibling:
        candidates.append(f"near sibling text {sibling[:80]}")
    if nearby:
        candidates.append(f"nearby text mentions {nearby[:80]}")
    seen = set()
    result = []
    for item in candidates:
        key = normalize_key(item)
        if key and key not in seen:
            result.append(item)
            seen.add(key)
    return result[:8]


def derive_negative_constraints(role: str, label: str, operation: str, section: str, region: str) -> list[str]:
    op = canonical_operation(operation)
    constraints: list[str] = []
    text = normalize_key(" ".join([role, label, section, region]))
    if op == "click" and "search" in text:
        constraints.append("not a site-wide search button unless it belongs to the task form")
    if op == "type":
        constraints.append("not a hidden field, disabled control, or unrelated newsletter/search box")
    if "filter" in text:
        constraints.append("not a navigation link with the same label outside the filter panel")
    if "more" in text or "next" in text:
        constraints.append("not an unrelated more/next control in another card or carousel")
    if "checkout" in text or "cart" in text:
        constraints.append("not a promotional link or recommendation card")
    return constraints


def derive_grounding_priority(role: str, operation: str, field_slot: str) -> list[str]:
    op = canonical_operation(operation)
    if op == "type":
        base = ["field_slot", "parent_context", "placeholder", "accessible_name", "label_or_text", "nearby_text", "role"]
    elif op == "select":
        base = ["field_slot", "label_or_text", "accessible_name", "parent_context", "option_text", "role"]
    elif op == "click":
        base = ["parent_context", "role", "label_or_text", "accessible_name", "nearby_text", "region"]
    else:
        base = ["role", "label_or_text", "parent_context", "nearby_text"]
    if not field_slot and "field_slot" in base:
        base.remove("field_slot")
    return base


def semantic_quality_warnings(semantics: CandidateSemantics, operation: str) -> list[str]:
    warnings: list[str] = []
    if not semantics.label_or_text and not semantics.accessible_name and not semantics.parent_context:
        warnings.append("descriptor has little lexical grounding evidence")
    if semantics.role in {"generic", "div", "span"} and canonical_operation(operation) in {"click", "type", "select"}:
        warnings.append("target role is generic; grounding may require context")
    if not semantics.expected_state_change:
        warnings.append("expected state change is missing")
    if not semantics.grounding_priority:
        warnings.append("grounding priority is missing")
    return warnings


def abstract_action_from_step(step: TrajectoryStep, dataset: str = "", episode_id: str = "", parent_stage_id: str = "") -> ActionNode:
    operation = canonical_operation(step.operation, step.target)
    semantics = CandidateSemantics.from_target(step.target, operation, step.argument, step.observation)
    draft = SemanticActionDraft(
        operation=operation,
        argument_template=step.argument,
        semantics=semantics,
        source_step_index=step.step_index,
        raw_action_text=step.action_text or step.summary(),
        quality_warnings=semantic_quality_warnings(semantics, operation),
    )
    return draft.to_action_node(parent_stage_id=parent_stage_id, dataset=dataset, episode_id=episode_id)


def enrich_semantic_description(description: SemanticDescription, operation: str, argument: str | None = None) -> SemanticDescription:
    data = description.to_dict()
    semantics = CandidateSemantics.from_target(data, operation=operation, argument=argument)
    merged = description.to_dict()
    for key, value in semantics.to_semantic_description().to_dict().items():
        if value not in ("", None, [], {}) and key not in merged:
            merged[key] = value
    return SemanticDescription.from_dict(merged)


def descriptor_similarity(left: SemanticDescription | dict[str, Any], right: SemanticDescription | dict[str, Any]) -> float:
    l = left.to_dict() if isinstance(left, SemanticDescription) else left
    r = right.to_dict() if isinstance(right, SemanticDescription) else right
    field_weights = {
        "role": 0.12,
        "label_or_text": 0.20,
        "accessible_name": 0.18,
        "element_purpose": 0.12,
        "parent_context": 0.12,
        "form_section": 0.10,
        "nearby_text": 0.08,
        "field_slot": 0.08,
    }
    total = 0.0
    weight_sum = 0.0
    for field, weight in field_weights.items():
        lv = l.get(field)
        rv = r.get(field)
        if not lv or not rv:
            continue
        if field == "role":
            score = 1.0 if normalize_key(lv) == normalize_key(rv) else 0.0
        else:
            score = _overlap(lv, rv)
        total += weight * score
        weight_sum += weight
    if weight_sum == 0:
        return _overlap(l, r)
    return total / weight_sum


def verify_transferable_action(action: ActionNode) -> list[str]:
    problems: list[str] = []
    semantic = action.semantic_description.to_dict()
    for key in semantic:
        if key in SEMANTIC_FORBIDDEN:
            problems.append(f"forbidden field retained in semantic description: {key}")
    if canonical_operation(action.operation) in {"click", "type", "select"}:
        if not semantic.get("role"):
            problems.append("role is missing")
        if not any(semantic.get(k) for k in ["label_or_text", "accessible_name", "element_purpose", "parent_context", "field_slot"]):
            problems.append("no transferable grounding evidence")
    if action.operation == "type" and not action.argument_template:
        problems.append("type action has no argument template")
    return problems


def make_step_context_payload(step: TrajectoryStep, max_candidates: int = 30) -> dict[str, Any]:
    snapshot = PageSnapshot.from_observation(step.observation)
    candidates = snapshot.candidate_dicts(max_elements=max_candidates)
    return {
        "step_index": step.step_index,
        "operation": canonical_operation(step.operation, step.target),
        "argument": step.argument,
        "target_hint": {k: v for k, v in step.target.items() if k not in SEMANTIC_FORBIDDEN},
        "observation_summary": snapshot.summary(max_elements=max_candidates),
        "candidate_elements": candidates,
    }
