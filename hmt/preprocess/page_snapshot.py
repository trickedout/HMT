from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Iterable
import hashlib
import json
import re
import unicodedata

INTERACTIVE_TAGS = {
    "a", "button", "input", "select", "textarea", "option", "summary", "details",
}
INTERACTIVE_ROLES = {
    "button", "link", "textbox", "combobox", "checkbox", "radio", "menuitem", "option",
    "tab", "switch", "slider", "searchbox", "spinbutton", "listbox", "treeitem",
}
TEXT_CONTAINER_TAGS = {
    "main", "section", "article", "aside", "nav", "header", "footer", "form", "fieldset",
    "div", "ul", "ol", "li", "table", "tr", "td", "th", "label", "p", "span", "h1", "h2", "h3", "h4",
}
SOURCE_ID_FIELDS = {
    "backend_node_id", "node_id", "raw_node_id", "css_selector", "selector", "xpath", "bbox",
    "bounding_box", "coordinates", "absolute_coordinates", "data_reactid", "data_vueid", "dom_path",
}
VISIBLE_TEXT_FIELDS = (
    "text", "visible_text", "inner_text", "label", "label_or_text", "accessible_name", "name",
    "aria_label", "placeholder", "value", "title", "alt",
)


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def normalize_space(text: Any) -> str:
    raw = _flatten(text)
    raw = unicodedata.normalize("NFKC", raw)
    raw = raw.replace("\u00a0", " ")
    raw = re.sub(r"[\t\r\n]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def normalize_key(text: Any) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^\w\s:/.-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def stable_element_id(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return "el_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _role_from_tag(tag: str, attrs: dict[str, Any]) -> str:
    explicit = normalize_key(attrs.get("role", ""))
    if explicit:
        return explicit
    tag = tag.lower()
    if tag == "a":
        return "link"
    if tag == "button":
        return "button"
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textbox"
    if tag == "option":
        return "option"
    if tag == "input":
        typ = normalize_key(attrs.get("type", "text"))
        if typ in {"submit", "button", "reset", "image"}:
            return "button"
        if typ in {"checkbox"}:
            return "checkbox"
        if typ in {"radio"}:
            return "radio"
        if typ in {"range"}:
            return "slider"
        if typ in {"search"}:
            return "searchbox"
        return "textbox"
    if tag == "form":
        return "form"
    if tag == "nav":
        return "navigation"
    if tag == "main":
        return "main"
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return "heading"
    if tag == "img":
        return "img"
    return tag


def _is_probably_visible(attrs: dict[str, Any]) -> bool:
    if "hidden" in attrs:
        return False
    aria_hidden = normalize_key(attrs.get("aria-hidden", ""))
    if aria_hidden == "true":
        return False
    style = normalize_key(attrs.get("style", ""))
    if "display none" in style or "visibility hidden" in style:
        return False
    return True


def _is_disabled(attrs: dict[str, Any]) -> bool:
    if "disabled" in attrs:
        return True
    aria_disabled = normalize_key(attrs.get("aria-disabled", ""))
    return aria_disabled == "true"


@dataclass
class ElementSnapshot:
    element_id: str
    role: str
    tag: str = ""
    text: str = ""
    accessible_name: str = ""
    label: str = ""
    placeholder: str = ""
    value: str = ""
    title: str = ""
    alt: str = ""
    url: str = ""
    parent_context: str = ""
    ancestor_text: str = ""
    sibling_text: str = ""
    nearby_text: str = ""
    form_section: str = ""
    region: str = ""
    relative_position: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: dict[str, Any], index: int = 0, parent_context: str = "", sibling_text: str = "") -> "ElementSnapshot":
        attrs = dict(item.get("attributes") or {})
        merged: dict[str, Any] = {**attrs, **item}
        role = normalize_key(merged.get("role") or _role_from_tag(str(merged.get("tag", "")), attrs))
        text = normalize_space(merged.get("text") or merged.get("visible_text") or merged.get("inner_text") or merged.get("label_or_text"))
        accessible = normalize_space(merged.get("accessible_name") or merged.get("name") or merged.get("aria_label") or attrs.get("aria-label"))
        label = normalize_space(merged.get("label") or merged.get("label_or_text"))
        placeholder = normalize_space(merged.get("placeholder") or attrs.get("placeholder"))
        value = normalize_space(merged.get("value") or attrs.get("value"))
        title = normalize_space(merged.get("title") or attrs.get("title"))
        alt = normalize_space(merged.get("alt") or attrs.get("alt"))
        element_id = str(merged.get("element_id") or merged.get("id") or stable_element_id(index, role, text, accessible, placeholder, value))
        state = dict(merged.get("state") or {})
        if _is_disabled(attrs) or bool(merged.get("disabled", False)):
            state["disabled"] = True
        if str(merged.get("checked", attrs.get("checked", ""))).lower() in {"true", "checked", "1"}:
            state["checked"] = True
        if str(merged.get("selected", attrs.get("selected", ""))).lower() in {"true", "selected", "1"}:
            state["selected"] = True
        if not _is_probably_visible(attrs):
            state["visible"] = False
        return cls(
            element_id=element_id,
            role=role or "generic",
            tag=str(merged.get("tag", "")),
            text=text,
            accessible_name=accessible,
            label=label,
            placeholder=placeholder,
            value=value,
            title=title,
            alt=alt,
            url=normalize_space(merged.get("url") or merged.get("href") or attrs.get("href")),
            parent_context=normalize_space(merged.get("parent_context") or parent_context),
            ancestor_text=normalize_space(merged.get("ancestor_text")),
            sibling_text=normalize_space(merged.get("sibling_text") or sibling_text),
            nearby_text=normalize_space(merged.get("nearby_text")),
            form_section=normalize_space(merged.get("form_section") or merged.get("section")),
            region=normalize_space(merged.get("region")),
            relative_position=normalize_space(merged.get("relative_position")),
            state=state,
            attributes={k: v for k, v in attrs.items() if k not in SOURCE_ID_FIELDS},
            raw={k: v for k, v in item.items() if k not in SOURCE_ID_FIELDS},
        )

    def primary_text(self) -> str:
        for value in [self.accessible_name, self.label, self.text, self.placeholder, self.value, self.title, self.alt]:
            if normalize_space(value):
                return normalize_space(value)
        return ""

    def semantic_text(self, include_state: bool = True) -> str:
        parts: list[str] = [
            self.role,
            self.primary_text(),
            self.parent_context,
            self.form_section,
            self.region,
            self.ancestor_text,
            self.sibling_text,
            self.nearby_text,
            self.relative_position,
        ]
        if include_state and self.state:
            state_text = " ".join(f"{k}:{v}" for k, v in sorted(self.state.items()))
            parts.append(state_text)
        return normalize_space(" ".join(p for p in parts if p))

    def is_interactive(self) -> bool:
        if self.role in INTERACTIVE_ROLES:
            return True
        if self.tag.lower() in INTERACTIVE_TAGS:
            return True
        if self.attributes.get("onclick") or self.attributes.get("tabindex"):
            return True
        return False

    def salience_score(self) -> float:
        score = 0.0
        if self.is_interactive():
            score += 2.0
        if self.primary_text():
            score += 1.0
        if self.placeholder:
            score += 0.8
        if self.form_section or self.parent_context:
            score += 0.5
        if self.state.get("disabled"):
            score -= 1.0
        text_len = len(self.semantic_text())
        score += min(1.0, text_len / 120.0)
        return score

    def to_candidate(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "element_id": self.element_id,
            "role": self.role,
            "tag": self.tag,
            "visible_text": self.text,
            "label_or_text": self.label or self.text,
            "accessible_name": self.accessible_name,
            "placeholder": self.placeholder,
            "value": self.value,
            "title": self.title,
            "alt": self.alt,
            "parent_context": self.parent_context,
            "ancestor_text": self.ancestor_text,
            "sibling_text": self.sibling_text,
            "nearby_text": self.nearby_text,
            "form_section": self.form_section,
            "region": self.region,
            "relative_position": self.relative_position,
            "state": self.state,
            "salience": round(self.salience_score(), 4),
        }
        return {k: v for k, v in item.items() if v not in ("", None, {}, [])}


@dataclass
class PageSnapshot:
    url: str = ""
    title: str = ""
    domain: str = ""
    elements: list[ElementSnapshot] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_observation(cls, observation: dict[str, Any] | str, candidate_elements: list[dict[str, Any]] | None = None) -> "PageSnapshot":
        if isinstance(observation, str):
            text = normalize_space(observation)
            if "<" in observation and ">" in observation:
                snapshot = PageSnapshotHTMLParser.parse(observation)
            else:
                snapshot = cls(text_blocks=[text], source="text")
        else:
            snapshot = cls(
                url=normalize_space(observation.get("url")),
                title=normalize_space(observation.get("title")),
                domain=normalize_space(observation.get("domain") or observation.get("site")),
                text_blocks=[normalize_space(x) for x in observation.get("text_blocks", []) if normalize_space(x)],
                source=str(observation.get("source", "mapping")),
                metadata=dict(observation.get("metadata", {})),
            )
            for key in ["html", "dom", "cleaned_html", "raw_html"]:
                if isinstance(observation.get(key), str) and observation.get(key):
                    parsed = PageSnapshotHTMLParser.parse(str(observation[key]))
                    snapshot.elements.extend(parsed.elements)
                    snapshot.text_blocks.extend(parsed.text_blocks)
                    break
            for key in ["accessibility_tree", "axtree", "tree"]:
                if observation.get(key):
                    snapshot.elements.extend(elements_from_accessibility_tree(observation[key]))
                    break
            for key in ["candidate_elements", "candidates", "elements"]:
                if isinstance(observation.get(key), list):
                    snapshot.elements.extend(elements_from_candidates(observation[key]))
                    break
        if candidate_elements:
            snapshot.elements.extend(elements_from_candidates(candidate_elements))
        snapshot.deduplicate()
        snapshot.infer_contexts()
        return snapshot

    def deduplicate(self) -> None:
        seen: set[str] = set()
        unique: list[ElementSnapshot] = []
        for idx, el in enumerate(self.elements):
            key = normalize_key("|".join([el.element_id, el.role, el.primary_text(), el.parent_context, str(idx)]))
            stable = el.element_id or stable_element_id(idx, key)
            if stable in seen:
                alt_key = stable_element_id(idx, key)
                if alt_key in seen:
                    continue
                el.element_id = alt_key
                seen.add(alt_key)
            else:
                el.element_id = stable
                seen.add(stable)
            unique.append(el)
        self.elements = unique

    def infer_contexts(self) -> None:
        if not self.elements:
            return
        for i, el in enumerate(self.elements):
            if not el.relative_position:
                el.relative_position = f"element {i + 1} of {len(self.elements)}"
            if not el.sibling_text:
                left = self.elements[i - 1].primary_text() if i > 0 else ""
                right = self.elements[i + 1].primary_text() if i + 1 < len(self.elements) else ""
                el.sibling_text = normalize_space(" ".join(x for x in [left, right] if x))
            if not el.nearby_text:
                nearby = []
                for j in range(max(0, i - 2), min(len(self.elements), i + 3)):
                    if j != i:
                        nearby.append(self.elements[j].primary_text())
                el.nearby_text = normalize_space(" ".join(x for x in nearby if x))
            if not el.region:
                el.region = self._infer_region_for(el)

    def _infer_region_for(self, el: ElementSnapshot) -> str:
        text = normalize_key(" ".join([el.parent_context, el.ancestor_text, el.form_section, el.nearby_text]))
        if any(w in text for w in ["filter", "sort", "refine"]):
            return "filter panel"
        if any(w in text for w in ["search", "query"]):
            return "search form"
        if any(w in text for w in ["cart", "checkout", "payment"]):
            return "checkout area"
        if any(w in text for w in ["navigation", "menu", "section"]):
            return "navigation area"
        if any(w in text for w in ["comment", "reply", "post"]):
            return "content discussion area"
        return "main content"

    def salient_elements(self, max_elements: int = 30, interactive_only: bool = False) -> list[ElementSnapshot]:
        candidates = [e for e in self.elements if (e.is_interactive() or not interactive_only)]
        candidates.sort(key=lambda e: (-e.salience_score(), e.element_id))
        return candidates[:max_elements]

    def candidate_dicts(self, max_elements: int = 30, interactive_only: bool = False) -> list[dict[str, Any]]:
        return [el.to_candidate() for el in self.salient_elements(max_elements=max_elements, interactive_only=interactive_only)]

    def summary(self, max_chars: int = 2400, max_elements: int = 30) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.url:
            parts.append(f"URL: {self.url}")
        text = normalize_space(" ".join(self.text_blocks[:20]))
        if text:
            parts.append(f"Visible text: {text[:max_chars // 2]}")
        element_lines = []
        for el in self.salient_elements(max_elements=max_elements):
            label = el.primary_text()
            context = normalize_space(" | ".join(x for x in [el.parent_context, el.form_section, el.region] if x))
            element_lines.append(f"- {el.element_id}: role={el.role}; text={label}; context={context}")
        if element_lines:
            parts.append("Salient elements:\n" + "\n".join(element_lines))
        summary = normalize_space("\n".join(parts))
        return summary[:max_chars]

    def to_llm_payload(self, max_elements: int = 30) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "summary": self.summary(max_elements=max_elements),
            "candidate_elements": self.candidate_dicts(max_elements=max_elements),
        }


class PageSnapshotHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[dict[str, Any]] = []
        self.elements: list[ElementSnapshot] = []
        self.text_blocks: list[str] = []
        self._counter = 0
        self.title = ""
        self._in_title = False

    @classmethod
    def parse(cls, html: str) -> PageSnapshot:
        parser = cls()
        parser.feed(html)
        parser.close()
        return PageSnapshot(title=parser.title, elements=parser.elements, text_blocks=parser.text_blocks, source="html")

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: (v if v is not None else "") for k, v in attrs_list}
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        parent_context = self._current_context()
        item = {
            "tag": tag,
            "attributes": attrs,
            "role": _role_from_tag(tag, attrs),
            "label_or_text": attrs.get("aria-label") or attrs.get("title") or attrs.get("placeholder") or attrs.get("value") or "",
            "accessible_name": attrs.get("aria-label") or attrs.get("title") or "",
            "placeholder": attrs.get("placeholder") or "",
            "value": attrs.get("value") or "",
            "title": attrs.get("title") or "",
            "alt": attrs.get("alt") or "",
            "href": attrs.get("href") or "",
        }
        frame = {"tag": tag, "attrs": attrs, "texts": [], "parent_context": parent_context, "item": item, "start_index": self._counter}
        self.stack.append(frame)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if not self.stack:
            return
        frame = self.stack.pop()
        text = normalize_space(" ".join(frame.get("texts", [])))
        if text:
            self.text_blocks.append(text)
            if self.stack:
                self.stack[-1].setdefault("texts", []).append(text)
        item = dict(frame.get("item", {}))
        item["text"] = text or item.get("label_or_text", "")
        item["parent_context"] = frame.get("parent_context", "")
        item["element_id"] = stable_element_id(frame.get("start_index"), item.get("tag"), item.get("role"), item.get("text"), item.get("accessible_name"))
        role = str(item.get("role", ""))
        tag_name = str(item.get("tag", ""))
        if tag_name in INTERACTIVE_TAGS or role in INTERACTIVE_ROLES or normalize_space(item.get("text")):
            if tag_name in INTERACTIVE_TAGS or role in INTERACTIVE_ROLES or tag_name in TEXT_CONTAINER_TAGS:
                self.elements.append(ElementSnapshot.from_mapping(item, index=len(self.elements), parent_context=frame.get("parent_context", "")))

    def handle_data(self, data: str) -> None:
        text = normalize_space(data)
        if not text:
            return
        if self._in_title:
            self.title = normalize_space(" ".join([self.title, text]))
        if self.stack:
            self.stack[-1].setdefault("texts", []).append(text)
        else:
            self.text_blocks.append(text)

    def _current_context(self) -> str:
        contexts: list[str] = []
        for frame in reversed(self.stack[-4:]):
            tag = frame.get("tag", "")
            attrs = frame.get("attrs", {})
            label = attrs.get("aria-label") or attrs.get("title") or ""
            texts = normalize_space(" ".join(frame.get("texts", [])[:3]))
            piece = normalize_space(" ".join(x for x in [tag, label, texts] if x))
            if piece:
                contexts.append(piece)
        return normalize_space(" / ".join(reversed(contexts)))


def elements_from_accessibility_tree(tree: Any) -> list[ElementSnapshot]:
    result: list[ElementSnapshot] = []

    def visit(node: Any, depth: int = 0, ancestors: list[str] | None = None) -> None:
        if ancestors is None:
            ancestors = []
        if isinstance(node, dict):
            role = normalize_key(node.get("role") or node.get("nodeRole") or node.get("type") or "")
            name = normalize_space(node.get("name") or node.get("accessible_name") or node.get("text") or "")
            parent_context = normalize_space(" / ".join(ancestors[-4:]))
            item = {
                "element_id": node.get("element_id") or node.get("backend_node_id") or stable_element_id(depth, role, name, parent_context, len(result)),
                "role": role,
                "accessible_name": name,
                "text": node.get("text") or name,
                "parent_context": parent_context,
                "state": node.get("state") or {},
                "attributes": node.get("attributes") or {},
            }
            if role or name:
                result.append(ElementSnapshot.from_mapping(item, index=len(result), parent_context=parent_context))
            next_ancestors = ancestors + ([f"{role}:{name}".strip(":")] if role or name else [])
            children = node.get("children") or node.get("childNodes") or []
            for child in children:
                visit(child, depth + 1, next_ancestors)
        elif isinstance(node, list):
            for child in node:
                visit(child, depth, ancestors)

    visit(tree)
    return result


def elements_from_candidates(candidates: Iterable[dict[str, Any]]) -> list[ElementSnapshot]:
    elements: list[ElementSnapshot] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        elements.append(ElementSnapshot.from_mapping(candidate, index=index))
    return elements


def normalize_candidate_elements(candidates: Iterable[dict[str, Any]], max_elements: int | None = None) -> list[dict[str, Any]]:
    snapshot = PageSnapshot(elements=elements_from_candidates(candidates))
    snapshot.deduplicate()
    snapshot.infer_contexts()
    items = snapshot.candidate_dicts(max_elements=max_elements or len(snapshot.elements), interactive_only=False)
    return items


def observation_to_summary(observation: dict[str, Any] | str, candidate_elements: list[dict[str, Any]] | None = None, max_chars: int = 2400) -> str:
    return PageSnapshot.from_observation(observation, candidate_elements).summary(max_chars=max_chars)


def observation_to_candidates(observation: dict[str, Any] | str, candidate_elements: list[dict[str, Any]] | None = None, max_elements: int = 30) -> list[dict[str, Any]]:
    return PageSnapshot.from_observation(observation, candidate_elements).candidate_dicts(max_elements=max_elements)
