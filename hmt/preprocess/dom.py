from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup, Tag


INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "option", "summary"}
INTERACTIVE_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "link",
    "menuitem",
    "option",
    "radio",
    "searchbox",
    "switch",
    "tab",
    "textbox",
}


@dataclass
class DOMElement:
    element_id: str
    role: str
    visible_text: str
    accessible_name: str
    value: str
    bbox: dict[str, float] | None
    dom_path: str
    parent_context: str
    sibling_text: str
    clickable: bool
    editable: bool
    visible: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "role": self.role,
            "visible_text": self.visible_text,
            "accessible_name": self.accessible_name,
            "value": self.value,
            "bbox": self.bbox,
            "dom_path": self.dom_path,
            "parent_context": self.parent_context,
            "sibling_text": self.sibling_text,
            "clickable": self.clickable,
            "editable": self.editable,
            "visible": self.visible,
        }


def _is_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True
    aria_hidden = str(tag.get("aria-hidden", "")).lower()
    if aria_hidden == "true":
        return True
    style = str(tag.get("style", "")).lower().replace(" ", "")
    return "display:none" in style or "visibility:hidden" in style


def _is_disabled_noninteractive(tag: Tag) -> bool:
    if not tag.has_attr("disabled"):
        return False
    return not _is_interactive(tag)


def _is_interactive(tag: Tag) -> bool:
    role = str(tag.get("role", "")).lower()
    return tag.name in INTERACTIVE_TAGS or role in INTERACTIVE_ROLES or tag.has_attr("onclick")


def _role(tag: Tag) -> str:
    if tag.get("role"):
        return str(tag.get("role")).lower()
    if tag.name == "a":
        return "link"
    if tag.name == "input":
        input_type = str(tag.get("type", "text")).lower()
        if input_type in {"submit", "button", "reset"}:
            return "button"
        if input_type == "checkbox":
            return "checkbox"
        if input_type == "radio":
            return "radio"
        return "textbox"
    return tag.name or "element"


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


def _name(tag: Tag) -> str:
    for attr in ("aria-label", "title", "alt", "placeholder", "name"):
        if tag.get(attr):
            return str(tag.get(attr)).strip()
    return _text(tag)


def _dom_path(tag: Tag) -> str:
    parts = []
    current: Tag | None = tag
    while current and isinstance(current, Tag) and current.name != "[document]":
        if current.parent:
            siblings = [s for s in current.parent.find_all(current.name, recursive=False)]
            index = siblings.index(current) + 1 if current in siblings else 1
        else:
            index = 1
        parts.append(f"{current.name}:nth-of-type({index})")
        current = current.parent if isinstance(current.parent, Tag) else None
    return " > ".join(reversed(parts))


def _parent_context(tag: Tag) -> str:
    parent = tag.parent if isinstance(tag.parent, Tag) else None
    if not parent:
        return ""
    return _text(parent)[:160]


def _sibling_text(tag: Tag) -> str:
    parent = tag.parent if isinstance(tag.parent, Tag) else None
    if not parent:
        return ""
    texts = []
    for sibling in parent.find_all(recursive=False):
        if sibling is not tag and isinstance(sibling, Tag):
            value = _text(sibling)
            if value:
                texts.append(value)
    return " ".join(texts)[:160]


def clean_dom(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if _is_hidden(tag) or _is_disabled_noninteractive(tag):
            tag.decompose()
    for tag in list(soup.find_all(True)):
        if not _text(tag) and not _is_interactive(tag) and not tag.find(True):
            tag.decompose()
    return soup


def extract_dom_candidates(html: str) -> list[dict[str, Any]]:
    soup = clean_dom(html)
    candidates: list[DOMElement] = []
    seen_layout_texts: set[tuple[str, str]] = set()
    for index, tag in enumerate(soup.find_all(True), start=1):
        text = _text(tag)
        interactive = _is_interactive(tag)
        if not interactive and not text:
            continue
        role = _role(tag)
        key = (role, text)
        if not interactive and key in seen_layout_texts:
            continue
        seen_layout_texts.add(key)
        editable = role in {"textbox", "searchbox", "combobox"} or tag.name in {"input", "textarea", "select"}
        element = DOMElement(
            element_id=str(tag.get("data-hmt-id") or tag.get("id") or f"dom_{index:04d}"),
            role=role,
            visible_text=text,
            accessible_name=_name(tag),
            value=str(tag.get("value", "")),
            bbox=None,
            dom_path=_dom_path(tag),
            parent_context=_parent_context(tag),
            sibling_text=_sibling_text(tag),
            clickable=interactive and not editable,
            editable=editable,
        )
        candidates.append(element)
    candidates.sort(key=lambda e: (not e.clickable and not e.editable, e.dom_path, e.element_id))
    return [candidate.to_dict() for candidate in candidates]
