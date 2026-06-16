from __future__ import annotations

from typing import Any


def flatten_accessibility_tree(node: dict[str, Any], prefix: str = "ax") -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []

    def walk(current: dict[str, Any], path: list[int]) -> None:
        role = str(current.get("role", ""))
        name = str(current.get("name", current.get("accessible_name", "")))
        value = str(current.get("value", ""))
        visible = bool(current.get("visible", True))
        if visible and (role or name or value):
            element_id = str(current.get("element_id") or f"{prefix}_{'_'.join(map(str, path)) or 'root'}")
            elements.append(
                {
                    "element_id": element_id,
                    "role": role,
                    "visible_text": str(current.get("text", name)),
                    "accessible_name": name,
                    "value": value,
                    "bbox": current.get("bbox"),
                    "dom_path": str(current.get("dom_path", "")),
                    "parent_context": str(current.get("parent_context", "")),
                    "sibling_text": str(current.get("sibling_text", "")),
                    "clickable": bool(current.get("clickable", role in {"button", "link", "menuitem"})),
                    "editable": bool(current.get("editable", role in {"textbox", "combobox", "searchbox"})),
                    "visible": visible,
                }
            )
        for index, child in enumerate(current.get("children", []) or []):
            if isinstance(child, dict):
                walk(child, path + [index])

    walk(node, [])
    elements.sort(key=lambda e: (not e["clickable"] and not e["editable"], e["element_id"]))
    return elements
