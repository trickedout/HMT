from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, sort_keys=True)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        expanded = _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)
        return os.path.expandvars(expanded)
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and data.get("extends"):
        parent_path = target.parent / str(data["extends"])
        parent = read_yaml(parent_path)
        data = _deep_merge(parent, data)
    return _expand_env_placeholders(data or {})


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]
