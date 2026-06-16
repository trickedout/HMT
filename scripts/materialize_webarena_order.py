from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmt.utils.io import read_json, write_json, write_jsonl


def canonical_domain(value: str) -> str:
    text = value.lower().replace("-", "_").strip()
    if text in {"shopping_admin", "cms", "admin"}:
        return "shopping_admin"
    if text in {"map", "maps"}:
        return "map"
    if "gitlab" in text:
        return "gitlab"
    if "reddit" in text:
        return "reddit"
    if "admin" in text or "cms" in text:
        return "shopping_admin"
    if "shop" in text:
        return "shopping"
    if "map" in text:
        return "map"
    return text or "unknown"


def _first_present(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _infer_domain(record: dict[str, Any], path: Path) -> str:
    raw = _first_present(record, "domain", "website", "site", default="")
    if not raw and record.get("sites"):
        sites = record.get("sites")
        raw = sites[0] if isinstance(sites, list) and sites else sites
    if raw:
        return canonical_domain(str(raw))
    text = " ".join([path.as_posix(), json.dumps(record, ensure_ascii=False)[:2000]])
    return canonical_domain(text)


def _task_id(record: dict[str, Any], path: Path) -> str:
    raw = _first_present(record, "task_id", "id", "task_index", default=None)
    if raw is not None:
        return str(raw)
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else path.stem


def _load_records(task_config_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file in sorted(task_config_dir.rglob("*.json")):
        payload = read_json(file)
        items = payload if isinstance(payload, list) else [payload]
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            task_id = _task_id(item, file) if len(items) == 1 else f"{_task_id(item, file)}_{index}"
            records.append(
                {
                    "task_id": task_id,
                    "domain": _infer_domain(item, file),
                    "config_file": str(file.relative_to(task_config_dir)),
                    "intent": str(_first_present(item, "intent", "instruction", "confirmed_task", default="")),
                    "sites": item.get("sites", item.get("site", item.get("domain", []))),
                }
            )
    return records


def _sort_key(record: dict[str, Any], sort_key: str) -> tuple[Any, ...]:
    if sort_key == "filename":
        return (str(record.get("config_file", "")), str(record.get("task_id", "")))
    if sort_key == "original":
        return (0,)
    task_id = str(record.get("task_id", ""))
    numeric = int(task_id) if task_id.isdigit() else 10**12
    return (numeric, task_id, str(record.get("config_file", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a WebArena task-order JSONL file from task configs.")
    parser.add_argument("--task-config-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sort-key", choices=["task_id", "filename", "original"], default="task_id")
    parser.add_argument("--domains", default="shopping,shopping_admin,reddit,gitlab,map")
    args = parser.parse_args()

    task_config_dir = Path(args.task_config_dir)
    records = _load_records(task_config_dir)
    domains = [canonical_domain(part) for part in args.domains.split(",") if part.strip()]
    ordered: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for domain in domains:
        selected = [r for r in records if canonical_domain(str(r.get("domain"))) == domain]
        if args.sort_key != "original":
            selected.sort(key=lambda row: _sort_key(row, args.sort_key))
        counts[domain] = len(selected)
        ordered.extend(selected)
    write_jsonl(args.output, ordered)
    write_json(Path(args.output).with_suffix(".counts.json"), {**counts, "total": sum(counts.values())})
    print(json.dumps({"output": args.output, "counts": counts, "num_tasks": len(ordered)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
