from __future__ import annotations

"""Convert Mind2Web-style JSON files into JSONL split files."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmt.utils.io import read_json, read_jsonl, write_json, write_jsonl


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_path(path: Path) -> dict[str, str | int]:
    if not path.exists():
        return {}
    if path.is_file():
        return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    h = hashlib.sha256()
    for file in files[:10000]:
        rel = file.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(file).encode("ascii"))
    return {"path": str(path), "num_files": len(files), "sha256": h.hexdigest()}


_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "cross_task": "cross_task",
    "cross-task": "cross_task",
    "cross task": "cross_task",
    "test_task": "cross_task",
    "cross_website": "cross_website",
    "cross-website": "cross_website",
    "cross website": "cross_website",
    "test_website": "cross_website",
    "cross_domain": "cross_domain",
    "cross-domain": "cross_domain",
    "cross domain": "cross_domain",
    "test_domain": "cross_domain",
}


def _normalize_split(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _SPLIT_ALIASES.get(text, text or "unknown")


def _read_any(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = read_json(path)
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ["data", "examples", "tasks", "annotations", "records"]:
            if isinstance(data.get(key), list):
                return [dict(item) for item in data[key] if isinstance(item, dict)]
        return [data]
    return []


def _iter_local_records(source: Path) -> Iterable[dict[str, Any]]:
    files = [source] if source.is_file() else sorted([*source.rglob("*.json"), *source.rglob("*.jsonl")])
    for file in files:
        for record in _read_any(file):
            row = dict(record)
            row.setdefault("_source_file", str(file.relative_to(source) if source.is_dir() else file.name))
            yield row


def _iter_hf_records(dataset_name: str, hf_config: str | None = None) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install `datasets` to load Mind2Web from HuggingFace.") from exc
    dataset = load_dataset(dataset_name, hf_config) if hf_config else load_dataset(dataset_name)
    for split_name, split in dataset.items():
        for record in split:
            row = dict(record)
            row.setdefault("split", split_name)
            yield row


def _split_for_record(record: dict[str, Any], source_hint: str = "") -> str:
    for key in ["split", "data_split", "benchmark_split", "annotation_split"]:
        if record.get(key):
            return _normalize_split(record[key])
    hint = source_hint.lower().replace("-", "_").replace(" ", "_")
    for alias, canonical in _SPLIT_ALIASES.items():
        if alias.replace("-", "_").replace(" ", "_") in hint:
            return canonical
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mind2Web JSON/HF data for HMT.")
    parser.add_argument("--source", required=True, help="Local file/dir or HuggingFace dataset name.")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-raw", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)
    if source_path.exists():
        records = list(_iter_local_records(source_path))
        source_kind = "local"
        if args.copy_raw:
            raw_target = output_dir / "raw"
            if source_path.is_dir():
                if raw_target.exists():
                    shutil.rmtree(raw_target)
                shutil.copytree(source_path, raw_target)
            else:
                raw_target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, raw_target / source_path.name)
    else:
        records = list(_iter_hf_records(args.source, args.hf_config))
        source_kind = "huggingface"

    by_split: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        split = _split_for_record(record, str(record.get("_source_file", "")))
        row = dict(record)
        row.setdefault("_prepare_index", index)
        row.setdefault("_prepare_unix", time.time())
        by_split.setdefault(split, []).append(row)

    outputs: dict[str, Any] = {}
    for split, rows in sorted(by_split.items()):
        target = output_dir / f"{split}.jsonl"
        write_jsonl(target, rows)
        outputs[split] = {"path": str(target), "num_records": len(rows), "sha256": sha256_file(target)}
    summary = {
        "source": args.source,
        "source_kind": source_kind,
        "source_fingerprint": fingerprint_path(source_path) if source_path.exists() else None,
        "outputs": outputs,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
