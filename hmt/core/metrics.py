from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hmt.utils.io import read_json
from hmt.utils.io import read_jsonl


def weighted_task_sr(domain_scores: dict[str, float], domain_counts: dict[str, int]) -> float:
    numerator = 0.0
    denominator = 0
    for domain, score in domain_scores.items():
        if domain not in domain_counts:
            raise KeyError(f"Missing count for domain: {domain}")
        count = int(domain_counts[domain])
        numerator += count * float(score)
        denominator += count
    if denominator == 0:
        raise ValueError("No tasks in denominator")
    return numerator / denominator


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    delta = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return center - delta, center + delta


def _read_webarena_csv(input_csv: str | Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with Path(input_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"method", "domain", "task_sr"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")
        for row in reader:
            method = row["method"].strip()
            domain = row["domain"].strip()
            results.setdefault(method, {})[domain] = float(row["task_sr"])
    return results


def _read_per_task_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    required = {"task_id", "domain", "method", "success"}
    for index, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Missing fields in per-task log {path}:{index}: {sorted(missing)}")
    return records


def _counts_without_metadata(counts: dict[str, Any]) -> dict[str, int]:
    return {
        domain: int(count)
        for domain, count in counts.items()
        if isinstance(count, int) and domain not in {"five_domain_total", "official_multisite_not_shown_in_domain_table"}
    }


def recompute_webarena_from_task_logs(input_jsonl: str, counts_json: str | None = None) -> dict[str, Any]:
    records = _read_per_task_jsonl(input_jsonl)
    counts_raw = read_json(counts_json) if counts_json else {}
    output: dict[str, Any] = {"counts": counts_raw, "methods": {}, "interval_source": "raw_per_task_logs"}
    by_method_domain: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        by_method_domain.setdefault(str(record["method"]), {}).setdefault(str(record["domain"]), []).append(record)
    for method, domains in by_method_domain.items():
        domain_scores: dict[str, float] = {}
        domain_counts: dict[str, int] = {}
        successes_by_domain: dict[str, int] = {}
        for domain, items in domains.items():
            successes = sum(1 for item in items if bool(item["success"]))
            n = len(items)
            successes_by_domain[domain] = successes
            domain_counts[domain] = n
            domain_scores[domain] = 100.0 * successes / n if n else 0.0
        total_successes = sum(successes_by_domain.values())
        total_n = sum(domain_counts.values())
        low, high = wilson_interval(total_successes, total_n)
        output["methods"][method] = {
            "domain_scores": {domain: round(score, 1) for domain, score in domain_scores.items()},
            "domain_counts_from_logs": domain_counts,
            "successes_by_domain": successes_by_domain,
            "five_domain_weighted_task_sr": round(weighted_task_sr(domain_scores, domain_counts), 1),
            "successes": total_successes,
            "n": total_n,
            "wilson_95": [round(low * 100, 1), round(high * 100, 1)],
            "interval_source": "raw_per_task_logs",
        }
    return output


def recompute_webarena_table(input_csv: str, counts_json: str) -> dict[str, Any]:
    counts_raw = read_json(counts_json)
    counts = _counts_without_metadata(counts_raw)
    methods = _read_webarena_csv(input_csv)
    output: dict[str, Any] = {"counts": counts_raw, "methods": {}, "interval_source": "rounded_percentage_table"}
    for method, scores in methods.items():
        total = weighted_task_sr(scores, counts)
        estimated_successes = round(sum(counts[domain] * (score / 100.0) for domain, score in scores.items()))
        low, high = wilson_interval(estimated_successes, sum(counts[domain] for domain in scores))
        output["methods"][method] = {
            "domain_scores": scores,
            "five_domain_weighted_task_sr": round(total, 1),
            "wilson_95_approx": [round(low * 100, 1), round(high * 100, 1)],
            "estimated_successes_from_rounded_percentages": estimated_successes,
            "interval_source": "approximate_from_rounded_percentages",
        }
    return output


def check_ablation_table(input_csv: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with Path(input_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"variant", "mind2web_cross_website_stepsr", "webarena_total_tasksr"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    full = next((row for row in rows if row["variant"] == "Full HMT"), None)
    if full is None:
        raise ValueError("Ablation table must include a 'Full HMT' row")
    return {
        "num_rows": len(rows),
        "full_hmt": {
            "mind2web_cross_website_stepsr": float(full["mind2web_cross_website_stepsr"]),
            "webarena_total_tasksr": float(full["webarena_total_tasksr"]),
        },
        "rows": rows,
    }


def check_mind2web_main_table(input_csv: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with Path(input_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "method",
            "cross_task_ea",
            "cross_task_af1",
            "cross_task_stepsr",
            "cross_task_tasksr",
            "cross_website_ea",
            "cross_website_af1",
            "cross_website_stepsr",
            "cross_website_tasksr",
            "cross_domain_ea",
            "cross_domain_af1",
            "cross_domain_stepsr",
            "cross_domain_tasksr",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    hmt = next((row for row in rows if row["method"] == "HMT"), None)
    if hmt is None:
        raise ValueError("Mind2Web table must include an HMT row")
    return {
        "num_rows": len(rows),
        "hmt": {
            "cross_task_stepsr": float(hmt["cross_task_stepsr"]),
            "cross_website_stepsr": float(hmt["cross_website_stepsr"]),
            "cross_domain_stepsr": float(hmt["cross_domain_stepsr"]),
        },
        "rows": rows,
    }


def check_webarena_summary_table(input_csv: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with Path(input_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"method", "total_tasksr", "shopping", "cms", "reddit", "gitlab", "maps", "avg_steps"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")
        for row in reader:
            rows.append(row)
    hmt = next((row for row in rows if row["method"] == "HMT"), None)
    if hmt is None:
        raise ValueError("WebArena summary table must include an HMT row")
    return {
        "num_rows": len(rows),
        "hmt": {
            "total_tasksr": float(hmt["total_tasksr"]),
            "avg_steps": float(hmt["avg_steps"]),
        },
        "rows": rows,
    }


def recompute_mind2web_counts(counts_json: str) -> dict[str, Any]:
    counts = read_json(counts_json)
    expected = {"Cross-Task", "Cross-Website", "Cross-Domain"}
    missing = expected - set(counts)
    if missing:
        raise ValueError(f"Missing Mind2Web split counts: {sorted(missing)}")
    return {
        "counts": {split: int(counts[split]) for split in sorted(expected)},
        "note": "Mind2Web splits are handled separately; no cross-split weighted average is computed by default.",
    }
