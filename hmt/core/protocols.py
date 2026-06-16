from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import json

from hmt.utils.io import read_jsonl, write_json, write_jsonl


@dataclass
class Mind2WebPredictionRecord:
    task_id: str
    step_index: int
    split: str
    domain: str = ""
    website: str = ""
    operation_gold: str = ""
    operation_pred: str = ""
    element_gold: list[str] = field(default_factory=list)
    element_pred: str | None = None
    value_gold: str | None = None
    value_pred: str | None = None
    action_repr_gold: str = ""
    action_repr_pred: str = ""
    confidence: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_index": self.step_index,
            "split": self.split,
            "domain": self.domain,
            "website": self.website,
            "operation_gold": self.operation_gold,
            "operation_pred": self.operation_pred,
            "element_gold": self.element_gold,
            "element_pred": self.element_pred,
            "value_gold": self.value_gold,
            "value_pred": self.value_pred,
            "action_repr_gold": self.action_repr_gold,
            "action_repr_pred": self.action_repr_pred,
            "confidence": self.confidence,
            "debug": self.debug,
        }


@dataclass
class WebArenaTaskRecord:
    task_id: str
    domain: str
    instruction: str
    success: bool
    num_steps: int
    task_config_path: str = ""
    order_index: int | None = None
    final_answer: str | None = None
    score: float | None = None
    trace_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "instruction": self.instruction,
            "success": self.success,
            "num_steps": self.num_steps,
            "task_config_path": self.task_config_path,
            "order_index": self.order_index,
            "final_answer": self.final_answer,
            "score": self.score,
            "trace_path": self.trace_path,
            "metadata": self.metadata,
        }


class ResultWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_records: list[dict[str, Any]] = []
        self.task_records: list[dict[str, Any]] = []

    def add_step(self, record: Mind2WebPredictionRecord | dict[str, Any]) -> None:
        data = record.to_record() if hasattr(record, "to_record") else dict(record)
        self.step_records.append(data)

    def add_task(self, record: WebArenaTaskRecord | dict[str, Any]) -> None:
        data = record.to_record() if hasattr(record, "to_record") else dict(record)
        self.task_records.append(data)

    def flush(self) -> None:
        if self.step_records:
            write_jsonl(self.output_dir / "predictions.jsonl", self.step_records)
            write_json(self.output_dir / "metrics.json", mind2web_metrics_from_records(self.step_records))
        if self.task_records:
            write_jsonl(self.output_dir / "task_results.jsonl", self.task_records)
            write_json(self.output_dir / "metrics.json", webarena_metrics_from_records(self.task_records))


def mind2web_metrics_from_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {"num_steps": 0, "element_accuracy": 0.0, "operation_accuracy": 0.0, "value_accuracy": 0.0, "action_f1": 0.0, "step_success_rate": 0.0, "task_success_rate": 0.0}
    element_correct = 0
    op_correct = 0
    value_correct = 0
    action_f1_sum = 0.0
    step_success = 0
    task_success: dict[str, bool] = {}
    for row in rows:
        gold_elements = {str(x) for x in row.get("element_gold", [])}
        pred_element = row.get("element_pred")
        e_ok = bool(pred_element is not None and str(pred_element) in gold_elements) if gold_elements else bool(pred_element is None)
        o_ok = str(row.get("operation_gold", "")).lower() == str(row.get("operation_pred", "")).lower()
        vg = row.get("value_gold")
        vp = row.get("value_pred")
        v_ok = (vg is None or vg == "") or str(vg).strip() == str(vp or "").strip()
        element_correct += int(e_ok)
        op_correct += int(o_ok)
        value_correct += int(v_ok)
        f1_parts = [float(e_ok), float(o_ok), float(v_ok)]
        f1 = sum(f1_parts) / len(f1_parts)
        action_f1_sum += f1
        s_ok = e_ok and o_ok and v_ok
        step_success += int(s_ok)
        task_id = str(row.get("task_id", ""))
        task_success[task_id] = task_success.get(task_id, True) and s_ok
    num_tasks = len(task_success) or 1
    return {
        "num_steps": n,
        "num_tasks": len(task_success),
        "element_accuracy": element_correct / n,
        "operation_accuracy": op_correct / n,
        "value_accuracy": value_correct / n,
        "action_f1": action_f1_sum / n,
        "step_success_rate": step_success / n,
        "task_success_rate": sum(1 for ok in task_success.values() if ok) / num_tasks,
    }


def webarena_metrics_from_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(str(row.get("domain", "unknown")), []).append(row)
    domain_metrics: dict[str, dict[str, Any]] = {}
    for domain, items in by_domain.items():
        total = len(items)
        success = sum(1 for item in items if bool(item.get("success", False)))
        steps = [int(item.get("num_steps", 0)) for item in items]
        domain_metrics[domain] = {
            "num_tasks": total,
            "successes": success,
            "task_success_rate": success / total if total else 0.0,
            "avg_steps": sum(steps) / len(steps) if steps else 0.0,
        }
    weighted_total = sum(m["task_success_rate"] * m["num_tasks"] for m in domain_metrics.values())
    total_tasks = sum(m["num_tasks"] for m in domain_metrics.values())
    return {
        "num_tasks": total_tasks,
        "task_success_rate": weighted_total / total_tasks if total_tasks else 0.0,
        "domain_metrics": domain_metrics,
    }


def load_task_order(path: str | Path, domain: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if domain is not None:
        rows = [row for row in rows if str(row.get("domain")) == domain]
    rows.sort(key=lambda item: (int(item.get("order_index", item.get("task_id", 0))) if str(item.get("order_index", item.get("task_id", 0))).isdigit() else str(item.get("order_index", item.get("task_id", "")))))
    return rows


def write_task_order(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    normalized = []
    for index, row in enumerate(rows):
        data = dict(row)
        data.setdefault("order_index", index)
        normalized.append(data)
    write_jsonl(path, normalized)
