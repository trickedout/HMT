from __future__ import annotations

"""Real Mind2Web adapter for HMT.

The adapter is intentionally permissive because Mind2Web has appeared in several
serialized forms: the original JSON files, HuggingFace-style rows, and local
preprocessed JSONL files used by web-agent projects.  The loader normalizes all common
variants into a task/step abstraction consumed by the HMT runtime.  It never
uses source-site DOM identifiers as transferable memory fields; those identifiers
are kept only as candidate IDs for within-page evaluation and optional debug
logs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator
import json
import re

from hmt.core.construction import build_memory
from hmt.core.memory_tree import Episode, HMTConfig, MemoryTree, Trajectory
from hmt.models.embeddings import EmbeddingModel
from hmt.models.openai_client import OpenAIChatClient
from hmt.models.reranker import CrossEncoderReranker
from hmt.runtime.hmt_agent import AgentStepInput, HMTAgent, HMTAgentRuntimeConfig
from hmt.utils.io import read_json, read_jsonl, write_json, write_jsonl

VALID_SPLITS = {"cross_task", "cross_website", "cross_domain", "train", "test", "dev", "validation"}
_OPERATION_ALIASES = {
    "click": "click",
    "CLICK": "click",
    "select": "select",
    "SELECT": "select",
    "type": "type",
    "TYPE": "type",
    "input": "type",
    "INPUT": "type",
    "hover": "hover",
    "HOVER": "hover",
    "press": "press",
    "PRESS": "press",
    "stop": "stop",
    "STOP": "stop",
}


@dataclass
class Mind2WebRunConfig:
    data_path: Path | None
    split: str
    memory_path: Path | None = None
    train_path: Path | None = None
    output_dir: Path = Path("outputs/runs/mind2web")
    use_llm_construction: bool = True
    use_llm_inference: bool = False
    max_tasks: int | None = None


def load_mind2web_config(config: dict[str, Any], split: str) -> Mind2WebRunConfig:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown Mind2Web split {split!r}; expected one of {sorted(VALID_SPLITS)}")
    dataset = config.get("dataset", {})
    benchmark = config.get("mind2web", {})
    data_path = benchmark.get(f"{split}_path") or dataset.get("mind2web_path") or dataset.get("path")
    train_path = benchmark.get("train_path") or dataset.get("mind2web_train_path")
    memory_path = config.get("memory", {}).get("path")
    return Mind2WebRunConfig(
        data_path=Path(data_path) if data_path else None,
        split=split,
        memory_path=Path(memory_path) if memory_path else None,
        train_path=Path(train_path) if train_path else None,
        output_dir=Path(benchmark.get("output_dir", f"outputs/runs/mind2web/{split}")),
        use_llm_construction=bool(benchmark.get("use_llm_construction", True)),
        use_llm_inference=bool(benchmark.get("use_llm_inference", False)),
        max_tasks=benchmark.get("max_tasks"),
    )


def check_mind2web_ready(run_config: Mind2WebRunConfig) -> None:
    if run_config.data_path is None:
        raise RuntimeError("Mind2Web data path is not configured. Set dataset.mind2web_path or mind2web.<split>_path.")
    if not run_config.data_path.exists():
        raise RuntimeError(f"Mind2Web data path does not exist: {run_config.data_path}")


def load_split_counts(counts_path: str | Path) -> dict[str, int]:
    data = read_json(counts_path)
    return {key: int(value) for key, value in data.items()}


def _coerce_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        records: list[dict[str, Any]] = []
        for file in sorted(source.rglob("*.json")) + sorted(source.rglob("*.jsonl")):
            records.extend(_coerce_records(file))
        return records
    if source.suffix.lower() == ".jsonl":
        return read_jsonl(source)
    data = read_json(source)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        for key in ["data", "examples", "records", "tasks", "annotations"]:
            if isinstance(data.get(key), list):
                return [dict(item) for item in data[key]]
        return [data]
    raise ValueError(f"Unsupported Mind2Web data format: {source}")


def _first_present(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _clean_text(value: Any, max_len: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_len]


def _normalize_operation(value: Any) -> str:
    return _OPERATION_ALIASES.get(str(value), str(value).lower() or "click")


def _candidate_id(candidate: dict[str, Any], fallback: str) -> str:
    for key in [
        "element_id",
        "backend_node_id",
        "node_id",
        "uid",
        "id",
        "candidate_id",
        "action_uid",
        "ref",
    ]:
        if candidate.get(key) not in [None, ""]:
            return str(candidate[key])
    attributes = candidate.get("attributes")
    if isinstance(attributes, dict):
        for key in ["backend_node_id", "node_id", "id", "data_pw_testid"]:
            if attributes.get(key):
                return str(attributes[key])
    return fallback


def _candidate_text(candidate: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ["text", "visible_text", "inner_text", "label", "label_or_text", "accessible_name", "aria_label", "name", "value"]:
        if candidate.get(key):
            pieces.append(str(candidate[key]))
    attributes = candidate.get("attributes")
    if isinstance(attributes, dict):
        for key in ["text", "title", "aria-label", "placeholder", "value", "alt", "name"]:
            if attributes.get(key):
                pieces.append(str(attributes[key]))
    return _clean_text(" ".join(pieces), 240)


def _candidate_role(candidate: dict[str, Any]) -> str:
    if candidate.get("role"):
        return str(candidate["role"]).lower()
    if candidate.get("tag"):
        tag = str(candidate["tag"]).lower()
    elif candidate.get("tagName"):
        tag = str(candidate["tagName"]).lower()
    else:
        attrs = candidate.get("attributes") if isinstance(candidate.get("attributes"), dict) else {}
        tag = str(attrs.get("tag") or attrs.get("tagName") or "").lower()
    if tag in {"button"}:
        return "button"
    if tag in {"a"}:
        return "link"
    if tag in {"input", "textarea"}:
        input_type = str(candidate.get("type") or candidate.get("attributes", {}).get("type", "")).lower()
        return "textbox" if input_type not in {"button", "submit", "checkbox", "radio"} else input_type or "input"
    if tag == "select":
        return "combobox"
    return tag or "element"


def _semantic_candidate(candidate: dict[str, Any], fallback: str) -> dict[str, Any]:
    element_id = _candidate_id(candidate, fallback)
    attrs = candidate.get("attributes") if isinstance(candidate.get("attributes"), dict) else {}
    parent = _first_present(candidate, "parent_context", "context", "ancestors", default="")
    siblings = _first_present(candidate, "sibling_text", "sibling_context", "nearby_text", default="")
    bbox = candidate.get("bbox") or candidate.get("bounding_box") or candidate.get("rect")
    text = _candidate_text(candidate)
    role = _candidate_role(candidate)
    return {
        "element_id": element_id,
        "role": role,
        "visible_text": text,
        "label_or_text": text,
        "accessible_name": _clean_text(_first_present(candidate, "accessible_name", "aria_label", "name", default=attrs.get("aria-label", "")), 240),
        "value": _clean_text(_first_present(candidate, "value", default=attrs.get("value", "")), 240),
        "parent_context": _clean_text(parent, 240),
        "sibling_text": _clean_text(siblings, 240),
        "relative_position": _clean_text(_first_present(candidate, "relative_position", "position", default=""), 120),
        "dom_path": _clean_text(_first_present(candidate, "xpath", "css_selector", "selector", "path", default=""), 240),
        "bbox": bbox,
        "clickable": role in {"button", "link", "menuitem", "checkbox", "radio", "input", "option"},
        "editable": role in {"textbox", "searchbox", "combobox", "textarea"},
        "raw_candidate": candidate,
    }


@dataclass
class Mind2WebStep:
    task_id: str
    step_index: int
    instruction: str
    operation: str
    value: str | None
    cleaned_html: str = ""
    raw_html: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    positive_candidate_ids: list[str] = field(default_factory=list)
    action_uid: str = ""
    action_repr: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def observation(self) -> dict[str, Any]:
        text = self.cleaned_html or self.raw_html or self.action_repr
        return {
            "text": text,
            "html": self.cleaned_html or self.raw_html,
            "url": str(self.metadata.get("url", "")),
            "candidates": self.candidates,
        }

    def gold_target(self) -> str | None:
        return self.positive_candidate_ids[0] if self.positive_candidate_ids else None


@dataclass
class Mind2WebTask:
    task_id: str
    website: str
    domain: str
    subdomain: str
    instruction: str
    steps: list[Mind2WebStep]
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_episode(self, success: bool = True) -> Episode:
        trajectory = []
        for step in self.steps:
            gold = self._gold_candidate_for_step(step)
            trajectory.append(
                {
                    "step_index": step.step_index,
                    "observation": step.observation(),
                    "operation": step.operation,
                    "argument": step.value,
                    "target": _target_for_memory(gold, fallback_id=step.gold_target() or f"{step.task_id}_{step.step_index}"),
                    "action_text": step.action_repr,
                }
            )
        return Episode(
            raw_instruction=self.instruction,
            trajectory=Trajectory.from_dicts(trajectory),
            dataset="mind2web",
            episode_id=self.task_id,
            success=success,
            metadata={
                "website": self.website,
                "domain": self.domain,
                "subdomain": self.subdomain,
                "split": self.split,
                **self.metadata,
            },
        )

    def _gold_candidate_for_step(self, step: Mind2WebStep) -> dict[str, Any]:
        gold_ids = set(step.positive_candidate_ids)
        for candidate in step.candidates:
            if str(candidate.get("element_id")) in gold_ids:
                return candidate
        return step.candidates[0] if step.candidates else {}


def _target_for_memory(candidate: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    raw = dict(candidate.get("raw_candidate", {})) if isinstance(candidate.get("raw_candidate"), dict) else {}
    # The transferable descriptor intentionally excludes raw ids/selectors.  The
    # fallback ID is used only for source metadata/debug before public export.
    return {
        "role": candidate.get("role") or _candidate_role(raw),
        "label_or_text": candidate.get("label_or_text") or candidate.get("visible_text") or _candidate_text(raw),
        "accessible_name": candidate.get("accessible_name", ""),
        "parent_context": candidate.get("parent_context", ""),
        "sibling_context": candidate.get("sibling_text", ""),
        "relative_position": candidate.get("relative_position", ""),
        "source_element_id_for_debug": fallback_id,
    }


def _extract_steps(record: dict[str, Any], task_id: str, instruction: str) -> list[Mind2WebStep]:
    actions = _first_present(record, "actions", "action_sequence", "trajectory", "steps", default=[])
    if isinstance(actions, dict):
        actions = list(actions.values())
    action_reprs = record.get("action_reprs") or record.get("action_representations") or []
    steps: list[Mind2WebStep] = []
    for index, action in enumerate(actions or []):
        if not isinstance(action, dict):
            continue
        positive = action.get("pos_candidates") or action.get("positive_candidates") or action.get("gold_candidates") or []
        negative = action.get("neg_candidates") or action.get("negative_candidates") or []
        if isinstance(positive, dict):
            positive = [positive]
        if isinstance(negative, dict):
            negative = [negative]
        raw_candidates = list(positive or []) + list(negative or [])
        candidates = [_semantic_candidate(candidate, f"{task_id}_{index}_{cand_index}") for cand_index, candidate in enumerate(raw_candidates)]
        positive_ids = [_candidate_id(candidate, f"{task_id}_{index}_{cand_index}") for cand_index, candidate in enumerate(positive or [])]
        op = _normalize_operation(_first_present(action, "operation", "op", "action_type", default="click"))
        value = _first_present(action, "value", "argument", "text", "input", default=None)
        action_repr = str(action_reprs[index]) if index < len(action_reprs) else str(action.get("action_repr", action.get("repr", "")))
        steps.append(
            Mind2WebStep(
                task_id=task_id,
                step_index=int(_first_present(action, "step_index", "index", default=index)),
                instruction=instruction,
                operation=op,
                value=None if value is None else str(value),
                cleaned_html=str(_first_present(action, "cleaned_html", "html", default="")),
                raw_html=str(_first_present(action, "raw_html", default="")),
                candidates=candidates,
                positive_candidate_ids=[str(item) for item in positive_ids],
                action_uid=str(_first_present(action, "action_uid", "uid", default="")),
                action_repr=action_repr,
                metadata={
                    "url": _first_present(action, "url", default=record.get("url", "")),
                    "raw_operation": _first_present(action, "operation", "op", "action_type", default=""),
                },
            )
        )
    return steps


def record_to_task(record: dict[str, Any], split: str = "") -> Mind2WebTask:
    task_id = str(_first_present(record, "annotation_id", "task_id", "id", "episode_id", default=""))
    if not task_id:
        task_id = f"mind2web_{abs(hash(json.dumps(record, sort_keys=True, default=str))) % 10_000_000}"
    instruction = str(_first_present(record, "confirmed_task", "raw_instruction", "instruction", "task", "query", default=""))
    website = str(_first_present(record, "website", "site", default=""))
    domain = str(_first_present(record, "domain", default=""))
    subdomain = str(_first_present(record, "subdomain", default=""))
    return Mind2WebTask(
        task_id=task_id,
        website=website,
        domain=domain,
        subdomain=subdomain,
        instruction=instruction,
        steps=_extract_steps(record, task_id, instruction),
        split=split,
        metadata={key: record[key] for key in ["action_reprs", "url"] if key in record},
    )


def record_to_episode(record: dict[str, Any]) -> Episode:
    if "raw_instruction" in record and "trajectory" in record:
        return Episode(
            raw_instruction=str(record["raw_instruction"]),
            trajectory=Trajectory.from_dicts(record.get("trajectory", [])),
            dataset=str(record.get("dataset", "mind2web")),
            episode_id=str(record.get("episode_id", "")),
            success=bool(record.get("success", True)),
            metadata=dict(record.get("metadata", {})),
        )
    return record_to_task(record).to_episode(success=True)


def load_episodes_jsonl(path: str | Path) -> list[Episode]:
    return [record_to_episode(record) for record in _coerce_records(path)]


class Mind2WebDataset:
    def __init__(self, path: str | Path, split: str = "") -> None:
        self.path = Path(path)
        self.split = split

    def __iter__(self) -> Iterator[Mind2WebTask]:
        for record in _coerce_records(self.path):
            task = record_to_task(record, split=self.split)
            if task.steps:
                yield task

    def to_list(self, max_tasks: int | None = None) -> list[Mind2WebTask]:
        tasks = []
        for task in self:
            tasks.append(task)
            if max_tasks is not None and len(tasks) >= max_tasks:
                break
        return tasks

    def to_episodes(self, max_tasks: int | None = None) -> list[Episode]:
        return [task.to_episode(success=True) for task in self.to_list(max_tasks=max_tasks)]


def _tokens_for_f1(text: Any) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())


def _token_f1(pred: Any, gold: Any) -> float:
    pred_tokens = _tokens_for_f1(pred)
    gold_tokens = _tokens_for_f1(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), gold_counts.get(token, 0)) for token in set(pred_counts) | set(gold_counts))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _candidate_by_id(candidates: list[dict[str, Any]], element_id: str | None) -> dict[str, Any]:
    if element_id is None:
        return {}
    for candidate in candidates:
        if str(candidate.get("element_id")) == str(element_id):
            return candidate
    return {}


def _candidate_label(candidate: dict[str, Any]) -> str:
    return _clean_text(
        candidate.get("label_or_text")
        or candidate.get("visible_text")
        or candidate.get("accessible_name")
        or candidate.get("value")
        or "",
        300,
    )


def _action_repr(operation: str, candidate: dict[str, Any], argument: Any = None) -> str:
    pieces = [str(operation or "").lower()]
    label = _candidate_label(candidate)
    role = str(candidate.get("role", ""))
    if role:
        pieces.append(role)
    if label:
        pieces.append(label)
    if argument not in [None, ""]:
        pieces.append(str(argument))
    return " ".join(pieces)


@dataclass
class Mind2WebStepResult:
    task_id: str
    step_index: int
    website: str
    domain: str
    operation_gold: str
    operation_pred: str
    gold_element_ids: list[str]
    pred_element_id: str | None
    value_gold: str | None
    value_pred: str | None
    element_correct: bool
    operation_correct: bool
    value_correct: bool
    step_success: bool
    action_f1: float
    gold_action_repr: str
    pred_action_repr: str
    selected_stage_id: str | None
    confidence: float
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_index": self.step_index,
            "website": self.website,
            "domain": self.domain,
            "operation_gold": self.operation_gold,
            "operation_pred": self.operation_pred,
            "gold_element_ids": self.gold_element_ids,
            "pred_element_id": self.pred_element_id,
            "value_gold": self.value_gold,
            "value_pred": self.value_pred,
            "element_correct": self.element_correct,
            "operation_correct": self.operation_correct,
            "value_correct": self.value_correct,
            "step_success": self.step_success,
            "action_f1": self.action_f1,
            "gold_action_repr": self.gold_action_repr,
            "pred_action_repr": self.pred_action_repr,
            "selected_stage_id": self.selected_stage_id,
            "confidence": self.confidence,
            "debug": self.debug,
        }


@dataclass
class Mind2WebMetrics:
    num_steps: int
    num_tasks: int
    element_accuracy: float
    operation_accuracy: float
    value_accuracy: float
    action_f1: float
    step_success_rate: float
    task_success_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_steps": self.num_steps,
            "num_tasks": self.num_tasks,
            "element_accuracy": self.element_accuracy,
            "operation_accuracy": self.operation_accuracy,
            "value_accuracy": self.value_accuracy,
            "action_f1": self.action_f1,
            "step_success_rate": self.step_success_rate,
            "task_success_rate": self.task_success_rate,
        }


def compute_mind2web_metrics(results: Iterable[Mind2WebStepResult]) -> Mind2WebMetrics:
    rows = list(results)
    n = len(rows)
    if not n:
        return Mind2WebMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    by_task: dict[str, list[Mind2WebStepResult]] = {}
    for row in rows:
        by_task.setdefault(row.task_id, []).append(row)
    task_successes = sum(1 for task_rows in by_task.values() if task_rows and all(row.step_success for row in task_rows))
    return Mind2WebMetrics(
        num_steps=n,
        num_tasks=len(by_task),
        element_accuracy=100.0 * sum(row.element_correct for row in rows) / n,
        operation_accuracy=100.0 * sum(row.operation_correct for row in rows) / n,
        value_accuracy=100.0 * sum(row.value_correct for row in rows) / n,
        action_f1=100.0 * sum(row.action_f1 for row in rows) / n,
        step_success_rate=100.0 * sum(row.step_success for row in rows) / n,
        task_success_rate=100.0 * task_successes / len(by_task) if by_task else 0.0,
    )


class Mind2WebRunner:
    def __init__(
        self,
        config: dict[str, Any],
        run_config: Mind2WebRunConfig,
        llm_client: OpenAIChatClient | None = None,
    ) -> None:
        self.config = config
        self.run_config = run_config
        self.llm_client = llm_client
        self.hmt_config = HMTConfig.from_mapping(config)
        self.embedding_model = EmbeddingModel.from_config(config)
        self.reranker = CrossEncoderReranker.from_config(config)

    def build_memory_from_train(self, output_path: str | Path | None = None) -> Path:
        train_path = self.run_config.train_path or self.run_config.data_path
        if train_path is None:
            raise RuntimeError("No Mind2Web train path is configured for memory construction.")
        dataset = Mind2WebDataset(train_path, split="train")
        episodes = dataset.to_episodes(max_tasks=None)
        if self.run_config.use_llm_construction and self.llm_client is None:
            self.llm_client = OpenAIChatClient.from_config(self.config)
        memory = build_memory(
            episodes,
            config=self.hmt_config,
            client=self.llm_client if self.run_config.use_llm_construction else None,
        )
        target = Path(output_path or self.run_config.memory_path or self.run_config.output_dir / "T_mem_mind2web_train.jsonl")
        memory.save_jsonl(target)
        return target

    def evaluate(self, memory_path: str | Path | None = None) -> dict[str, Any]:
        check_mind2web_ready(self.run_config)
        memory_source = memory_path or self.run_config.memory_path
        if memory_source is None:
            resolved_memory_path = self.build_memory_from_train(None)
        else:
            resolved_memory_path = Path(memory_source)
            if not resolved_memory_path.exists():
                resolved_memory_path = self.build_memory_from_train(resolved_memory_path)
        dataset = Mind2WebDataset(self.run_config.data_path, split=self.run_config.split)
        memory = MemoryTree.load_jsonl(resolved_memory_path)
        inference_cfg = self.config.get("inference", {})
        if self.llm_client is None and (
            self.run_config.use_llm_inference
            or bool(inference_cfg.get("use_llm_normalizer", True))
            or bool(inference_cfg.get("use_base_policy", True))
        ):
            self.llm_client = OpenAIChatClient.from_config(self.config, call_log_path=self.run_config.output_dir / "openai_calls.jsonl")
        runtime_data = dict(self.config)
        runtime_data.setdefault("inference", {})
        runtime_data["inference"]["use_llm_planner"] = self.run_config.use_llm_inference
        runtime_data["inference"]["use_llm_actor"] = self.run_config.use_llm_inference
        runtime = HMTAgentRuntimeConfig.from_mapping(runtime_data)
        agent = HMTAgent(
            memory=memory,
            runtime_config=runtime,
            llm_client=self.llm_client,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
        )
        results: list[Mind2WebStepResult] = []
        for task in dataset.to_list(max_tasks=self.run_config.max_tasks):
            previous_actions: list[dict[str, Any]] = []
            for step in task.steps:
                prediction = agent.predict(
                    AgentStepInput(
                        instruction=task.instruction,
                        observation=step.observation(),
                        previous_actions=previous_actions,
                        candidate_elements=step.candidates,
                        task_id=task.task_id,
                        step_index=step.step_index,
                        metadata={"website": task.website, "domain": task.domain},
                    )
                )
                result = self._score_step(task, step, prediction.to_dict())
                results.append(result)
                previous_actions.append(
                    {
                        "operation": prediction.operation,
                        "target_element_id": prediction.target_element_id,
                        "argument": prediction.argument,
                        "success": result.step_success,
                    }
                )
        metrics = compute_mind2web_metrics(results)
        self.run_config.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.run_config.output_dir / "predictions.jsonl", [row.to_dict() for row in results])
        write_json(self.run_config.output_dir / "metrics.json", metrics.to_dict())
        return {"split": self.run_config.split, "memory_path": str(resolved_memory_path), "metrics": metrics.to_dict()}

    def _score_step(self, task: Mind2WebTask, step: Mind2WebStep, prediction: dict[str, Any]) -> Mind2WebStepResult:
        pred_op = _normalize_operation(prediction.get("operation"))
        gold_op = _normalize_operation(step.operation)
        pred_id = prediction.get("target_element_id")
        gold_ids = [str(item) for item in step.positive_candidate_ids]
        element_correct = pred_id is not None and str(pred_id) in set(gold_ids)
        operation_correct = pred_op == gold_op
        if gold_op in {"type", "select"}:
            value_correct = _clean_text(prediction.get("argument"), 200).lower() == _clean_text(step.value, 200).lower()
        else:
            value_correct = True
        step_success = element_correct and operation_correct and value_correct
        gold_candidate = task._gold_candidate_for_step(step)
        pred_candidate = _candidate_by_id(step.candidates, None if pred_id is None else str(pred_id))
        gold_action_repr = step.action_repr or _action_repr(gold_op, gold_candidate, step.value)
        pred_action_repr = _action_repr(pred_op, pred_candidate, prediction.get("argument"))
        action_f1 = _token_f1(pred_action_repr, gold_action_repr)
        return Mind2WebStepResult(
            task_id=task.task_id,
            step_index=step.step_index,
            website=task.website,
            domain=task.domain,
            operation_gold=gold_op,
            operation_pred=pred_op,
            gold_element_ids=gold_ids,
            pred_element_id=None if pred_id is None else str(pred_id),
            value_gold=step.value,
            value_pred=prediction.get("argument"),
            element_correct=element_correct,
            operation_correct=operation_correct,
            value_correct=value_correct,
            step_success=step_success,
            action_f1=action_f1,
            gold_action_repr=gold_action_repr,
            pred_action_repr=pred_action_repr,
            selected_stage_id=prediction.get("selected_stage_id"),
            confidence=float(prediction.get("confidence", 0.0)),
            debug=prediction.get("debug", {}),
        )
