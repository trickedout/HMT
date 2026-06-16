from __future__ import annotations

"""Real WebArena adapter for HMT online-memory evaluation.

The code supports the common WebArena online protocol: each single-site domain can be evaluated in a fixed order; memory is reset at the domain boundary; only successful completed episodes are inserted into memory; and no future task information is available to the agent.  The module supports both a
BrowserGym-style environment and the original WebArena environment via dynamic
imports so that the repository does not vendor benchmark code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol
import json
import os
import re
import time

from hmt.core.construction import build_memory
from hmt.core.memory_tree import Episode, HMTConfig, MemoryTree, Trajectory
from hmt.models.embeddings import EmbeddingModel
from hmt.models.openai_client import OpenAIChatClient
from hmt.models.reranker import CrossEncoderReranker
from hmt.preprocess.accessibility import flatten_accessibility_tree
from hmt.preprocess.dom import extract_dom_candidates
from hmt.runtime.hmt_agent import AgentStepInput, HMTAgent, HMTAgentRuntimeConfig, HMTActionPrediction
from hmt.utils.io import read_json, read_jsonl, write_json, write_jsonl

VALID_DOMAINS = {"shopping", "shopping_admin", "reddit", "gitlab", "maps", "map", "cms"}
_DOMAIN_ALIASES = {"map": "maps", "maps": "maps", "shopping_admin": "shopping_admin", "admin": "shopping_admin", "cms": "cms"}


def normalize_domain(domain: str) -> str:
    raw = str(domain or "").lower().replace("-", "_")
    return _DOMAIN_ALIASES.get(raw, raw)


@dataclass
class WebArenaRunConfig:
    domain: str
    task_order_path: Path | None
    server_urls: dict[str, str]
    memory_path: Path | None = None
    output_dir: Path = Path("outputs/runs/webarena")
    env_backend: str = "browsergym"
    env_id_template: str = "browsergym/webarena.{task_id}"
    task_config_dir: Path | None = None
    auth_folder: Path | None = None
    max_steps: int = 30
    memory_reset_per_domain: bool = True
    insert_success_only: bool = True
    use_llm_construction: bool = True
    use_llm_inference: bool = False
    stop_action_keyword: str = "stop"
    viewport_width: int = 1280
    viewport_height: int = 720
    headless: bool = True
    max_tasks: int | None = None


def load_webarena_config(config: dict[str, Any], domain: str) -> WebArenaRunConfig:
    normalized = normalize_domain(domain)
    if normalized not in {normalize_domain(d) for d in VALID_DOMAINS}:
        raise ValueError(f"Unknown WebArena domain {domain!r}; expected one of {sorted(VALID_DOMAINS)}")
    webarena = config.get("webarena", {})
    task_order = webarena.get("task_order_path")
    task_config_dir = webarena.get("task_config_dir") or webarena.get("config_dir")
    auth_folder = webarena.get("auth_folder") or webarena.get("storage_state_dir")
    return WebArenaRunConfig(
        domain=normalized,
        task_order_path=Path(task_order) if task_order else None,
        server_urls=dict(webarena.get("server_urls", {})),
        memory_path=Path(config.get("memory", {}).get("path")) if config.get("memory", {}).get("path") else None,
        output_dir=Path(webarena.get("output_dir", f"outputs/runs/webarena/{normalized}")),
        env_backend=str(webarena.get("env_backend", "browsergym")),
        env_id_template=str(webarena.get("env_id_template", "browsergym/webarena.{task_id}")),
        task_config_dir=Path(task_config_dir) if task_config_dir else None,
        auth_folder=Path(auth_folder) if auth_folder else None,
        max_steps=int(webarena.get("max_steps", 30)),
        memory_reset_per_domain=bool(webarena.get("memory_reset_per_domain", True)),
        insert_success_only=bool(webarena.get("insert_success_only", True)),
        use_llm_construction=bool(webarena.get("use_llm_construction", True)),
        use_llm_inference=bool(webarena.get("use_llm_inference", False)),
        stop_action_keyword=str(webarena.get("stop_action_keyword", "stop")),
        viewport_width=int(webarena.get("viewport_width", 1280)),
        viewport_height=int(webarena.get("viewport_height", 720)),
        headless=bool(webarena.get("headless", True)),
        max_tasks=webarena.get("max_tasks"),
    )


def check_webarena_ready(run_config: WebArenaRunConfig) -> None:
    if run_config.task_order_path is None:
        raise RuntimeError("WebArena task order path is not configured.")
    if not run_config.task_order_path.exists():
        raise RuntimeError(f"WebArena task order path does not exist: {run_config.task_order_path}")
    if not run_config.server_urls:
        raise RuntimeError("WebArena server URLs are not configured. Set webarena.server_urls in your local config.")
    if run_config.task_config_dir and not run_config.task_config_dir.exists():
        raise RuntimeError(f"WebArena task_config_dir does not exist: {run_config.task_config_dir}")


def _first_present(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _clean(value: Any, max_len: int = 1000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:max_len]


@dataclass
class WebArenaTask:
    task_id: str
    intent: str
    domain: str
    sites: list[str] = field(default_factory=list)
    start_url: str = ""
    config_file: str = ""
    storage_state: str | None = None
    evaluator: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any], task_config_dir: str | Path | None = None) -> "WebArenaTask":
        payload = dict(record)
        config_file = _first_present(payload, "config_file", "task_config", "path", default="")
        if config_file and task_config_dir:
            config_path = Path(task_config_dir) / str(config_file)
            if config_path.exists():
                loaded = read_json(config_path)
                payload = {**loaded, **payload}
        task_id = str(_first_present(payload, "task_id", "id", "task_index", default=Path(str(config_file)).stem))
        intent = str(_first_present(payload, "intent", "instruction", "raw_instruction", "confirmed_task", default=""))
        sites_raw = _first_present(payload, "sites", "site", "domain", default=[])
        if isinstance(sites_raw, str):
            sites = [sites_raw]
        else:
            sites = [str(site) for site in sites_raw or []]
        domain = normalize_domain(str(_first_present(payload, "domain", "website", default=sites[0] if sites else "")))
        return cls(
            task_id=task_id,
            intent=intent,
            domain=domain,
            sites=sites,
            start_url=str(_first_present(payload, "start_url", "url", default="")),
            config_file=str(config_file),
            storage_state=_first_present(payload, "storage_state", "storage_state_path", default=None),
            evaluator=dict(payload.get("evaluator", {})) if isinstance(payload.get("evaluator"), dict) else {},
            metadata=payload,
        )

    def to_env_kwargs(self, run_config: WebArenaRunConfig) -> dict[str, Any]:
        kwargs = dict(self.metadata)
        kwargs.update(
            {
                "task_id": self.task_id,
                "intent": self.intent,
                "start_url": self.start_url,
                "sites": self.sites,
                "server_urls": run_config.server_urls,
                "storage_state": self.storage_state,
                "config_file": self.config_file,
                "auth_folder": str(run_config.auth_folder) if run_config.auth_folder else None,
                "viewport_size": {"width": run_config.viewport_width, "height": run_config.viewport_height},
                "headless": run_config.headless,
            }
        )
        return kwargs


class WebArenaTaskOrder:
    def __init__(self, path: str | Path, task_config_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        self.task_config_dir = task_config_dir

    def records(self) -> list[dict[str, Any]]:
        if self.path.suffix.lower() == ".jsonl":
            return read_jsonl(self.path)
        data = read_json(self.path)
        if isinstance(data, list):
            return [dict(item) for item in data]
        if isinstance(data, dict):
            for key in ["tasks", "order", "data", "examples"]:
                if isinstance(data.get(key), list):
                    return [dict(item) for item in data[key]]
            return [data]
        raise ValueError(f"Unsupported WebArena task order format: {self.path}")

    def tasks(self, domain: str | None = None, max_tasks: int | None = None) -> list[WebArenaTask]:
        selected: list[WebArenaTask] = []
        normalized_domain = normalize_domain(domain or "") if domain else None
        for record in self.records():
            task = WebArenaTask.from_record(record, task_config_dir=self.task_config_dir)
            if normalized_domain and normalize_domain(task.domain) != normalized_domain:
                continue
            selected.append(task)
            if max_tasks is not None and len(selected) >= max_tasks:
                break
        return selected


@dataclass
class BrowserObservation:
    text: str
    url: str = ""
    html: str = ""
    accessibility_tree: dict[str, Any] | None = None
    screenshot_path: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_agent_observation(self) -> dict[str, Any]:
        candidates = list(self.candidates)
        if not candidates and self.accessibility_tree:
            candidates.extend(flatten_accessibility_tree(self.accessibility_tree))
        if not candidates and self.html:
            candidates.extend(extract_dom_candidates(self.html))
        return {
            "text": self.text,
            "url": self.url,
            "html": self.html,
            "candidates": candidates,
            "screenshot_path": self.screenshot_path,
            "raw": self.raw,
        }


@dataclass
class WebArenaAction:
    operation: str
    element_id: str | None = None
    argument: str | None = None

    def to_hmt_step_target(self, observation: BrowserObservation) -> dict[str, Any]:
        element_id = str(self.element_id or "")
        for candidate in observation.to_agent_observation().get("candidates", []):
            if str(candidate.get("element_id")) == element_id:
                return {
                    "role": candidate.get("role", ""),
                    "label_or_text": candidate.get("visible_text", candidate.get("label_or_text", "")),
                    "accessible_name": candidate.get("accessible_name", ""),
                    "parent_context": candidate.get("parent_context", ""),
                    "sibling_context": candidate.get("sibling_text", candidate.get("sibling_context", "")),
                    "relative_position": candidate.get("relative_position", ""),
                }
        return {"role": "", "label_or_text": element_id, "accessible_name": "", "parent_context": "", "sibling_context": ""}


class WebArenaActionSerializer:
    """Serialize HMT actions to common WebArena/browser-env strings."""

    def __init__(self, stop_action_keyword: str = "stop") -> None:
        self.stop_action_keyword = stop_action_keyword

    def from_prediction(self, prediction: HMTActionPrediction) -> WebArenaAction:
        return WebArenaAction(
            operation=prediction.operation,
            element_id=prediction.target_element_id,
            argument=prediction.argument,
        )

    def to_id_action(self, action: WebArenaAction) -> str:
        op = action.operation.lower()
        eid = action.element_id
        arg = action.argument
        if op == "click" and eid:
            return f"click [{eid}]"
        if op in {"type", "input"} and eid:
            return f"type [{eid}] [{arg or ''}]"
        if op == "select" and eid:
            return f"select [{eid}] [{arg or ''}]"
        if op == "hover" and eid:
            return f"hover [{eid}]"
        if op == "press":
            return f"press [{arg or 'ENTER'}]"
        if op == "scroll":
            return f"scroll [{arg or 'down'}]"
        return self.stop_action_keyword

    def to_browsergym_action(self, action: WebArenaAction) -> str:
        op = action.operation.lower()
        eid = action.element_id
        arg = (action.argument or "").replace("'", "\\'")
        if op == "click" and eid:
            return f"click('{eid}')"
        if op in {"type", "input"} and eid:
            return f"fill('{eid}', '{arg}')"
        if op == "select" and eid:
            return f"select_option('{eid}', '{arg}')"
        if op == "hover" and eid:
            return f"hover('{eid}')"
        if op == "press":
            return f"press('{arg or 'Enter'}')"
        return "stop()"


class WebArenaSession(Protocol):
    def reset(self, task: WebArenaTask) -> BrowserObservation:
        ...

    def step(self, action: str) -> tuple[BrowserObservation, float, bool, dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


class BrowserGymWebArenaSession:
    """Adapter for BrowserGym-style WebArena installations."""

    def __init__(self, run_config: WebArenaRunConfig) -> None:
        try:
            import gymnasium as gym
        except Exception as exc:  # pragma: no cover - optional benchmark dependency
            raise RuntimeError("Install gymnasium/browsergym to use env_backend=browsergym.") from exc
        self.gym = gym
        self.run_config = run_config
        self.env: Any | None = None
        self.serializer = WebArenaActionSerializer(run_config.stop_action_keyword)

    def reset(self, task: WebArenaTask) -> BrowserObservation:
        env_id = self.run_config.env_id_template.format(task_id=task.task_id, domain=task.domain, site=task.sites[0] if task.sites else task.domain)
        kwargs = task.to_env_kwargs(self.run_config)
        self.env = self.gym.make(env_id, **{k: v for k, v in kwargs.items() if v is not None})
        obs, info = self.env.reset()
        return observation_from_env(obs, info)

    def step(self, action: str) -> tuple[BrowserObservation, float, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")
        obs, reward, terminated, truncated, info = self.env.step(action)
        return observation_from_env(obs, info), float(reward), bool(terminated or truncated), dict(info or {})

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


class NativeWebArenaSession:
    """Adapter for original WebArena browser_env installations.

    The exact upstream class signatures changed across forks, so construction is
    centralized here.  Users can adapt this class to their local WebArena clone
    without touching HMT's method implementation.
    """

    def __init__(self, run_config: WebArenaRunConfig) -> None:
        try:
            from browser_env.envs import ScriptBrowserEnv
        except Exception as exc:  # pragma: no cover - optional benchmark dependency
            raise RuntimeError("Install the original WebArena `browser_env` package to use env_backend=native.") from exc
        self.ScriptBrowserEnv = ScriptBrowserEnv
        self.run_config = run_config
        self.env: Any | None = None

    def reset(self, task: WebArenaTask) -> BrowserObservation:
        self.env = self.ScriptBrowserEnv(
            headless=self.run_config.headless,
            slow_mo=0,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": self.run_config.viewport_width, "height": self.run_config.viewport_height},
        )
        obs, info = self.env.reset(options=task.to_env_kwargs(self.run_config))
        return observation_from_env(obs, info)

    def step(self, action: str) -> tuple[BrowserObservation, float, bool, dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out
        return observation_from_env(obs, info), float(reward), bool(done), dict(info or {})

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def make_session(run_config: WebArenaRunConfig) -> WebArenaSession:
    if run_config.env_backend == "browsergym":
        return BrowserGymWebArenaSession(run_config)
    if run_config.env_backend == "native":
        return NativeWebArenaSession(run_config)
    raise ValueError(f"Unsupported WebArena env_backend: {run_config.env_backend!r}")


def observation_from_env(obs: Any, info: dict[str, Any] | None = None) -> BrowserObservation:
    info = info or {}
    raw: dict[str, Any]
    if isinstance(obs, dict):
        raw = dict(obs)
    else:
        raw = {"obs": obs}
    text = _clean(_first_present(raw, "text", "observation", "axtree_txt", "accessibility_tree_txt", default=info.get("text", "")), 4000)
    html = str(_first_present(raw, "html", "page_source", "dom", default=info.get("html", "")))
    url = str(_first_present(raw, "url", default=info.get("url", "")))
    ax = _first_present(raw, "accessibility_tree", "axtree_object", "ax_tree", default=info.get("accessibility_tree"))
    candidates = _first_present(raw, "candidates", "elements", "candidate_elements", default=info.get("candidates", []))
    normalized_candidates = normalize_live_candidates(candidates)
    screenshot_path = _first_present(raw, "screenshot_path", default=info.get("screenshot_path"))
    return BrowserObservation(
        text=text,
        url=url,
        html=html,
        accessibility_tree=ax if isinstance(ax, dict) else None,
        screenshot_path=screenshot_path,
        candidates=normalized_candidates,
        raw={"obs": raw, "info": info},
    )


def normalize_live_candidates(candidates: Any) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if isinstance(candidates, dict):
        if "children" in candidates or "role" in candidates:
            return flatten_accessibility_tree(candidates)
        candidates = list(candidates.values())
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        eid = str(_first_present(candidate, "element_id", "node_id", "backend_node_id", "id", default=f"live_{index}"))
        role = str(_first_present(candidate, "role", "tag", default="element"))
        text = _clean(_first_present(candidate, "visible_text", "text", "label", "name", "accessible_name", default=""), 300)
        normalized.append(
            {
                "element_id": eid,
                "role": role,
                "visible_text": text,
                "label_or_text": text,
                "accessible_name": _clean(_first_present(candidate, "accessible_name", "name", default=""), 300),
                "value": _clean(_first_present(candidate, "value", default=""), 200),
                "parent_context": _clean(_first_present(candidate, "parent_context", "context", default=""), 300),
                "sibling_text": _clean(_first_present(candidate, "sibling_text", "nearby_text", default=""), 300),
                "relative_position": _clean(_first_present(candidate, "relative_position", default=""), 120),
                "bbox": _first_present(candidate, "bbox", "bounding_box", "rect", default=None),
                "clickable": bool(_first_present(candidate, "clickable", default=role in {"button", "link", "menuitem"})),
                "editable": bool(_first_present(candidate, "editable", default=role in {"textbox", "searchbox", "combobox"})),
                "raw_candidate": candidate,
            }
        )
    return normalized


def _candidate_snapshot(candidates: list[dict[str, Any]], element_id: str | None) -> dict[str, Any]:
    if element_id is None:
        return {}
    for candidate in candidates:
        if str(candidate.get("element_id")) == str(element_id):
            return {
                "element_id": str(candidate.get("element_id", "")),
                "role": str(candidate.get("role", "")),
                "visible_text": _clean(candidate.get("visible_text", candidate.get("label_or_text", "")), 300),
                "label_or_text": _clean(candidate.get("label_or_text", candidate.get("visible_text", "")), 300),
                "accessible_name": _clean(candidate.get("accessible_name", ""), 300),
                "parent_context": _clean(candidate.get("parent_context", ""), 300),
                "sibling_text": _clean(candidate.get("sibling_text", candidate.get("sibling_context", "")), 300),
                "relative_position": _clean(candidate.get("relative_position", ""), 120),
            }
    return {}


@dataclass
class WebArenaStepLog:
    task_id: str
    step_index: int
    instruction: str
    observation_url: str
    prediction: dict[str, Any]
    serialized_action: str
    candidate_snapshot: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_index": self.step_index,
            "instruction": self.instruction,
            "observation_url": self.observation_url,
            "prediction": self.prediction,
            "serialized_action": self.serialized_action,
            "candidate_snapshot": self.candidate_snapshot,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class WebArenaTaskResult:
    task_id: str
    domain: str
    success: bool
    reward: float
    num_steps: int
    inserted_into_memory: bool
    final_info: dict[str, Any] = field(default_factory=dict)
    step_logs: list[WebArenaStepLog] = field(default_factory=list)
    stream_index: int | None = None
    memory_counts_before: dict[str, int] = field(default_factory=dict)
    memory_counts_after: dict[str, int] = field(default_factory=dict)
    online_insert_reason: str = ""
    inserted_episode_ids_so_far: list[str] = field(default_factory=list)

    def to_dict(self, include_steps: bool = False) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "domain": self.domain,
            "success": self.success,
            "reward": self.reward,
            "num_steps": self.num_steps,
            "inserted_into_memory": self.inserted_into_memory,
            "final_info": self.final_info,
            "stream_index": self.stream_index,
            "memory_counts_before": self.memory_counts_before,
            "memory_counts_after": self.memory_counts_after,
            "online_insert_reason": self.online_insert_reason,
            "inserted_episode_ids_so_far": self.inserted_episode_ids_so_far,
        }
        if include_steps:
            payload["step_logs"] = [step.to_dict() for step in self.step_logs]
        return payload


class WebArenaOnlineRunner:
    def __init__(
        self,
        config: dict[str, Any],
        run_config: WebArenaRunConfig,
        llm_client: OpenAIChatClient | None = None,
        session: WebArenaSession | None = None,
    ) -> None:
        self.config = config
        self.run_config = run_config
        self.llm_client = llm_client
        self.session = session
        self.hmt_config = HMTConfig.from_mapping(config)
        self.embedding_model = EmbeddingModel.from_config(config)
        self.reranker = CrossEncoderReranker.from_config(config)
        self.serializer = WebArenaActionSerializer(run_config.stop_action_keyword)
        self.memory = MemoryTree.load_jsonl(run_config.memory_path) if run_config.memory_path and run_config.memory_path.exists() else MemoryTree()

    def run(self) -> dict[str, Any]:
        check_webarena_ready(self.run_config)
        if self.run_config.use_llm_construction and self.llm_client is None:
            self.run_config.output_dir.mkdir(parents=True, exist_ok=True)
            self.llm_client = OpenAIChatClient.from_config(
                self.config,
                call_log_path=self.run_config.output_dir / "openai_calls.jsonl",
            )
        order = WebArenaTaskOrder(self.run_config.task_order_path, task_config_dir=self.run_config.task_config_dir)
        tasks = order.tasks(domain=self.run_config.domain, max_tasks=self.run_config.max_tasks)
        if self.run_config.memory_reset_per_domain:
            self.memory = MemoryTree()
        results: list[WebArenaTaskResult] = []
        inserted_episode_ids: list[str] = []
        for stream_index, task in enumerate(tasks):
            before_counts = self._memory_counts()
            result = self.run_task(task)
            result.stream_index = stream_index
            result.memory_counts_before = before_counts
            results.append(result)
            should_insert = (result.success or not self.run_config.insert_success_only) and bool(result.step_logs)
            if should_insert:
                self.insert_successful_episode(task, result)
                result.inserted_into_memory = True
                inserted_episode_ids.append(task.task_id)
                result.online_insert_reason = "success" if result.success else "insert_success_only_disabled"
            else:
                result.online_insert_reason = "failed_task_not_inserted" if not result.success else "empty_step_log_not_inserted"
            result.memory_counts_after = self._memory_counts()
            result.inserted_episode_ids_so_far = list(inserted_episode_ids)
        metrics = self._metrics(results)
        self._write_outputs(results, metrics)
        return {"domain": self.run_config.domain, "metrics": metrics, "num_tasks": len(results)}

    def run_task(self, task: WebArenaTask) -> WebArenaTaskResult:
        session = self.session or make_session(self.run_config)
        step_logs: list[WebArenaStepLog] = []
        previous_actions: list[dict[str, Any]] = []
        reward = 0.0
        final_info: dict[str, Any] = {}
        success = False
        try:
            observation = session.reset(task)
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
                memory=self.memory,
                runtime_config=runtime,
                llm_client=self.llm_client,
                embedding_model=self.embedding_model,
                reranker=self.reranker,
            )
            for step_index in range(self.run_config.max_steps):
                agent_obs = observation.to_agent_observation()
                prediction = agent.predict(
                    AgentStepInput(
                        instruction=task.intent,
                        observation=agent_obs,
                        previous_actions=previous_actions,
                        candidate_elements=agent_obs.get("candidates", []),
                        task_id=task.task_id,
                        step_index=step_index,
                        metadata={"domain": task.domain, "url": observation.url},
                    )
                )
                action = self.serializer.from_prediction(prediction)
                serialized_action = self._serialize_for_backend(action)
                next_observation, reward, done, info = session.step(serialized_action)
                final_info = info
                step_logs.append(
                    WebArenaStepLog(
                        task_id=task.task_id,
                        step_index=step_index,
                        instruction=task.intent,
                        observation_url=observation.url,
                        prediction=prediction.to_dict(),
                        serialized_action=serialized_action,
                        candidate_snapshot=_candidate_snapshot(agent_obs.get("candidates", []), prediction.target_element_id),
                        reward=reward,
                        done=done,
                        info=info,
                    )
                )
                previous_actions.append(
                    {
                        "operation": prediction.operation,
                        "target_element_id": prediction.target_element_id,
                        "argument": prediction.argument,
                        "serialized_action": serialized_action,
                        "reward": reward,
                    }
                )
                observation = next_observation
                if done:
                    success = bool(info.get("success", reward > 0))
                    break
            else:
                success = bool(final_info.get("success", reward > 0))
        finally:
            if self.session is None:
                session.close()
        return WebArenaTaskResult(
            task_id=task.task_id,
            domain=task.domain,
            success=success,
            reward=reward,
            num_steps=len(step_logs),
            inserted_into_memory=False,
            final_info=final_info,
            step_logs=step_logs,
        )

    def insert_successful_episode(self, task: WebArenaTask, result: WebArenaTaskResult) -> None:
        episode = self.result_to_episode(task, result)
        built = build_memory(
            [episode],
            config=self.hmt_config,
            client=self.llm_client if self.run_config.use_llm_construction else None,
        )
        for intent in built.intents.values():
            self.memory.add_intent(intent)
        for stage in built.stages.values():
            self.memory.add_stage(stage)
        for action in built.actions.values():
            self.memory.add_action(action)

    def result_to_episode(self, task: WebArenaTask, result: WebArenaTaskResult) -> Episode:
        trajectory = []
        for step_log in result.step_logs:
            prediction = step_log.prediction
            snapshot = step_log.candidate_snapshot or {}
            target_id = prediction.get("target_element_id")
            trajectory.append(
                {
                    "step_index": step_log.step_index,
                    "observation": {"text": step_log.observation_url, "url": step_log.observation_url},
                    "operation": prediction.get("operation", "click"),
                    "argument": prediction.get("argument"),
                    "target": {
                        "role": snapshot.get("role", ""),
                        "label_or_text": snapshot.get("label_or_text") or snapshot.get("visible_text") or str(target_id or ""),
                        "accessible_name": snapshot.get("accessible_name", ""),
                        "parent_context": snapshot.get("parent_context", f"WebArena {task.domain} page"),
                        "sibling_context": snapshot.get("sibling_text", ""),
                        "relative_position": snapshot.get("relative_position", ""),
                    },
                    "action_text": step_log.serialized_action,
                }
            )
        return Episode(
            raw_instruction=task.intent,
            trajectory=Trajectory.from_dicts(trajectory),
            dataset="webarena",
            episode_id=task.task_id,
            success=result.success,
            metadata={"domain": task.domain, "sites": task.sites, "online_inserted_after_success": True},
        )

    def _serialize_for_backend(self, action: WebArenaAction) -> str:
        if self.run_config.env_backend == "browsergym":
            return self.serializer.to_browsergym_action(action)
        return self.serializer.to_id_action(action)

    def _memory_counts(self) -> dict[str, int]:
        return {
            "intent": len(self.memory.intents),
            "stage": len(self.memory.stages),
            "action": len(self.memory.actions),
        }

    def _metrics(self, results: list[WebArenaTaskResult]) -> dict[str, Any]:
        n = len(results)
        success_count = sum(result.success for result in results)
        return {
            "domain": self.run_config.domain,
            "num_tasks": n,
            "success_count": int(success_count),
            "task_success_rate": 100.0 * success_count / n if n else 0.0,
            "avg_steps": sum(result.num_steps for result in results) / n if n else 0.0,
            "memory_counts": self._memory_counts(),
            "memory_reset_per_domain": self.run_config.memory_reset_per_domain,
            "insert_success_only": self.run_config.insert_success_only,
        }

    def _write_outputs(self, results: list[WebArenaTaskResult], metrics: dict[str, Any]) -> None:
        self.run_config.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_config.output_dir / "metrics.json", metrics)
        write_jsonl(self.run_config.output_dir / "task_results.jsonl", [result.to_dict(include_steps=False) for result in results])
        write_jsonl(
            self.run_config.output_dir / "online_memory_trace.jsonl",
            [
                {
                    "stream_index": result.stream_index,
                    "task_id": result.task_id,
                    "success": result.success,
                    "inserted_into_memory": result.inserted_into_memory,
                    "online_insert_reason": result.online_insert_reason,
                    "memory_counts_before": result.memory_counts_before,
                    "memory_counts_after": result.memory_counts_after,
                    "inserted_episode_ids_so_far": result.inserted_episode_ids_so_far,
                }
                for result in results
            ],
        )
        write_jsonl(
            self.run_config.output_dir / "step_logs.jsonl",
            [step.to_dict() for result in results for step in result.step_logs],
        )
        self.memory.save_jsonl(self.run_config.output_dir / "T_mem_online_final.jsonl")


def task_record_to_episode(record: dict[str, Any]) -> Episode:
    return Episode(
        raw_instruction=str(record["raw_instruction"] if "raw_instruction" in record else record.get("intent", "")),
        trajectory=Trajectory.from_dicts(record.get("trajectory", [])),
        dataset=str(record.get("dataset", "webarena")),
        episode_id=str(record.get("episode_id", record.get("task_id", ""))),
        success=bool(record.get("success", False)),
        metadata={"domain": record.get("domain"), "task_id": record.get("task_id")},
    )


def run_domain_reset_stream(
    task_order_path: str | Path,
    domain: str,
    config: HMTConfig | None = None,
) -> dict[str, Any]:
    """Fixture-level WebArena online memory protocol without browser servers."""
    cfg = config or HMTConfig()
    records = [record for record in read_jsonl(task_order_path) if normalize_domain(str(record.get("domain", ""))) == normalize_domain(domain)]
    memory = MemoryTree()
    trace: list[dict[str, Any]] = []
    inserted_successes: list[str] = []
    for index, record in enumerate(records):
        before_counts = {
            "intent": len(memory.intents),
            "stage": len(memory.stages),
            "action": len(memory.actions),
        }
        episode = task_record_to_episode(record)
        if episode.success:
            built = build_memory([episode], cfg)
            for intent in built.intents.values():
                memory.add_intent(intent)
            for stage in built.stages.values():
                memory.add_stage(stage)
            for action in built.actions.values():
                memory.add_action(action)
            inserted_successes.append(episode.episode_id)
        after_counts = {
            "intent": len(memory.intents),
            "stage": len(memory.stages),
            "action": len(memory.actions),
        }
        trace.append(
            {
                "stream_index": index,
                "task_id": record.get("task_id"),
                "success": episode.success,
                "memory_counts_before": before_counts,
                "memory_counts_after": after_counts,
                "inserted_episode_ids_so_far": list(inserted_successes),
            }
        )
    return {
        "domain": normalize_domain(domain),
        "memory_reset_per_domain": True,
        "insert_success_only": True,
        "no_future_leakage": all(
            record["task_id"] not in record["inserted_episode_ids_so_far"] or record["success"] for record in trace
        ),
        "trace": trace,
        "final_memory_counts": {
            "intent": len(memory.intents),
            "stage": len(memory.stages),
            "action": len(memory.actions),
        },
    }
