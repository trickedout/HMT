from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hmt.core.cost import estimate_openai_cost_usd
from hmt.core.structured_output import format_schema_instructions, validate_data
from hmt.models.llm_cache import LLMCache, LLMCacheRecord, prompt_cache_key
from hmt.utils.io import write_jsonl
from hmt.utils.json_utils import extract_json_object


@dataclass
class OpenAIChatClient:
    """Structured JSON GPT-4 client used by HMT.

    The only remote-model credential required by the repository is the OpenAI API
    key used by the OpenAI Python SDK.  The client can log every call and optionally read/write a prompt-response cache.  When
    ``replay_cache`` is true, no remote request is sent; every prompt must be
    present in the cache or the run fails loudly.
    """

    model: str = "gpt-4-0613"
    temperature: float = 0.0
    max_retries: int = 3
    request_timeout: int = 120
    call_log_path: Path | None = None
    input_usd_per_million_tokens: float | None = None
    output_usd_per_million_tokens: float | None = None
    cache_path: Path | None = None
    replay_cache: bool = False
    write_cache: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cache: LLMCache | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: dict[str, Any], call_log_path: str | Path | None = None) -> "OpenAIChatClient":
        llm = config.get("llm", {})
        pricing = llm.get("pricing_usd_per_million_tokens", {})
        cache_path = llm.get("cache_path")
        return cls(
            model=str(llm.get("model", "gpt-4-0613")),
            temperature=float(llm.get("temperature", 0)),
            max_retries=int(llm.get("max_retries", 3)),
            request_timeout=int(llm.get("request_timeout", 120)),
            call_log_path=Path(call_log_path) if call_log_path else None,
            input_usd_per_million_tokens=pricing.get("input"),
            output_usd_per_million_tokens=pricing.get("output"),
            cache_path=Path(cache_path) if cache_path else None,
            replay_cache=bool(llm.get("replay_cache", False)),
            write_cache=bool(llm.get("write_cache", True)),
        )

    @property
    def cache(self) -> LLMCache | None:
        if self.cache_path is None and not self.replay_cache:
            return None
        if self._cache is None:
            self._cache = LLMCache(path=self.cache_path, replay_only=self.replay_cache, write_through=self.write_cache)
        return self._cache

    def complete_json(self, prompt_name: str, variables: dict[str, Any], format_name: str) -> dict[str, Any]:
        from hmt.core.construction import load_prompt

        template = load_prompt(prompt_name)
        prompt = render_prompt(template, variables)
        schema_hint = format_schema_instructions(format_name)
        if schema_hint:
            prompt = f"{prompt}\n\n{schema_hint}"
        key = prompt_cache_key(self.model, prompt_name, format_name, prompt)
        prompt_sha256 = key
        cache = self.cache
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                validate_data(cached, format_name)
                self._record_call(
                    {
                        "prompt_name": prompt_name,
                        "format_name": format_name,
                        "model": self.model,
                        "latency_sec": 0.0,
                        "token_usage": None,
                        "estimated_cost_usd": 0.0,
                        "status": "cache_hit",
                        "cache_key": key,
                        "record_id": variables.get("record_id"),
                    }
                )
                return cached
            if self.replay_cache:
                required = cache.require(key)
                validate_data(required, format_name)
                self._record_call(
                    {
                        "prompt_name": prompt_name,
                        "format_name": format_name,
                        "model": self.model,
                        "latency_sec": 0.0,
                        "token_usage": None,
                        "estimated_cost_usd": 0.0,
                        "status": "cache_replay_miss_error",
                        "cache_key": key,
                        "record_id": variables.get("record_id"),
                    }
                )
                return required

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the optional `openai` package to use OpenAIChatClient.") from exc

        client = OpenAI(timeout=self.request_timeout, max_retries=self.max_retries)
        started = time.perf_counter()
        status = "ok"
        usage = None
        data: dict[str, Any] | None = None
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            usage_obj = getattr(response, "usage", None)
            if usage_obj is not None:
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
            content = response.choices[0].message.content or ""
            data = extract_json_object(content)
            validate_data(data, format_name)
            if cache is not None:
                cache.put(
                    LLMCacheRecord(
                        key=key,
                        model=self.model,
                        prompt_name=prompt_name,
                        format_name=format_name,
                        prompt_sha256=prompt_sha256,
                        response=data,
                        metadata={"record_id": variables.get("record_id")},
                    )
                )
            return data
        except Exception:
            status = "error"
            raise
        finally:
            self._record_call(
                {
                    "prompt_name": prompt_name,
                    "format_name": format_name,
                    "model": self.model,
                    "latency_sec": round(time.perf_counter() - started, 4),
                    "token_usage": usage,
                    "estimated_cost_usd": estimate_openai_cost_usd(
                        usage,
                        self.input_usd_per_million_tokens,
                        self.output_usd_per_million_tokens,
                    ),
                    "status": status,
                    "cache_key": key,
                    "record_id": variables.get("record_id"),
                }
            )

    def _record_call(self, record: dict[str, Any]) -> None:
        self.calls.append(record)
        if self.call_log_path:
            write_jsonl(self.call_log_path, self.calls)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def render_prompt(template: str, variables: dict[str, Any]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{" + key + "}", _stringify(value))
    return rendered
