from __future__ import annotations

"""Prompt-response cache for LLM calls.

The cache is optional. During a run, structured GPT-4 calls can be written with a prompt hash and the JSON response. A later run may set ``replay_only=True`` to fail if any prompt is missing from the cache.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json
import time

from hmt.utils.io import read_jsonl, write_jsonl


def prompt_cache_key(model: str, prompt_name: str, format_name: str, prompt: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(format_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    return digest.hexdigest()


@dataclass
class LLMCacheRecord:
    key: str
    model: str
    prompt_name: str
    format_name: str
    prompt_sha256: str
    response: dict[str, Any]
    created_unix: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model": self.model,
            "prompt_name": self.prompt_name,
            "format_name": self.format_name,
            "prompt_sha256": self.prompt_sha256,
            "response": self.response,
            "created_unix": self.created_unix,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMCacheRecord":
        return cls(
            key=str(data["key"]),
            model=str(data.get("model", "")),
            prompt_name=str(data.get("prompt_name", "")),
            format_name=str(data.get("format_name", "")),
            prompt_sha256=str(data.get("prompt_sha256", "")),
            response=dict(data.get("response", {})),
            created_unix=float(data.get("created_unix", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


class LLMCache:
    def __init__(self, path: str | Path | None = None, replay_only: bool = False, write_through: bool = True) -> None:
        self.path = Path(path) if path else None
        self.replay_only = replay_only
        self.write_through = write_through
        self._records: dict[str, LLMCacheRecord] = {}
        if self.path and self.path.exists():
            for row in read_jsonl(self.path):
                record = LLMCacheRecord.from_dict(row)
                self._records[record.key] = record

    def get(self, key: str) -> dict[str, Any] | None:
        record = self._records.get(key)
        if record is None:
            return None
        return dict(record.response)

    def put(self, record: LLMCacheRecord) -> None:
        self._records[record.key] = record
        if self.path and self.write_through:
            write_jsonl(self.path, [item.to_dict() for item in self._records.values()])

    def require(self, key: str) -> dict[str, Any]:
        cached = self.get(key)
        if cached is None:
            raise RuntimeError(
                "LLM cache replay was requested, but this prompt was not found in the cache. "
                "Disable llm.replay_cache or provide a cache containing this prompt."
            )
        return cached
