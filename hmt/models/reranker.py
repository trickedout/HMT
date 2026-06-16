from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hmt.models.qwen_transformers import DeterministicOverlapReranker, QwenCrossEncoderReranker, QwenRuntimeConfig


@dataclass
class CrossEncoderReranker:
    """Backward-compatible Qwen reranker wrapper using local transformers."""

    model: str = "Qwen/Qwen3-Reranker-0.6B"
    allow_fallback: bool = False
    backend: str = "qwen_transformers"
    device: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = False
    cache_dir: str | None = None
    max_length: int = 8192
    batch_size: int = 8
    score_activation: str = "identity"
    _reranker: Any = field(default=None, init=False, repr=False)

    def _build(self) -> Any:
        if self._reranker is not None:
            return self._reranker
        runtime = QwenRuntimeConfig(
            model_name_or_path=self.model,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        try:
            self._reranker = QwenCrossEncoderReranker(config=runtime, score_activation=self.score_activation)
            self.backend = getattr(self._reranker, "backend", "qwen_transformers")
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "Qwen reranking requires local `torch` and `transformers` plus a loadable "
                    f"{self.model!r} model. Set retrieval.allow_model_fallback=true only for debugging."
                ) from exc
            self._reranker = DeterministicOverlapReranker()
            self.backend = "deterministic_fallback"
        return self._reranker

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CrossEncoderReranker":
        reranker = config.get("reranker", {})
        retrieval = config.get("retrieval", {})
        return cls(
            model=str(reranker.get("model", "Qwen/Qwen3-Reranker-0.6B")),
            allow_fallback=bool(reranker.get("allow_fallback", retrieval.get("allow_model_fallback", False))),
            device=reranker.get("device"),
            dtype=str(reranker.get("dtype", "auto")),
            trust_remote_code=bool(reranker.get("trust_remote_code", True)),
            local_files_only=bool(reranker.get("local_files_only", False)),
            cache_dir=reranker.get("cache_dir"),
            max_length=int(reranker.get("max_length", 8192)),
            batch_size=int(reranker.get("batch_size", 8)),
            score_activation=str(reranker.get("score_activation", "identity")),
        )

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        return self._build().rerank(query, candidates, top_k)
