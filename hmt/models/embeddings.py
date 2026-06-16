from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Any

from hmt.models.qwen_transformers import DeterministicHashEncoder, QwenDenseEncoder, QwenRuntimeConfig


@dataclass
class EmbeddingModel:
    """Backward-compatible embedding wrapper using local transformers.

    Previous releases exposed ``EmbeddingModel.encode``.  The public API is kept,
    but the implementation now loads Qwen through Hugging Face transformers, as
    used by HMT. No embedding API key is used.
    """

    model: str = "Qwen/Qwen3-Embedding-0.6B"
    normalize: bool = True
    allow_fallback: bool = False
    backend: str = "qwen_transformers"
    device: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = False
    cache_dir: str | None = None
    max_length: int = 8192
    batch_size: int = 8
    _encoder: Any = field(default=None, init=False, repr=False)

    def _build(self) -> Any:
        if self._encoder is not None:
            return self._encoder
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
            self._encoder = QwenDenseEncoder(config=runtime, normalize=self.normalize)
            self.backend = getattr(self._encoder, "backend", "qwen_transformers")
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "Qwen embedding retrieval requires local `torch` and `transformers` plus a loadable "
                    f"{self.model!r} model. Set retrieval.allow_model_fallback=true only for debugging."
                ) from exc
            self._encoder = DeterministicHashEncoder(normalize=self.normalize)
            self.backend = "deterministic_fallback"
        return self._encoder

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingModel":
        embedding = config.get("embedding", {})
        retrieval = config.get("retrieval", {})
        return cls(
            model=str(embedding.get("model", "Qwen/Qwen3-Embedding-0.6B")),
            normalize=bool(embedding.get("normalize", True)),
            allow_fallback=bool(embedding.get("allow_fallback", retrieval.get("allow_model_fallback", False))),
            device=embedding.get("device"),
            dtype=str(embedding.get("dtype", "auto")),
            trust_remote_code=bool(embedding.get("trust_remote_code", True)),
            local_files_only=bool(embedding.get("local_files_only", False)),
            cache_dir=embedding.get("cache_dir"),
            max_length=int(embedding.get("max_length", 8192)),
            batch_size=int(embedding.get("batch_size", 8)),
        )

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        return self._build().encode(texts)

    def encode_queries(self, texts: Iterable[str]) -> list[list[float]]:
        encoder = self._build()
        if hasattr(encoder, "encode_queries"):
            return encoder.encode_queries(texts)
        return encoder.encode(texts)

    def encode_documents(self, texts: Iterable[str]) -> list[list[float]]:
        encoder = self._build()
        if hasattr(encoder, "encode_documents"):
            return encoder.encode_documents(texts)
        return encoder.encode(texts)
