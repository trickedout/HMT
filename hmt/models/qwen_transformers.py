from __future__ import annotations

"""Local Qwen model adapters used by HMT retrieval.

This module intentionally uses Hugging Face ``transformers`` directly rather
than remote APIs.  The default HMT configuration uses GPT-4 only for the structured abstraction / Planner / Actor JSON calls; all embedding and reranking models are
loaded locally from the configured model names, e.g. ``Qwen/Qwen3-Embedding-0.6B``
and ``Qwen/Qwen3-Reranker-0.6B``.

The classes are written to be conservative about dependencies: importing this
module does not import torch or transformers.  The heavy libraries are imported
only when a model is instantiated.  This makes command-line help and packaging
work on machines where the local model stack is not yet installed, while real
benchmark execution still fails loudly if the Qwen dependencies or weights are
missing.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence
import math
import os


PoolingStrategy = Literal["mean", "last_token", "cls"]


def _require_transformers() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Local Qwen inference requires `torch` and `transformers`. Install the "
            "model extras and make sure the Qwen weights are available locally or "
            "downloadable from Hugging Face."
        ) from exc
    return torch, AutoTokenizer, AutoModel, AutoModelForSequenceClassification


def _as_list(texts: Iterable[str]) -> list[str]:
    return ["" if text is None else str(text) for text in texts]


def _l2_normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if not norm:
        return [float(x) for x in vector]
    return [float(x) / norm for x in vector]


@dataclass
class QwenRuntimeConfig:
    model_name_or_path: str
    device: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    local_files_only: bool = False
    cache_dir: str | None = None
    max_length: int = 8192
    batch_size: int = 8

    @classmethod
    def from_mapping(cls, data: dict[str, Any], default_model: str) -> "QwenRuntimeConfig":
        return cls(
            model_name_or_path=str(data.get("model", data.get("model_name_or_path", default_model))),
            device=data.get("device"),
            dtype=str(data.get("dtype", "auto")),
            trust_remote_code=bool(data.get("trust_remote_code", True)),
            local_files_only=bool(data.get("local_files_only", False)),
            cache_dir=data.get("cache_dir") or os.environ.get("HF_HOME"),
            max_length=int(data.get("max_length", 8192)),
            batch_size=int(data.get("batch_size", 8)),
        )


def _resolve_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype!r}")


class _TorchModelMixin:
    config: QwenRuntimeConfig
    _torch: Any
    tokenizer: Any
    model: Any

    def _device(self) -> str:
        if self.config.device:
            return self.config.device
        if self._torch.cuda.is_available():
            return "cuda"
        if getattr(self._torch.backends, "mps", None) and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = self._device()
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


@dataclass
class QwenDenseEncoder(_TorchModelMixin):
    """Dense text encoder backed by a local Qwen embedding model.

    The encoder can be used as a drop-in replacement for the previous
    sentence-transformers wrapper.  It uses direct transformer forward passes
    and deterministic pooling.  ``query_instruction`` mirrors the common Qwen
    embedding convention of prefixing retrieval queries with a task statement;
    memory-node texts are encoded without this prefix.
    """

    config: QwenRuntimeConfig
    normalize: bool = True
    pooling: PoolingStrategy = "last_token"
    query_instruction: str = (
        "Given a web-agent instruction and current observation, retrieve HMT memory nodes "
        "that match the reusable intent, stage conditions, or semantic action pattern."
    )
    _torch: Any = field(init=False, repr=False)
    tokenizer: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)
    backend: str = field(default="qwen_transformers", init=False)

    def __post_init__(self) -> None:
        torch, AutoTokenizer, AutoModel, _ = _require_transformers()
        self._torch = torch
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        if self.config.cache_dir:
            model_kwargs["cache_dir"] = self.config.cache_dir
        dtype = _resolve_dtype(torch, self.config.dtype)
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        tokenizer_kwargs = {key: value for key, value in model_kwargs.items() if key != "torch_dtype"}
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, **tokenizer_kwargs)
        self.model = AutoModel.from_pretrained(self.config.model_name_or_path, **model_kwargs)
        self.model.to(self._device())
        self.model.eval()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QwenDenseEncoder":
        embedding_cfg = config.get("embedding", {})
        runtime = QwenRuntimeConfig.from_mapping(embedding_cfg, "Qwen/Qwen3-Embedding-0.6B")
        return cls(
            config=runtime,
            normalize=bool(embedding_cfg.get("normalize", True)),
            pooling=str(embedding_cfg.get("pooling", "last_token")),
            query_instruction=str(
                embedding_cfg.get(
                    "query_instruction",
                    "Given a web-agent instruction and current observation, retrieve HMT memory nodes "
                    "that match the reusable intent, stage conditions, or semantic action pattern.",
                )
            ),
        )

    def _format_query(self, text: str) -> str:
        if not self.query_instruction:
            return text
        return f"Instruct: {self.query_instruction}\nQuery: {text}"

    def encode_queries(self, texts: Iterable[str]) -> list[list[float]]:
        return self.encode([self._format_query(text) for text in texts])

    def encode_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return self.encode(texts)

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = _as_list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(text_list), self.config.batch_size):
            batch_texts = text_list[start : start + self.config.batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            batch = self._move_batch(batch)
            with self._torch.no_grad():
                outputs = self.model(**batch)
            pooled = self._pool(outputs, batch.get("attention_mask"))
            pooled = pooled.detach().cpu().float().tolist()
            if self.normalize:
                pooled = [_l2_normalize_vector(vector) for vector in pooled]
            vectors.extend(pooled)
        return vectors

    def _pool(self, outputs: Any, attention_mask: Any) -> Any:
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "mean":
            if attention_mask is None:
                return hidden.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            return summed / denom
        if self.pooling == "last_token":
            if attention_mask is None:
                return hidden[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = self._torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_indices, sequence_lengths]
        raise ValueError(f"Unsupported pooling strategy: {self.pooling!r}")


@dataclass
class QwenCrossEncoderReranker(_TorchModelMixin):
    """Pairwise reranker backed by a local Qwen reranker model."""

    config: QwenRuntimeConfig
    score_activation: Literal["identity", "sigmoid", "softmax_yes"] = "identity"
    _torch: Any = field(init=False, repr=False)
    tokenizer: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)
    backend: str = field(default="qwen_transformers", init=False)

    def __post_init__(self) -> None:
        torch, AutoTokenizer, _, AutoModelForSequenceClassification = _require_transformers()
        self._torch = torch
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        if self.config.cache_dir:
            model_kwargs["cache_dir"] = self.config.cache_dir
        dtype = _resolve_dtype(torch, self.config.dtype)
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        tokenizer_kwargs = {key: value for key, value in model_kwargs.items() if key != "torch_dtype"}
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, **tokenizer_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name_or_path, **model_kwargs)
        self.model.to(self._device())
        self.model.eval()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QwenCrossEncoderReranker":
        reranker_cfg = config.get("reranker", {})
        runtime = QwenRuntimeConfig.from_mapping(reranker_cfg, "Qwen/Qwen3-Reranker-0.6B")
        return cls(
            config=runtime,
            score_activation=str(reranker_cfg.get("score_activation", "identity")),
        )

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        if not candidates:
            return []
        scored: list[tuple[str, float]] = []
        for start in range(0, len(candidates), self.config.batch_size):
            batch_items = candidates[start : start + self.config.batch_size]
            queries = [query for _ in batch_items]
            docs = [text for _, text in batch_items]
            batch = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            batch = self._move_batch(batch)
            with self._torch.no_grad():
                outputs = self.model(**batch)
            batch_scores = self._extract_scores(outputs)
            for (item_id, _), score in zip(batch_items, batch_scores):
                scored.append((item_id, float(score)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    def _extract_scores(self, outputs: Any) -> list[float]:
        logits = outputs.logits
        if logits.ndim == 1:
            raw = logits
        elif logits.shape[-1] == 1:
            raw = logits[:, 0]
        else:
            if self.score_activation == "softmax_yes":
                probs = self._torch.softmax(logits, dim=-1)
                raw = probs[:, -1]
            else:
                raw = logits[:, -1]
        if self.score_activation == "sigmoid":
            raw = self._torch.sigmoid(raw)
        return raw.detach().cpu().float().tolist()


class DeterministicHashEncoder:
    """Small deterministic fallback used only when explicitly enabled."""

    backend = "deterministic_fallback"

    def __init__(self, dim: int = 256, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        from hmt.models.local_index import _vectorize

        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token, count in _vectorize(text).items():
                vector[hash(token) % self.dim] += float(count)
            if self.normalize:
                vector = _l2_normalize_vector(vector)
            vectors.append(vector)
        return vectors

    def encode_queries(self, texts: Iterable[str]) -> list[list[float]]:
        return self.encode(texts)

    def encode_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return self.encode(texts)


class DeterministicOverlapReranker:
    backend = "deterministic_fallback"

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        from hmt.core.condition_match import condition_overlap

        scored = [(item_id, condition_overlap(query, text)) for item_id, text in candidates]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]


def build_dense_encoder_from_config(config: dict[str, Any], allow_fallback: bool = False) -> Any:
    embedding_cfg = config.get("embedding", {})
    try:
        return QwenDenseEncoder.from_config(config)
    except Exception:
        if allow_fallback or bool(embedding_cfg.get("allow_fallback", False)):
            return DeterministicHashEncoder(normalize=bool(embedding_cfg.get("normalize", True)))
        raise


def build_reranker_from_config(config: dict[str, Any], allow_fallback: bool = False) -> Any:
    reranker_cfg = config.get("reranker", {})
    try:
        return QwenCrossEncoderReranker.from_config(config)
    except Exception:
        if allow_fallback or bool(reranker_cfg.get("allow_fallback", False)):
            return DeterministicOverlapReranker()
        raise
