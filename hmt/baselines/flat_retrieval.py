from __future__ import annotations

from dataclasses import dataclass

from hmt.core.memory_tree import HMTConfig, MemoryTree
from hmt.core.retrieval import ScoredNode, _top_k
from hmt.models.embeddings import EmbeddingModel
from hmt.models.local_index import LocalTextIndex
from hmt.models.reranker import CrossEncoderReranker


@dataclass
class FlatRetrievalHit:
    record: dict
    score: float


class FlatRetrievalMemory:
    def __init__(
        self,
        tree: MemoryTree,
        config: HMTConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.records = tree.to_records()
        self.config = config or HMTConfig()
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.index = LocalTextIndex()
        for idx, record in enumerate(self.records):
            self.index.add(str(idx), _record_text(record))

    def search(self, query: str, top_k: int = 5) -> list[FlatRetrievalHit]:
        if self.config.retrieval_backend == "lexical_fallback":
            return [FlatRetrievalHit(self.records[int(idx)], score) for idx, score in self.index.search(query, top_k)]
        hits = _top_k(
            self.records,
            query,
            _record_text,
            top_k,
            config=self.config,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
        )
        return [FlatRetrievalHit(hit.node, hit.score) for hit in hits]


def _record_text(record: dict) -> str:
    if record.get("record_type") == "intent":
        return f"{record.get('canonical_intent', '')} {record.get('domain_hint', '')}"
    if record.get("record_type") == "stage":
        return " ".join([record.get("name", "")] + record.get("pre_conditions", []) + record.get("post_conditions", []))
    if record.get("record_type") == "action":
        desc = record.get("semantic_description", {})
        return " ".join(str(v) for v in desc.values())
    return str(record)
