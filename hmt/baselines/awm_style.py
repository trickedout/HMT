from __future__ import annotations

from dataclasses import dataclass

from hmt.adapters.awm_adapter import AWMWorkflowUnit, memory_tree_to_awm_units
from hmt.core.memory_tree import HMTConfig, MemoryTree
from hmt.core.retrieval import _top_k
from hmt.models.embeddings import EmbeddingModel
from hmt.models.local_index import LocalTextIndex
from hmt.models.reranker import CrossEncoderReranker


@dataclass
class AWMStyleHit:
    workflow: AWMWorkflowUnit
    score: float


class AWMStyleMemory:
    """Linear workflow baseline without HMT stage verification."""

    def __init__(
        self,
        tree: MemoryTree,
        config: HMTConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.workflows = memory_tree_to_awm_units(tree)
        self.config = config or HMTConfig()
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.index = LocalTextIndex()
        for idx, workflow in enumerate(self.workflows):
            self.index.add(str(idx), workflow.to_text())

    def search(self, query: str, top_k: int = 5) -> list[AWMStyleHit]:
        if self.config.retrieval_backend == "lexical_fallback":
            return [AWMStyleHit(self.workflows[int(idx)], score) for idx, score in self.index.search(query, top_k)]
        hits = _top_k(
            self.workflows,
            query,
            lambda workflow: workflow.to_text(),
            top_k,
            config=self.config,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
        )
        return [AWMStyleHit(hit.node, hit.score) for hit in hits]
