from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import hashlib
import json
import math
import time

from hmt.core.condition_match import match_stage_conditions, tokenize
from hmt.core.memory_tree import ActionNode, HMTConfig, IntentNode, MemoryTree, StageNode
from hmt.models.embeddings import EmbeddingModel
from hmt.models.local_index import LocalTextIndex
from hmt.models.reranker import CrossEncoderReranker
from hmt.utils.io import read_json, write_json

Node = IntentNode | StageNode | ActionNode


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    n = min(len(left), len(right))
    dot = sum(float(left[i]) * float(right[i]) for i in range(n))
    ln = math.sqrt(sum(float(x) * float(x) for x in left[:n]))
    rn = math.sqrt(sum(float(x) * float(x) for x in right[:n]))
    if ln == 0 or rn == 0:
        return 0.0
    return dot / (ln * rn)


@dataclass
class IndexedNode:
    node_id: str
    node_type: str
    parent_id: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None

    def fingerprint(self) -> str:
        return _hash_text(json.dumps({"id": self.node_id, "type": self.node_type, "parent": self.parent_id, "text": self.text}, sort_keys=True, ensure_ascii=False))

    def to_manifest(self, include_vector: bool = False) -> dict[str, Any]:
        data = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "parent_id": self.parent_id,
            "text_sha1": _hash_text(self.text),
            "text_chars": len(self.text),
            "metadata": self.metadata,
        }
        if include_vector and self.vector is not None:
            data["vector_dim"] = len(self.vector)
            data["vector_checksum"] = _hash_text(json.dumps(self.vector[:16]))
        return data


@dataclass
class IndexHit:
    node_id: str
    node_type: str
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    condition_score: float = 0.0
    parent_id: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "parent_id": self.parent_id,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
            "rerank_score": self.rerank_score,
            "condition_score": self.condition_score,
            "reason": self.reason,
        }


@dataclass
class QueryTrace:
    query: str
    node_type: str
    top_k: int
    expanded_k: int
    backend: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    hits: list[IndexHit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, hits: list[IndexHit]) -> list[IndexHit]:
        self.finished_at = time.time()
        self.hits = hits
        return hits

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "node_type": self.node_type,
            "top_k": self.top_k,
            "expanded_k": self.expanded_k,
            "backend": self.backend,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": None if self.finished_at is None else round((self.finished_at - self.started_at) * 1000, 2),
            "hits": [hit.to_dict() for hit in self.hits],
            "metadata": self.metadata,
        }


class NodeTextBuilder:
    def intent_text(self, node: IntentNode) -> str:
        constraints = " ".join(f"{c.slot}: {c.value}; evidence: {c.evidence}" for c in node.constraints)
        return " ".join(x for x in [node.canonical_intent, constraints, node.domain_hint] if x)

    def stage_text(self, node: StageNode) -> str:
        parts = [
            node.name,
            "pre: " + " ; ".join(node.pre_conditions),
            "post: " + " ; ".join(node.post_conditions),
            _flatten(node.extra),
        ]
        return " ".join(x for x in parts if x)

    def action_text(self, node: ActionNode) -> str:
        descriptor = node.semantic_description.to_dict()
        fields = [
            node.operation,
            node.argument_template or "",
            descriptor.get("element_purpose", ""),
            descriptor.get("role", ""),
            descriptor.get("label_or_text", ""),
            descriptor.get("accessible_name", ""),
            descriptor.get("parent_context", ""),
            descriptor.get("form_section", ""),
            descriptor.get("field_slot", ""),
            _flatten(descriptor.get("disambiguators", [])),
            _flatten(descriptor.get("negative_constraints", [])),
        ]
        return " ".join(str(x) for x in fields if x)

    def build(self, node: Node) -> str:
        if isinstance(node, IntentNode):
            return self.intent_text(node)
        if isinstance(node, StageNode):
            return self.stage_text(node)
        if isinstance(node, ActionNode):
            return self.action_text(node)
        return _flatten(node)


class HierarchicalIndex:
    def __init__(
        self,
        tree: MemoryTree,
        config: HMTConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: CrossEncoderReranker | None = None,
        text_builder: NodeTextBuilder | None = None,
    ) -> None:
        self.tree = tree
        self.config = config or HMTConfig()
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.text_builder = text_builder or NodeTextBuilder()
        self.nodes: dict[str, IndexedNode] = {}
        self.by_type: dict[str, list[str]] = {"intent": [], "stage": [], "action": []}
        self.children: dict[str, list[str]] = {}
        self.lexical_indices: dict[str, LocalTextIndex] = {}
        self.query_traces: list[QueryTrace] = []
        self._built = False

    def build(self, include_vectors: bool = True) -> None:
        self.nodes.clear()
        self.by_type = {"intent": [], "stage": [], "action": []}
        self.children.clear()
        for intent in self.tree.intents.values():
            self._add_indexed_node("intent", intent.intent_id, None, self.text_builder.intent_text(intent), {"domain_hint": intent.domain_hint})
        for stage in self.tree.stages.values():
            self._add_indexed_node("stage", stage.stage_id, stage.parent_intent_id, self.text_builder.stage_text(stage), {"span": stage.span.to_list(), "name": stage.name})
        for action in self.tree.actions.values():
            self._add_indexed_node("action", action.action_id, action.parent_stage_id, self.text_builder.action_text(action), {"operation": action.operation})
        self._build_lexical_indices()
        if include_vectors and self.config.retrieval_backend != "lexical_fallback":
            self._build_vectors()
        self._built = True

    def _add_indexed_node(self, node_type: str, node_id: str, parent_id: str | None, text: str, metadata: dict[str, Any]) -> None:
        indexed = IndexedNode(node_id=node_id, node_type=node_type, parent_id=parent_id, text=text, metadata=metadata)
        self.nodes[node_id] = indexed
        self.by_type.setdefault(node_type, []).append(node_id)
        if parent_id is not None:
            self.children.setdefault(parent_id, []).append(node_id)

    def _build_lexical_indices(self) -> None:
        self.lexical_indices = {}
        for node_type, ids in self.by_type.items():
            index = LocalTextIndex()
            index.add_many([(node_id, self.nodes[node_id].text) for node_id in ids])
            self.lexical_indices[node_type] = index

    def _build_vectors(self) -> None:
        if not self.nodes:
            return
        embedder = self.embedding_model or EmbeddingModel.from_config({
            "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B", "normalize": True, "allow_fallback": self.config.allow_model_fallback}
        })
        for node_type, ids in self.by_type.items():
            if not ids:
                continue
            texts = [self.nodes[node_id].text for node_id in ids]
            if hasattr(embedder, "encode_documents"):
                vectors = embedder.encode_documents(texts)
            else:
                vectors = embedder.encode(texts)
            for node_id, vector in zip(ids, vectors):
                self.nodes[node_id].vector = [float(x) for x in vector]

    def ensure_built(self) -> None:
        if not self._built:
            self.build(include_vectors=self.config.retrieval_backend != "lexical_fallback")

    def query_intents(self, instruction: str, top_k: int | None = None) -> list[IndexHit]:
        return self.query("intent", instruction, top_k=top_k or self.config.task_top_k)

    def query_stages(self, query: str, intent_ids: Iterable[str], observation_summary: str = "", top_k: int | None = None) -> list[IndexHit]:
        allowed = set()
        for intent_id in intent_ids:
            allowed.update(self.children.get(intent_id, []))
        hits = self.query("stage", query, top_k=max(top_k or self.config.stage_top_k, len(allowed) or 1), allowed_ids=allowed or None)
        if observation_summary:
            hits = self.apply_condition_scores(hits, observation_summary)
        hits.sort(key=lambda hit: (-hit.score, hit.node_id))
        return hits[: (top_k or self.config.stage_top_k)]

    def query_actions(self, query: str, stage_id: str, top_k: int | None = None) -> list[IndexHit]:
        allowed = set(self.children.get(stage_id, []))
        return self.query("action", query, top_k=top_k or self.config.action_top_k, allowed_ids=allowed)

    def query_all_actions(self, query: str, top_k: int | None = None) -> list[IndexHit]:
        return self.query("action", query, top_k=top_k or self.config.action_top_k)

    def query(self, node_type: str, query: str, top_k: int, allowed_ids: set[str] | None = None, expanded_k: int | None = None) -> list[IndexHit]:
        self.ensure_built()
        expanded = expanded_k or max(top_k, top_k * 4)
        trace = QueryTrace(query=query, node_type=node_type, top_k=top_k, expanded_k=expanded, backend=self.config.retrieval_backend)
        candidate_ids = [node_id for node_id in self.by_type.get(node_type, []) if allowed_ids is None or node_id in allowed_ids]
        if not candidate_ids:
            return trace.finish([])
        if self.config.retrieval_backend == "lexical_fallback" or not any(self.nodes[i].vector for i in candidate_ids):
            hits = self._lexical_query(node_type, query, candidate_ids, expanded)
        else:
            hits = self._vector_query(node_type, query, candidate_ids, expanded)
        hits = self._rerank(query, hits, top_k)
        self.query_traces.append(trace)
        return trace.finish(hits)

    def _lexical_query(self, node_type: str, query: str, candidate_ids: list[str], top_k: int) -> list[IndexHit]:
        index = LocalTextIndex()
        index.add_many([(node_id, self.nodes[node_id].text) for node_id in candidate_ids])
        raw_hits = index.search(query, top_k=top_k)
        hits: list[IndexHit] = []
        for node_id, score in raw_hits:
            node = self.nodes[node_id]
            hits.append(IndexHit(node_id=node_id, node_type=node_type, parent_id=node.parent_id, score=score, lexical_score=score, reason="lexical_index"))
        return hits

    def _vector_query(self, node_type: str, query: str, candidate_ids: list[str], top_k: int) -> list[IndexHit]:
        embedder = self.embedding_model or EmbeddingModel.from_config({
            "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B", "normalize": True, "allow_fallback": self.config.allow_model_fallback}
        })
        if hasattr(embedder, "encode_queries"):
            qvec = embedder.encode_queries([query])[0]
        else:
            qvec = embedder.encode([query])[0]
        scored = []
        for node_id in candidate_ids:
            node = self.nodes[node_id]
            semantic = _cosine(qvec, node.vector or [])
            lexical = self._lexical_overlap_score(query, node.text)
            score = 0.82 * semantic + 0.18 * lexical
            scored.append(IndexHit(node_id=node_id, node_type=node_type, parent_id=node.parent_id, score=score, semantic_score=semantic, lexical_score=lexical, reason="qwen_embedding"))
        scored.sort(key=lambda hit: (-hit.score, hit.node_id))
        return scored[:top_k]

    def _rerank(self, query: str, hits: list[IndexHit], top_k: int) -> list[IndexHit]:
        if not hits:
            return []
        ranker = self.reranker
        if ranker is None and self.config.retrieval_backend != "lexical_fallback":
            try:
                ranker = CrossEncoderReranker.from_config({
                    "reranker": {"model": "Qwen/Qwen3-Reranker-0.6B", "allow_fallback": self.config.allow_model_fallback}
                })
            except Exception:
                ranker = None
        if ranker is None:
            hits.sort(key=lambda hit: (-hit.score, hit.node_id))
            return hits[:top_k]
        pairs = [(hit.node_id, self.nodes[hit.node_id].text) for hit in hits]
        reranked = ranker.rerank(query, pairs, top_k=top_k)
        score_by_id = {hit.node_id: hit for hit in hits}
        result: list[IndexHit] = []
        for node_id, rerank_score in reranked:
            hit = score_by_id[node_id]
            hit.rerank_score = float(rerank_score)
            if hit.rerank_score != 0.0:
                hit.score = 0.65 * hit.rerank_score + 0.35 * hit.score
                hit.reason = hit.reason + "+qwen_reranker"
            result.append(hit)
        result.sort(key=lambda hit: (-hit.score, hit.node_id))
        return result[:top_k]

    def apply_condition_scores(self, stage_hits: list[IndexHit], observation_summary: str) -> list[IndexHit]:
        result: list[IndexHit] = []
        for hit in stage_hits:
            stage = self.tree.stages.get(hit.node_id)
            if stage is None:
                continue
            cond = match_stage_conditions(stage, observation_summary, self.config.theta_pre, self.config.theta_post_done, self.config.theta_conflict)
            hit.condition_score = cond.score
            if cond.has_conflict:
                hit.score = -1.0
                hit.reason += "+condition_conflict"
            elif cond.already_completed:
                hit.score *= 0.1
                hit.reason += "+already_completed"
            else:
                hit.score = (1.0 - self.config.condition_weight_lambda) * hit.score + self.config.condition_weight_lambda * cond.score
                hit.reason += f"+condition_{cond.decision}"
            result.append(hit)
        return result

    def _lexical_overlap_score(self, query: str, text: str) -> float:
        a = tokenize(query)
        b = tokenize(text)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def node_object(self, hit: IndexHit) -> Node | None:
        if hit.node_type == "intent":
            return self.tree.intents.get(hit.node_id)
        if hit.node_type == "stage":
            return self.tree.stages.get(hit.node_id)
        if hit.node_type == "action":
            return self.tree.actions.get(hit.node_id)
        return None

    def manifest(self) -> dict[str, Any]:
        self.ensure_built()
        return {
            "created_at": time.time(),
            "backend": self.config.retrieval_backend,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
            "node_counts": {node_type: len(ids) for node_type, ids in self.by_type.items()},
            "nodes": [self.nodes[node_id].to_manifest(include_vector=True) for node_id in sorted(self.nodes)],
        }

    def save_manifest(self, path: str | Path) -> None:
        write_json(path, self.manifest())

    def save_query_traces(self, path: str | Path) -> None:
        write_json(path, {"queries": [trace.to_dict() for trace in self.query_traces]})


def index_fingerprint(tree: MemoryTree) -> str:
    builder = NodeTextBuilder()
    payload = []
    for intent in sorted(tree.intents.values(), key=lambda x: x.intent_id):
        payload.append(("intent", intent.intent_id, builder.intent_text(intent)))
    for stage in sorted(tree.stages.values(), key=lambda x: x.stage_id):
        payload.append(("stage", stage.stage_id, stage.parent_intent_id, builder.stage_text(stage)))
    for action in sorted(tree.actions.values(), key=lambda x: x.action_id):
        payload.append(("action", action.action_id, action.parent_stage_id, builder.action_text(action)))
    return _hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
