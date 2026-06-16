from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Iterable

from hmt.core.condition_match import match_stage_conditions
from hmt.core.memory_tree import ActionNode, HMTConfig, IntentNode, MemoryTree, StageNode
from hmt.models.embeddings import EmbeddingModel
from hmt.models.local_index import LocalTextIndex
from hmt.models.reranker import CrossEncoderReranker
from hmt.utils.io import write_json


@dataclass
class ScoredNode:
    node: Any
    score: float
    reason: str = ""


def _intent_text(node: IntentNode) -> str:
    constraints = " ".join(f"{c.slot} {c.value}" for c in node.constraints)
    return f"{node.canonical_intent} {constraints} {node.domain_hint}".strip()


def _stage_text(node: StageNode) -> str:
    def flatten(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            return " ".join(flatten(v) for v in value)
        if isinstance(value, dict):
            return " ".join(flatten(v) for v in value.values())
        return str(value)

    conditions = " ".join(node.pre_conditions + node.post_conditions)
    extra = " ".join(flatten(v) for v in getattr(node, "extra", {}).values() if v)
    return f"{node.name} {conditions} {extra}".strip()


def _action_text(node: ActionNode) -> str:
    return f"{node.operation} {node.argument_template or ''} {node.semantic_description.text()}".strip()


def _lexical_top_k(nodes: Iterable[Any], query: str, text_fn: Callable[[Any], str], k: int) -> list[ScoredNode]:
    index = LocalTextIndex()
    node_list = list(nodes)
    index.add_many([(str(i), text_fn(node)) for i, node in enumerate(node_list)])
    scored = []
    for item_id, score in index.search(query, top_k=k):
        scored.append(ScoredNode(node=node_list[int(item_id)], score=score, reason="lexical_fallback"))
    return scored


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _qwen_top_k(
    nodes: Iterable[Any],
    query: str,
    text_fn: Callable[[Any], str],
    k: int,
    expanded_k: int | None = None,
    embedding_model: EmbeddingModel | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[ScoredNode]:
    node_list = list(nodes)
    if not node_list:
        return []
    expanded = max(k, expanded_k or min(len(node_list), k * 4))
    texts = [text_fn(node) for node in node_list]
    embedder = embedding_model or EmbeddingModel(
        model="Qwen/Qwen3-Embedding-0.6B",
        normalize=True,
        allow_fallback=False,
    )
    if hasattr(embedder, "encode_queries"):
        query_vector = embedder.encode_queries([query])[0]
    else:
        query_vector = embedder.encode([query])[0]
    if hasattr(embedder, "encode_documents"):
        node_vectors = embedder.encode_documents(texts)
    else:
        node_vectors = embedder.encode(texts)
    vector_hits = sorted(
        [(index, _cosine(query_vector, vector)) for index, vector in enumerate(node_vectors)],
        key=lambda item: (-item[1], str(item[0])),
    )[:expanded]
    ranker = reranker or CrossEncoderReranker(model="Qwen/Qwen3-Reranker-0.6B", allow_fallback=False)
    reranked = ranker.rerank(query, [(str(index), texts[index]) for index, _ in vector_hits], top_k=k)
    vector_score_by_id = {str(index): score for index, score in vector_hits}
    reason = (
        "qwen_transformers_embedding_reranker"
        if "qwen" in getattr(embedder, "backend", "qwen_transformers") and "qwen" in getattr(ranker, "backend", "qwen_transformers")
        else "deterministic_fallback"
    )
    hits: list[ScoredNode] = []
    for item_id, rerank_score in reranked:
        index = int(item_id)
        score = float(rerank_score)
        if score == 0.0:
            score = vector_score_by_id.get(item_id, 0.0)
        hits.append(ScoredNode(node=node_list[index], score=score, reason=reason))
    return hits


def _top_k(
    nodes: Iterable[Any],
    query: str,
    text_fn: Callable[[Any], str],
    k: int,
    config: HMTConfig | None = None,
    embedding_model: EmbeddingModel | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[ScoredNode]:
    if config and config.retrieval_backend == "lexical_fallback":
        return _lexical_top_k(nodes, query, text_fn, k)
    if config and config.allow_model_fallback and embedding_model is None:
        embedding_model = EmbeddingModel(model="Qwen/Qwen3-Embedding-0.6B", normalize=True, allow_fallback=True)
    if config and config.allow_model_fallback and reranker is None:
        reranker = CrossEncoderReranker(model="Qwen/Qwen3-Reranker-0.6B", allow_fallback=True)
    try:
        return _qwen_top_k(nodes, query, text_fn, k, embedding_model=embedding_model, reranker=reranker)
    except Exception:
        if config and config.allow_model_fallback:
            return [ScoredNode(hit.node, hit.score, "deterministic_fallback") for hit in _lexical_top_k(nodes, query, text_fn, k)]
        raise


def retrieve_intents(
    tree: MemoryTree,
    instruction: str,
    top_k: int = 5,
    config: HMTConfig | None = None,
    embedding_model: EmbeddingModel | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[ScoredNode]:
    return _top_k(tree.intents.values(), instruction, _intent_text, top_k, config, embedding_model, reranker)


def retrieve_stages(
    tree: MemoryTree,
    intent_hits: list[ScoredNode],
    query: str,
    observation_summary: str,
    config: HMTConfig,
    embedding_model: EmbeddingModel | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[ScoredNode]:
    candidate_stages = tree.stages_for_intents([hit.node.intent_id for hit in intent_hits])
    semantic_hits = _top_k(
        candidate_stages,
        query,
        _stage_text,
        max(config.stage_top_k, len(candidate_stages)),
        config,
        embedding_model,
        reranker,
    )
    ranked: list[ScoredNode] = []
    for hit in semantic_hits:
        if not config.use_pre_conditions and not config.use_post_conditions:
            combined = hit.score
            reason = "semantic_only_no_prepost_ablation"
        else:
            stage_for_match = hit.node
            if not config.use_pre_conditions or not config.use_post_conditions:
                stage_for_match = replace(
                    hit.node,
                    pre_conditions=hit.node.pre_conditions if config.use_pre_conditions else [],
                    post_conditions=hit.node.post_conditions if config.use_post_conditions else [],
                )
            cond = match_stage_conditions(
                stage_for_match,
                observation_summary,
                theta_pre=config.theta_pre,
                theta_post_done=config.theta_post_done,
                theta_conflict=config.theta_conflict,
            )
            if cond.has_conflict:
                combined = -1.0
            elif cond.already_completed:
                combined = 0.1 * hit.score
            else:
                combined = (1 - config.condition_weight_lambda) * hit.score + config.condition_weight_lambda * cond.score
            reason = cond.decision
        ranked.append(ScoredNode(node=hit.node, score=combined, reason=reason))
    ranked.sort(key=lambda x: (-x.score, x.node.stage_id))
    return ranked[: config.stage_top_k]


def retrieve_actions(
    tree: MemoryTree,
    stage_id: str,
    query: str,
    top_k: int = 5,
    config: HMTConfig | None = None,
    embedding_model: EmbeddingModel | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[ScoredNode]:
    return _top_k(tree.actions_for_stage(stage_id), query, _action_text, top_k, config, embedding_model, reranker)




def retrieve_all_actions(
    tree: MemoryTree,
    query: str,
    top_k: int = 5,
    config: HMTConfig | None = None,
    embedding_model: EmbeddingModel | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> list[ScoredNode]:
    """Retrieve actions globally for the flat/no-stage ablation."""
    return _top_k(tree.actions.values(), query, _action_text, top_k, config, embedding_model, reranker)


def write_index_manifest(tree: MemoryTree, path: str, config: HMTConfig | None = None) -> None:
    cfg = config or HMTConfig()
    write_json(
        path,
        {
            "format_version": "1.0",
            "retrieval_backend": cfg.retrieval_backend,
            "allow_model_fallback": cfg.allow_model_fallback,
            "index_persistence": "manifest_only_rebuild_vectors_from_memory_records",
            "embedding": {
                "provider": "transformers",
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "normalize": True,
            },
            "reranker": {
                "provider": "transformers_cross_encoder",
                "model": "Qwen/Qwen3-Reranker-0.6B",
            },
            "node_counts": {
                "intent": len(tree.intents),
                "stage": len(tree.stages),
                "action": len(tree.actions),
            },
        },
    )
