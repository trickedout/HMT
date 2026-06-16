from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from hmt.core.condition_match import tokenize


def _vectorize(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class LocalTextIndex:
    vectors: dict[str, Counter[str]] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)

    def add(self, item_id: str, text: str) -> None:
        self.texts[item_id] = text
        self.vectors[item_id] = _vectorize(text)

    def add_many(self, items: list[tuple[str, str]]) -> None:
        for item_id, text in items:
            self.add(item_id, text)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_vector = _vectorize(query)
        scored = [(item_id, _cosine(query_vector, vector)) for item_id, vector in self.vectors.items()]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
