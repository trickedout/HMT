from __future__ import annotations

from typing import Any


def construction_call_count(num_steps: int, num_stages: int, repair_calls: int = 0) -> dict[str, int]:
    return {
        "normalize": 1,
        "segment": 1,
        "describe": num_stages,
        "abstract_step": num_steps,
        "consistency_repair": repair_calls,
        "total": 2 + num_stages + num_steps + repair_calls,
    }


def estimate_openai_cost_usd(
    token_usage: dict[str, Any] | None,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
) -> float | None:
    if token_usage is None or input_usd_per_million_tokens is None or output_usd_per_million_tokens is None:
        return None
    prompt_tokens = token_usage.get("prompt_tokens")
    completion_tokens = token_usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    return round(
        (float(prompt_tokens) / 1_000_000.0) * input_usd_per_million_tokens
        + (float(completion_tokens) / 1_000_000.0) * output_usd_per_million_tokens,
        8,
    )
