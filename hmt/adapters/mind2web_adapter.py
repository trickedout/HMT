from __future__ import annotations

# Backward-compatible import surface.  The real benchmark implementation lives
# in hmt.datasets.mind2web.
from hmt.datasets.mind2web import (  # noqa: F401
    VALID_SPLITS,
    Mind2WebDataset,
    Mind2WebMetrics,
    Mind2WebRunConfig,
    Mind2WebRunner,
    Mind2WebStep,
    Mind2WebStepResult,
    Mind2WebTask,
    check_mind2web_ready,
    compute_mind2web_metrics,
    load_episodes_jsonl,
    load_mind2web_config,
    load_split_counts,
    record_to_episode,
    record_to_task,
)
