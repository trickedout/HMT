from __future__ import annotations

# Backward-compatible import surface.  The real benchmark implementation lives
# in hmt.datasets.webarena.
from hmt.datasets.webarena import (  # noqa: F401
    VALID_DOMAINS,
    BrowserGymWebArenaSession,
    BrowserObservation,
    NativeWebArenaSession,
    WebArenaAction,
    WebArenaActionSerializer,
    WebArenaOnlineRunner,
    WebArenaRunConfig,
    WebArenaTask,
    WebArenaTaskOrder,
    WebArenaTaskResult,
    WebArenaStepLog,
    check_webarena_ready,
    load_webarena_config,
    make_session,
    normalize_domain,
    observation_from_env,
    run_domain_reset_stream,
    task_record_to_episode,
)
