from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmt.core.metrics import (
    check_ablation_table,
    check_mind2web_main_table,
    check_webarena_summary_table,
    recompute_mind2web_counts,
    recompute_webarena_from_task_logs,
    recompute_webarena_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute HMT metrics from saved counts/results files.")
    parser.add_argument("--webarena", help="CSV with method,domain,task_sr columns.")
    parser.add_argument("--raw-log", help="Per-task JSONL with task_id,domain,method,success,num_steps.")
    parser.add_argument("--ablation", help="Ablation CSV to validate.")
    parser.add_argument("--mind2web-main", help="Mind2Web main-table CSV to validate.")
    parser.add_argument("--webarena-summary", help="WebArena summary-table CSV to validate.")
    parser.add_argument("--counts", help="Counts JSON for WebArena or Mind2Web.")
    parser.add_argument("--mind2web", action="store_true", help="Treat --counts as Mind2Web split counts.")
    args = parser.parse_args()

    if args.raw_log:
        if not args.counts:
            raise SystemExit("--counts is required with --raw-log")
        output = recompute_webarena_from_task_logs(args.raw_log, args.counts)
    elif args.mind2web:
        if not args.counts:
            raise SystemExit("--counts is required with --mind2web")
        output = recompute_mind2web_counts(args.counts)
    elif args.ablation:
        output = check_ablation_table(args.ablation)
    elif args.mind2web_main:
        output = check_mind2web_main_table(args.mind2web_main)
    elif args.webarena_summary:
        output = check_webarena_summary_table(args.webarena_summary)
    else:
        if not args.webarena or not args.counts:
            raise SystemExit("--webarena is required unless --mind2web is set")
        output = recompute_webarena_table(args.webarena, args.counts)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
