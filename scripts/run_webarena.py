from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmt.core.memory_tree import HMTConfig
from hmt.datasets.webarena import WebArenaOnlineRunner, load_webarena_config
from hmt.models.openai_client import OpenAIChatClient
from hmt.utils.io import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HMT on WebArena with online memory.")
    parser.add_argument("--config", default="configs/real_webarena.yaml")
    parser.add_argument("--domain", required=True, choices=["shopping", "shopping_admin", "reddit", "gitlab", "maps", "map", "cms"])
    parser.add_argument("--task-order")
    parser.add_argument("--task-config-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--memory")
    parser.add_argument("--backend", choices=["browsergym", "native"])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--no-llm-construction", action="store_true")
    parser.add_argument("--llm-inference", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config)
    run_config = load_webarena_config(config, args.domain)
    if args.task_order:
        run_config.task_order_path = Path(args.task_order)
    if args.task_config_dir:
        run_config.task_config_dir = Path(args.task_config_dir)
    if args.output_dir:
        run_config.output_dir = Path(args.output_dir)
    if args.memory:
        run_config.memory_path = Path(args.memory)
    if args.backend:
        run_config.env_backend = args.backend
    if args.max_tasks is not None:
        run_config.max_tasks = args.max_tasks
    if args.max_steps is not None:
        run_config.max_steps = args.max_steps
    if args.no_llm_construction:
        run_config.use_llm_construction = False
    if args.llm_inference:
        run_config.use_llm_inference = True

    inference_cfg = config.get("inference", {})
    needs_openai = (
        run_config.use_llm_construction
        or run_config.use_llm_inference
        or bool(inference_cfg.get("use_llm_normalizer", True))
        or bool(inference_cfg.get("use_base_policy", True))
    )
    call_log = run_config.output_dir / "openai_calls.jsonl"
    llm_client = OpenAIChatClient.from_config(config, call_log_path=call_log) if needs_openai else None
    runner = WebArenaOnlineRunner(config=config, run_config=run_config, llm_client=llm_client)
    summary = runner.run()
    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_config.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
