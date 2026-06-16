from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmt.datasets.mind2web import Mind2WebRunner, check_mind2web_ready, load_mind2web_config
from hmt.models.openai_client import OpenAIChatClient
from hmt.utils.io import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HMT on Mind2Web files.")
    parser.add_argument("--config", default="configs/real_mind2web.yaml")
    parser.add_argument("--split", required=True, choices=["cross_task", "cross_website", "cross_domain", "train", "dev", "validation", "test"])
    parser.add_argument("--mode", choices=["build-memory", "evaluate", "build-and-evaluate"], default="evaluate")
    parser.add_argument("--data-path")
    parser.add_argument("--train-path")
    parser.add_argument("--memory")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--no-llm-construction", action="store_true")
    parser.add_argument("--llm-inference", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config)
    run_config = load_mind2web_config(config, args.split)
    if args.data_path:
        run_config.data_path = Path(args.data_path)
    if args.train_path:
        run_config.train_path = Path(args.train_path)
    if args.memory:
        run_config.memory_path = Path(args.memory)
    if args.output_dir:
        run_config.output_dir = Path(args.output_dir)
    if args.max_tasks is not None:
        run_config.max_tasks = args.max_tasks
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
    runner = Mind2WebRunner(config=config, run_config=run_config, llm_client=llm_client)

    if args.mode == "build-memory":
        target = runner.build_memory_from_train(output_path=run_config.memory_path)
        summary = {"memory_path": str(target)}
    elif args.mode == "build-and-evaluate":
        target = runner.build_memory_from_train(output_path=run_config.memory_path)
        summary = runner.evaluate(memory_path=target)
        summary["memory_path"] = str(target)
    else:
        check_mind2web_ready(run_config)
        summary = runner.evaluate(memory_path=run_config.memory_path)

    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_config.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
