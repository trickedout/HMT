from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmt.core.construction import build_memory
from hmt.core.memory_tree import Episode, HMTConfig
from hmt.core.retrieval import write_index_manifest
from hmt.core.structured_output import validate_memory_file
from hmt.models.openai_client import OpenAIChatClient
from hmt.utils.io import read_jsonl, read_yaml
from hmt.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HMT memory from successful episode JSONL.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, help="JSONL file with raw_instruction and trajectory fields.")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--call-log", default="outputs/construction_calls.jsonl")
    parser.add_argument("--index-manifest", default=None)
    parser.add_argument("--output", default="outputs/T_mem.jsonl")
    args = parser.parse_args()

    config = read_yaml(args.config)
    set_seed(int(config.get("random_seed", 3407)))
    episodes = [Episode.from_dict(record) for record in read_jsonl(args.input)]
    client = OpenAIChatClient.from_config(config, call_log_path=args.call_log) if args.use_llm else None
    tree = build_memory(episodes, HMTConfig.from_mapping(config), client=client)
    tree.save_jsonl(args.output)
    manifest = args.index_manifest or str(Path(args.output).with_name("index_manifest.json"))
    write_index_manifest(tree, manifest, HMTConfig.from_mapping(config))
    validate_memory_file(args.output)
    print(f"Wrote {len(tree.to_records())} memory records to {Path(args.output)}")


if __name__ == "__main__":
    main()
