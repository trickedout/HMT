from __future__ import annotations

import argparse
from pathlib import Path

from hmt.core.memory_maintenance import MaintenanceConfig, MemoryInspector, MemoryMaintainer, save_memory_report
from hmt.core.memory_tree import MemoryTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or lightly clean an HMT memory JSONL file.")
    parser.add_argument("--memory", required=True, help="Input T_mem JSONL file.")
    parser.add_argument("--output-dir", default="outputs/memory_inspection", help="Directory for stats and issue files.")
    parser.add_argument("--clean-output", default=None, help="Optional output JSONL path for a cleaned memory file.")
    parser.add_argument("--merge-duplicates", action="store_true", help="Merge near-duplicate nodes using semantic thresholds.")
    parser.add_argument("--keep-source-debug", action="store_true", help="Keep source_debug fields in action records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tree = MemoryTree.load_jsonl(args.memory)
    cfg = MaintenanceConfig(merge_duplicates=args.merge_duplicates, keep_source_debug=args.keep_source_debug)
    if args.clean_output:
        cleaned, report = MemoryMaintainer(tree, cfg).run()
        cleaned.save_jsonl(args.clean_output)
        save_memory_report(cleaned, args.output_dir, cfg)
    else:
        report = MemoryInspector(tree).inspect(cfg)
        save_memory_report(tree, args.output_dir, cfg)
    issues = len(report.issues)
    stats = report.stats_after or report.stats_before
    print(f"memory: intents={stats.get('num_intents')} stages={stats.get('num_stages')} actions={stats.get('num_actions')} issues={issues}")


if __name__ == "__main__":
    main()
