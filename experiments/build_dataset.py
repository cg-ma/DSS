"""Build the P2 benchmark dataset and split artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dss_guard.data import (  # noqa: E402
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_STATS_PATH,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
    build_dataset_artifacts,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build P2 JSONL benchmark and train/valid/test splits")
    parser.add_argument("--benchmark-output", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--train-output", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(DEFAULT_VALID_PATH))
    parser.add_argument("--test-output", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--stats-output", default=str(DEFAULT_STATS_PATH))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = build_dataset_artifacts(
        benchmark_path=args.benchmark_output,
        train_path=args.train_output,
        valid_path=args.valid_output,
        test_path=args.test_output,
        stats_path=args.stats_output,
    )
    print(
        "P2 dataset built: "
        f"{result['case_count']} cases, "
        f"train={result['train_count']}, "
        f"valid={result['valid_count']}, "
        f"test={result['test_count']}"
    )


if __name__ == "__main__":
    main()
