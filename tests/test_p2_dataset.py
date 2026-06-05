from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dss_guard.data.benchmark import (  # noqa: E402
    DATASET_COUNTS,
    build_benchmark_cases,
    build_dataset_artifacts,
    read_jsonl_cases,
    split_cases,
    validate_cases,
    validate_splits,
)


def test_p2_benchmark_counts_schema_and_hard_negatives() -> None:
    cases = build_benchmark_cases()

    assert len(cases) == 300
    assert Counter(case["attack_type"] for case in cases) == Counter(DATASET_COUNTS)
    assert validate_cases(cases) == []

    hard_negative_counts = Counter(
        case["hard_negative_for"]
        for case in cases
        if case["label"] == "clean" and case.get("hard_negative_for")
    )
    for attack_type in DATASET_COUNTS:
        if attack_type != "clean":
            assert hard_negative_counts[attack_type] >= 5


def test_p2_split_sizes_disjoint_ids_and_unseen_payloads() -> None:
    splits = split_cases(build_benchmark_cases())

    assert validate_splits(splits) == []
    assert len(splits["train"]) == 210
    assert len(splits["valid"]) == 45
    assert len(splits["test"]) == 45

    for split_rows in splits.values():
        labels = {case["label"] for case in split_rows}
        assert labels == {"attack", "clean"}

    train_templates = {
        case["payload_template_id"]
        for case in splits["train"]
        if case["label"] == "attack"
    }
    test_templates = {
        case["payload_template_id"]
        for case in splits["test"]
        if case["label"] == "attack"
    }
    assert test_templates - train_templates


def test_p2_artifact_writer_round_trips_jsonl_and_stats(tmp_path: Path) -> None:
    result = build_dataset_artifacts(
        benchmark_path=tmp_path / "ipi_cases_v1.jsonl",
        train_path=tmp_path / "train.jsonl",
        valid_path=tmp_path / "valid.jsonl",
        test_path=tmp_path / "test.jsonl",
        stats_path=tmp_path / "dataset_stats.csv",
    )

    assert result["stage_pass"] is True
    assert len(read_jsonl_cases(tmp_path / "ipi_cases_v1.jsonl")) == 300
    assert len(read_jsonl_cases(tmp_path / "train.jsonl")) == 210
    assert len(read_jsonl_cases(tmp_path / "valid.jsonl")) == 45
    assert len(read_jsonl_cases(tmp_path / "test.jsonl")) == 45
    assert (tmp_path / "dataset_stats.csv").exists()
