"""Dataset loading helpers shared by experiment entry points."""

from __future__ import annotations

from pathlib import Path

from src.dss_guard.data.benchmark import build_benchmark_cases, read_jsonl_cases, split_cases


def load_processed_split(path: str | Path, split_name: str | None = None) -> list[dict]:
    input_path = Path(path)
    if input_path.exists():
        return read_jsonl_cases(input_path)

    name = split_name or input_path.stem
    if name not in {"train", "valid", "test"}:
        raise FileNotFoundError(f"找不到数据文件：{input_path}")
    return split_cases(build_benchmark_cases())[name]
