from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_ablation_eval import CandidateCache, _candidate_answer, run_eval  # noqa: E402


def test_ablation_eval_runs_all_variants_and_passes_stage_gate(tmp_path: Path) -> None:
    result = run_eval(
        type(
            "Args",
            (),
            {
                "input": "data/processed/test.jsonl",
                "metrics_output": str(tmp_path / "ablation_metrics.csv"),
                "log_output": str(tmp_path / "ablation_eval.jsonl"),
                "variant": "all",
                "stage": "p7_ablation_eval",
                "perception_threshold": 0.35,
                "llm_audit_call_ms": 80.0,
                "disable_stress_cases": False,
                "append": False,
                "require_stage_pass": False,
            },
        )()
    )

    metrics_text = (tmp_path / "ablation_metrics.csv").read_text(encoding="utf-8")
    first_log = json.loads((tmp_path / "ablation_eval.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert result["case_count"] == 69
    assert result["stress_case_count"] == 24
    assert result["variant_count"] == 6
    assert result["row_count"] == 414
    assert result["stage_pass"] is True
    assert "P7.5 ablation study passed" in metrics_text
    assert "no_sliding_window" in metrics_text
    assert "no_rules_ppl_features" in metrics_text
    assert "no_logic_leakage_experts" in metrics_text
    assert first_log["stage"] == "p7_ablation_eval"


def test_candidate_answer_api_cache_reuses_real_output(tmp_path: Path) -> None:
    class FakeBaseline:
        def __init__(self) -> None:
            self.config = type("Config", (), {"use_api": True})()
            self.model = "unit-api-model"
            self.calls = 0

        def get_external_content(self, case):
            return str(case["external_content"])

        def call_model(self, case, prompt_text, external_content):
            self.calls += 1
            assert "external body" in prompt_text
            return "api candidate"

    baseline = FakeBaseline()
    cache = CandidateCache(tmp_path / "candidate_cache.jsonl")
    case = {
        "case_id": "case_1",
        "user_query": "Please summarize.",
        "external_content": "external body",
    }

    first = _candidate_answer(baseline, case, cache)
    second = _candidate_answer(baseline, case, cache)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.text == second.text == "api candidate"
    assert baseline.calls == 1
    assert (tmp_path / "candidate_cache.jsonl").exists()
