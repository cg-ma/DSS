from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_ablation_eval import run_eval  # noqa: E402


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
