from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_arbitration_latency import run_eval  # noqa: E402


def _metric(rows: list[dict[str, str]], policy: str, risk_level: str) -> dict[str, str]:
    return next(row for row in rows if row["row_type"] == "risk_level" and row["policy"] == policy and row["risk_level"] == risk_level)


def test_arbitration_latency_eval_passes_stage_gate(tmp_path: Path) -> None:
    result = run_eval(
        type(
            "Args",
            (),
            {
                "metrics_output": str(tmp_path / "arbitration_latency.csv"),
                "log_output": str(tmp_path / "arbitration_latency.jsonl"),
                "policy": "all",
                "stage": "p7_arbitration_latency",
                "base_ttft_ms": 12.0,
                "base_e2e_ms": 36.0,
                "audit_step_ttft_ms": 5.0,
                "audit_depth_e2e_ms": 6.0,
                "llm_audit_call_ms": 80.0,
                "static_sanitize_threshold": 0.2,
                "static_audit_threshold": 0.2,
                "static_block_threshold": 0.85,
                "append": False,
                "require_stage_pass": False,
            },
        )()
    )

    metrics_rows = list(csv.DictReader((tmp_path / "arbitration_latency.csv").open(encoding="utf-8")))
    first_log = json.loads((tmp_path / "arbitration_latency.jsonl").read_text(encoding="utf-8").splitlines()[0])
    dynamic_low = _metric(metrics_rows, "dynamic_arbitration", "low")
    static_low = _metric(metrics_rows, "static_threshold", "low")
    dynamic_high = _metric(metrics_rows, "dynamic_arbitration", "high")
    static_high = _metric(metrics_rows, "static_threshold", "high")
    dynamic_critical = _metric(metrics_rows, "dynamic_arbitration", "critical")
    static_critical = _metric(metrics_rows, "static_threshold", "critical")

    assert result["sample_count"] == 24
    assert result["policy_count"] == 2
    assert result["row_count"] == 48
    assert result["stage_pass"] is True
    assert float(dynamic_low["avg_audit_count"]) < float(static_low["avg_audit_count"])
    assert float(dynamic_low["avg_ttft_ms"]) < float(static_low["avg_ttft_ms"])
    assert float(dynamic_high["avg_audit_depth"]) > float(static_high["avg_audit_depth"])
    assert float(dynamic_critical["avg_audit_depth"]) > float(static_critical["avg_audit_depth"])
    assert float(dynamic_critical["block_rate"]) == 1.0
    assert "P7.6 arbitration latency passed" in (tmp_path / "arbitration_latency.csv").read_text(encoding="utf-8")
    assert first_log["stage"] == "p7_arbitration_latency"
