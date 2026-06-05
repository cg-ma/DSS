"""P0 冒烟测试：不调用模型，只验证导入、数据结构和日志链路。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dss_guard.evaluation.metrics import ExperimentLogRecord, append_jsonl, jsonl_to_csv, read_jsonl
from src.dss_guard.schemas import DefenseVerdict, ExternalContent, RiskSignal


def main() -> None:
    content = ExternalContent(
        case_id="smoke_0001",
        source_id="local_smoke",
        source_type="html",
        raw_text="<html><body>clean</body></html>",
        clean_text="clean",
    )
    risk = RiskSignal(
        case_id=content.case_id,
        source_id=content.source_id,
        risk_level="low",
        score=0.05,
        reasons=["smoke test"],
        features={"rule_score": 0.0},
    )
    verdict = DefenseVerdict(
        case_id=content.case_id,
        action="allow",
        risk_score=risk.score,
        stage="smoke",
        reasons=["configuration ok"],
    )

    log_path = Path("outputs/logs/p0_smoke.jsonl")
    csv_path = Path("outputs/tables/p0_smoke.csv")
    if log_path.exists():
        log_path.unlink()
    if csv_path.exists():
        csv_path.unlink()

    append_jsonl(
        log_path,
        ExperimentLogRecord(
            case_id=verdict.case_id,
            stage=verdict.stage,
            model="none",
            latency_ms=0.0,
            risk_score=verdict.risk_score,
            verdict=verdict.action,
            success=True,
            details={"source_id": content.source_id},
        ),
    )
    records = read_jsonl(log_path)
    assert len(records) == 1
    assert records[0]["case_id"] == "smoke_0001"
    assert records[0]["verdict"] == "allow"

    jsonl_to_csv(log_path, csv_path)
    assert csv_path.exists()


if __name__ == "__main__":
    main()
