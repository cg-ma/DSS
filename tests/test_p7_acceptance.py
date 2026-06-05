from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.generate_p7_artifacts import FIGURE_FILES, REQUIRED_TABLES, generate_artifacts  # noqa: E402


def test_p7_acceptance_generates_required_figures_and_report(tmp_path: Path) -> None:
    report_output = tmp_path / "p7_acceptance.csv"
    figures_dir = tmp_path / "figures"
    result = generate_artifacts(
        type(
            "Args",
            (),
            {
                "tables_dir": "outputs/tables",
                "logs_dir": "outputs/logs",
                "figures_dir": str(figures_dir),
                "report_output": str(report_output),
            },
        )()
    )

    rows = list(csv.DictReader(report_output.open(encoding="utf-8")))
    stage_gate = next(row for row in rows if row["artifact_type"] == "stage_gate")

    assert result["stage_pass"] is True
    assert result["table_count"] == len(REQUIRED_TABLES)
    assert result["figure_count"] == len(FIGURE_FILES)
    assert stage_gate["stage_pass"] == "True"
    for filename in FIGURE_FILES.values():
        path = figures_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0
