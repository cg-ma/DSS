"""Generate P7 acceptance figures and validate required experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib_cache"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REQUIRED_TABLES = {
    "attack_reproduction": "attack_reproduction.csv",
    "perception_metrics": "perception_metrics.csv",
    "dual_loop_metrics": "dual_loop_metrics.csv",
    "end_to_end_metrics": "end_to_end_metrics.csv",
    "ablation_metrics": "ablation_metrics.csv",
    "arbitration_latency": "arbitration_latency.csv",
}
FIGURE_FILES = {
    "architecture": "architecture_overview.png",
    "perception": "perception_detection_comparison.png",
    "dual_loop": "dual_loop_blind_spot.png",
    "end_to_end": "end_to_end_asr_far.png",
    "latency": "latency_by_input_length.png",
    "arbitration": "dynamic_arbitration_depth.png",
}
REPORT_FIELDS = ["artifact_type", "name", "path", "exists", "stage_pass", "notes"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _float(row: dict[str, str], field_name: str, default: float = 0.0) -> float:
    try:
        value = row.get(field_name, "")
        return default if value == "" else float(value)
    except (TypeError, ValueError):
        return default


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999999) - 1))
    return ordered[index]


def _stage_gate_passed(rows: list[dict[str, str]]) -> bool:
    stage_rows = [row for row in rows if row.get("row_type") == "stage_gate"]
    return bool(stage_rows) and all(str(row.get("stage_pass", "")).lower() == "true" for row in stage_rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _strategy_rows(rows: list[dict[str, str]], key: str = "strategy") -> list[dict[str, str]]:
    return [row for row in rows if row.get("row_type") in {"strategy", "overall", "variant", "policy"} and row.get(key)]


def _draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_axis_off()
    nodes = [
        ("External\nSources", 0.03, 0.56, "#e8f1ff"),
        ("Agile\nPerception", 0.19, 0.56, "#e8fff2"),
        ("Intent\nSnapshot", 0.35, 0.56, "#fff7df"),
        ("MoE Dual-loop\nValidation", 0.51, 0.56, "#f3edff"),
        ("Dynamic\nArbitration", 0.69, 0.56, "#ffeceb"),
        ("Guarded Agent\nOutput", 0.86, 0.56, "#edf7f7"),
    ]
    for text, x, y, color in nodes:
        box = FancyBboxPatch(
            (x, y),
            0.12,
            0.22,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#2f3a4a",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + 0.06, y + 0.11, text, ha="center", va="center", fontsize=10, color="#1f2937")
    for index in range(len(nodes) - 1):
        _, x1, y1, _ = nodes[index]
        _, x2, y2, _ = nodes[index + 1]
        ax.annotate(
            "",
            xy=(x2 - 0.01, y2 + 0.11),
            xytext=(x1 + 0.13, y1 + 0.11),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#334155"},
        )
    ax.text(0.5, 0.24, "P3 low-latency signals + P4 intent validation + P5 adaptive audit depth", ha="center", fontsize=11, color="#334155")
    ax.set_title("DSS Defense Pipeline", fontsize=14, weight="bold")
    _save(fig, path)


def _draw_perception(rows: list[dict[str, str]], path: Path) -> None:
    data = [row for row in rows if row.get("row_type") == "overall"]
    methods = [row["method"] for row in data]
    x = range(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar([i - 0.2 for i in x], [_float(row, "precision") for row in data], width=0.2, label="Precision", color="#2f80ed")
    axes[0].bar(x, [_float(row, "recall") for row in data], width=0.2, label="Recall", color="#27ae60")
    axes[0].bar([i + 0.2 for i in x], [_float(row, "f1") for row in data], width=0.2, label="F1", color="#f2994a")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xticks(list(x), methods, rotation=20, ha="right")
    axes[0].set_title("Detection Quality")
    axes[0].legend()
    axes[1].bar(methods, [_float(row, "avg_latency_ms") for row in data], color="#6c5ce7")
    axes[1].set_title("Average Latency")
    axes[1].set_ylabel("ms")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("P7.2 Perception Performance", fontsize=14, weight="bold")
    fig.tight_layout()
    _save(fig, path)


def _draw_dual_loop(rows: list[dict[str, str]], path: Path) -> None:
    data = [row for row in rows if row.get("row_type") == "strategy"]
    strategies = [row["strategy"] for row in data]
    x = range(len(strategies))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar([i - 0.18 for i in x], [_float(row, "logic_trap_recall") for row in data], width=0.36, label="Logic Trap Recall", color="#d35400")
    ax.bar([i + 0.18 for i in x], [_float(row, "leakage_block_rate") for row in data], width=0.36, label="Leakage Block Rate", color="#16a085")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(list(x), strategies, rotation=18, ha="right")
    ax.set_title("P7.3 Blind Spot Coverage")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def _draw_end_to_end(rows: list[dict[str, str]], path: Path) -> None:
    data = [row for row in rows if row.get("row_type") == "strategy"]
    strategies = [row["strategy"] for row in data]
    x = range(len(strategies))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar([i - 0.2 for i in x], [_float(row, "asr") for row in data], width=0.4, label="ASR", color="#c0392b")
    ax.bar([i + 0.2 for i in x], [_float(row, "far") for row in data], width=0.4, label="FAR", color="#2980b9")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(list(x), strategies, rotation=20, ha="right")
    ax.set_title("P7.4 End-to-End ASR/FAR")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def _draw_latency_curve(ablation_log_path: Path, path: Path) -> None:
    import json

    groups: dict[int, list[float]] = {}
    if ablation_log_path.exists():
        with ablation_log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                record = json.loads(line)
                details = record.get("details") or {}
                if details.get("variant") != "full_system":
                    continue
                windows = int(details.get("perception_windows") or 1)
                groups.setdefault(windows, []).append(float(details.get("raw_latency_ms") or 0.0))
    xs = sorted(groups) or [1]
    avgs = [sum(groups.get(x, [0.0])) / len(groups.get(x, [0.0])) for x in xs]
    p95s = [_p95(groups.get(x, [0.0])) for x in xs]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(xs, avgs, marker="o", label="Avg raw latency", color="#2f80ed")
    ax.plot(xs, p95s, marker="s", label="P95 raw latency", color="#eb5757")
    ax.set_title("P7.5 Latency by Input Length Proxy")
    ax.set_xlabel("Perception windows")
    ax.set_ylabel("ms")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, path)


def _draw_arbitration(rows: list[dict[str, str]], path: Path) -> None:
    data = [row for row in rows if row.get("row_type") == "risk_level"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for policy, color, marker in [("static_threshold", "#7f8c8d", "o"), ("dynamic_arbitration", "#8e44ad", "s")]:
        policy_rows = [row for row in data if row.get("policy") == policy]
        values = [_float(next(row for row in policy_rows if row["risk_level"] == level), "avg_audit_depth") for level in ["low", "medium", "high", "critical"]]
        ax.plot(["low", "medium", "high", "critical"], values, marker=marker, linewidth=2, label=policy, color=color)
    ax.set_ylim(0, 4.4)
    ax.set_title("P7.6 Audit Depth by Risk Level")
    ax.set_ylabel("Average audit depth")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def _validate_tables(tables_dir: Path) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]], bool]:
    table_data: dict[str, list[dict[str, str]]] = {}
    report_rows: list[dict[str, Any]] = []
    all_passed = True
    for name, filename in REQUIRED_TABLES.items():
        path = tables_dir / filename
        exists = path.exists()
        rows = _read_csv(path) if exists else []
        passed = exists and _stage_gate_passed(rows)
        all_passed = all_passed and passed
        table_data[name] = rows
        report_rows.append(
            {
                "artifact_type": "table",
                "name": name,
                "path": str(path),
                "exists": exists,
                "stage_pass": passed,
                "notes": "stage gate passed" if passed else "missing or failed stage gate",
            }
        )
    return table_data, report_rows, all_passed


def generate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    tables_dir = Path(args.tables_dir)
    logs_dir = Path(args.logs_dir)
    figures_dir = Path(args.figures_dir)
    report_output = Path(args.report_output)
    table_data, report_rows, tables_passed = _validate_tables(tables_dir)

    figure_paths = {name: figures_dir / filename for name, filename in FIGURE_FILES.items()}
    _draw_architecture(figure_paths["architecture"])
    _draw_perception(table_data["perception_metrics"], figure_paths["perception"])
    _draw_dual_loop(table_data["dual_loop_metrics"], figure_paths["dual_loop"])
    _draw_end_to_end(table_data["end_to_end_metrics"], figure_paths["end_to_end"])
    _draw_latency_curve(logs_dir / "ablation_eval.jsonl", figure_paths["latency"])
    _draw_arbitration(table_data["arbitration_latency"], figure_paths["arbitration"])

    figures_passed = True
    for name, path in figure_paths.items():
        exists = path.exists() and path.stat().st_size > 0
        figures_passed = figures_passed and exists
        report_rows.append(
            {
                "artifact_type": "figure",
                "name": name,
                "path": str(path),
                "exists": exists,
                "stage_pass": exists,
                "notes": "generated" if exists else "missing",
            }
        )

    stage_pass = tables_passed and figures_passed
    report_rows.append(
        {
            "artifact_type": "stage_gate",
            "name": "p7_acceptance",
            "path": str(report_output),
            "exists": True,
            "stage_pass": stage_pass,
            "notes": "P7.7 acceptance passed" if stage_pass else "P7.7 acceptance below threshold",
        }
    )
    _write_report(report_output, report_rows)
    return {
        "stage_pass": stage_pass,
        "table_count": len(REQUIRED_TABLES),
        "figure_count": len(FIGURE_FILES),
        "report_output": str(report_output),
        "figures_dir": str(figures_dir),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate P7 acceptance figures and report")
    parser.add_argument("--tables-dir", default="outputs/tables")
    parser.add_argument("--logs-dir", default="outputs/logs")
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument("--report-output", default="outputs/tables/p7_acceptance.csv")
    parser.add_argument("--require-stage-pass", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = generate_artifacts(args)
    print(
        "P7 artifacts: "
        f"tables={result['table_count']}, "
        f"figures={result['figure_count']}, "
        f"stage_pass={result['stage_pass']}, "
        f"report={result['report_output']}"
    )
    if args.require_stage_pass and not result["stage_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
