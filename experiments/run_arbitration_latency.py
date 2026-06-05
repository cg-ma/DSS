"""Run P7.6 dynamic arbitration latency and usability evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dss_guard.arbitration import ArbitrationInput, StaticPolicyConfig, arbitrate_dynamic, arbitrate_static  # noqa: E402
from src.dss_guard.evaluation.metrics import ExperimentLogRecord, append_jsonl  # noqa: E402
from src.dss_guard.schemas import DefenseVerdict  # noqa: E402

DEFAULT_LOG_OUTPUT = "outputs/logs/arbitration_latency.jsonl"
DEFAULT_METRICS_OUTPUT = "outputs/tables/arbitration_latency.csv"

POLICIES = ["static_threshold", "dynamic_arbitration"]
RISK_LEVELS = ["low", "medium", "high", "critical"]
METRIC_FIELDS = [
    "row_type",
    "policy",
    "risk_level",
    "sample_count",
    "avg_risk_score",
    "allow_rate",
    "sanitize_rate",
    "audit_action_rate",
    "block_rate",
    "audit_rate",
    "avg_audit_depth",
    "max_audit_depth",
    "avg_audit_count",
    "avg_llm_audit_calls",
    "avg_ttft_ms",
    "p95_ttft_ms",
    "avg_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "avg_raw_latency_ms",
    "p95_raw_latency_ms",
    "stage_pass",
    "notes",
]


@dataclass(frozen=True)
class ArbitrationSample:
    case_id: str
    risk_level: str
    input: ArbitrationInput
    expected_min_depth: int
    expected_max_depth: int


@dataclass(frozen=True)
class LatencyConfig:
    base_ttft_ms: float = 12.0
    base_e2e_ms: float = 36.0
    audit_step_ttft_ms: float = 5.0
    audit_depth_e2e_ms: float = 6.0
    llm_audit_call_ms: float = 80.0
    stage: str = "p7_arbitration_latency"


def _round(value: float) -> float:
    return round(value, 6)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _samples() -> list[ArbitrationSample]:
    data: list[tuple[str, str, ArbitrationInput, int, int]] = [
        (
            "low_clean_hint_01",
            "low",
            ArbitrationInput(case_id="low_clean_hint_01", p_perception=0.18, s_semantic=0.98, p_logic=0.0, p_leakage=0.0, tool_risk="low"),
            0,
            0,
        ),
        (
            "low_clean_hint_02",
            "low",
            ArbitrationInput(case_id="low_clean_hint_02", p_perception=0.22, s_semantic=0.97, p_logic=0.02, p_leakage=0.0, tool_risk="low"),
            0,
            0,
        ),
        (
            "low_clean_hint_03",
            "low",
            ArbitrationInput(case_id="low_clean_hint_03", p_perception=0.26, s_semantic=0.96, p_logic=0.02, p_leakage=0.02, tool_risk="low"),
            0,
            0,
        ),
        (
            "low_clean_hint_04",
            "low",
            ArbitrationInput(case_id="low_clean_hint_04", p_perception=0.31, s_semantic=0.95, p_logic=0.03, p_leakage=0.02, tool_risk="low"),
            0,
            0,
        ),
        (
            "low_clean_hint_05",
            "low",
            ArbitrationInput(case_id="low_clean_hint_05", p_perception=0.28, s_semantic=0.97, p_logic=0.0, p_leakage=0.03, tool_risk="low"),
            0,
            0,
        ),
        (
            "low_clean_hint_06",
            "low",
            ArbitrationInput(case_id="low_clean_hint_06", p_perception=0.2, s_semantic=0.99, p_logic=0.01, p_leakage=0.0, tool_risk="low"),
            0,
            0,
        ),
        (
            "medium_prompt_risk_01",
            "medium",
            ArbitrationInput(case_id="medium_prompt_risk_01", p_perception=0.39, s_semantic=0.91, p_logic=0.08, p_leakage=0.04, tool_risk="low"),
            1,
            2,
        ),
        (
            "medium_prompt_risk_02",
            "medium",
            ArbitrationInput(case_id="medium_prompt_risk_02", p_perception=0.43, s_semantic=0.89, p_logic=0.12, p_leakage=0.06, tool_risk="medium"),
            1,
            2,
        ),
        (
            "medium_prompt_risk_03",
            "medium",
            ArbitrationInput(case_id="medium_prompt_risk_03", p_perception=0.46, s_semantic=0.87, p_logic=0.16, p_leakage=0.08, tool_risk="medium"),
            1,
            2,
        ),
        (
            "medium_prompt_risk_04",
            "medium",
            ArbitrationInput(case_id="medium_prompt_risk_04", p_perception=0.5, s_semantic=0.84, p_logic=0.22, p_leakage=0.1, tool_risk="medium"),
            1,
            3,
        ),
        (
            "medium_prompt_risk_05",
            "medium",
            ArbitrationInput(case_id="medium_prompt_risk_05", p_perception=0.48, s_semantic=0.86, p_logic=0.18, p_leakage=0.12, tool_risk="medium"),
            1,
            3,
        ),
        (
            "medium_prompt_risk_06",
            "medium",
            ArbitrationInput(case_id="medium_prompt_risk_06", p_perception=0.41, s_semantic=0.9, p_logic=0.14, p_leakage=0.07, tool_risk="low"),
            1,
            2,
        ),
        (
            "high_logic_risk_01",
            "high",
            ArbitrationInput(case_id="high_logic_risk_01", p_perception=0.36, s_semantic=0.68, p_logic=0.72, p_leakage=0.14, tool_risk="high"),
            3,
            4,
        ),
        (
            "high_logic_risk_02",
            "high",
            ArbitrationInput(case_id="high_logic_risk_02", p_perception=0.42, s_semantic=0.62, p_logic=0.78, p_leakage=0.2, tool_risk="high"),
            3,
            4,
        ),
        (
            "high_logic_risk_03",
            "high",
            ArbitrationInput(case_id="high_logic_risk_03", p_perception=0.5, s_semantic=0.6, p_logic=0.82, p_leakage=0.24, tool_risk="high"),
            3,
            4,
        ),
        (
            "high_logic_risk_04",
            "high",
            ArbitrationInput(case_id="high_logic_risk_04", p_perception=0.47, s_semantic=0.58, p_logic=0.86, p_leakage=0.28, tool_risk="high"),
            3,
            4,
        ),
        (
            "high_logic_risk_05",
            "high",
            ArbitrationInput(case_id="high_logic_risk_05", p_perception=0.54, s_semantic=0.64, p_logic=0.76, p_leakage=0.18, tool_risk="high"),
            3,
            4,
        ),
        (
            "high_logic_risk_06",
            "high",
            ArbitrationInput(case_id="high_logic_risk_06", p_perception=0.4, s_semantic=0.7, p_logic=0.75, p_leakage=0.16, tool_risk="high"),
            3,
            4,
        ),
        (
            "critical_leakage_01",
            "critical",
            ArbitrationInput(case_id="critical_leakage_01", p_perception=0.45, s_semantic=0.7, p_logic=0.3, p_leakage=0.88, tool_risk="critical"),
            4,
            4,
        ),
        (
            "critical_leakage_02",
            "critical",
            ArbitrationInput(case_id="critical_leakage_02", p_perception=0.52, s_semantic=0.66, p_logic=0.36, p_leakage=0.9, tool_risk="critical"),
            4,
            4,
        ),
        (
            "critical_leakage_03",
            "critical",
            ArbitrationInput(case_id="critical_leakage_03", p_perception=0.58, s_semantic=0.6, p_logic=0.42, p_leakage=0.93, tool_risk="critical"),
            4,
            4,
        ),
        (
            "critical_leakage_04",
            "critical",
            ArbitrationInput(case_id="critical_leakage_04", p_perception=0.62, s_semantic=0.55, p_logic=0.48, p_leakage=0.95, tool_risk="critical"),
            4,
            4,
        ),
        (
            "critical_leakage_05",
            "critical",
            ArbitrationInput(case_id="critical_leakage_05", p_perception=0.5, s_semantic=0.64, p_logic=0.32, p_leakage=0.91, tool_risk="critical"),
            4,
            4,
        ),
        (
            "critical_leakage_06",
            "critical",
            ArbitrationInput(case_id="critical_leakage_06", p_perception=0.56, s_semantic=0.58, p_logic=0.4, p_leakage=0.94, tool_risk="critical"),
            4,
            4,
        ),
    ]
    return [
        ArbitrationSample(
            case_id=case_id,
            risk_level=risk_level,
            input=arbitration_input,
            expected_min_depth=expected_min_depth,
            expected_max_depth=expected_max_depth,
        )
        for case_id, risk_level, arbitration_input, expected_min_depth, expected_max_depth in data
    ]


def _static_config(args: argparse.Namespace) -> StaticPolicyConfig:
    return StaticPolicyConfig(
        sanitize_threshold=args.static_sanitize_threshold,
        audit_threshold=args.static_audit_threshold,
        block_threshold=args.static_block_threshold,
    )


def _policy_verdict(policy: str, sample: ArbitrationSample, args: argparse.Namespace) -> DefenseVerdict:
    if policy == "dynamic_arbitration":
        return arbitrate_dynamic(sample.input)
    if policy == "static_threshold":
        return arbitrate_static(sample.input, config=_static_config(args))
    raise ValueError(f"Unsupported policy: {policy}")


def _audit_count(verdict: DefenseVerdict) -> int:
    return 1 if verdict.audit_depth > 0 else 0


def _llm_audit_calls(verdict: DefenseVerdict) -> int:
    return 1 if verdict.audit_depth >= 3 else 0


def _latency(row: dict[str, Any], cfg: LatencyConfig) -> tuple[float, float]:
    audit_count = int(row["audit_count"])
    depth = int(row["audit_depth"])
    llm_calls = int(row["llm_audit_calls"])
    ttft_ms = cfg.base_ttft_ms + audit_count * cfg.audit_step_ttft_ms + llm_calls * (cfg.llm_audit_call_ms * 0.35)
    e2e_ms = cfg.base_e2e_ms + row["raw_latency_ms"] + depth * cfg.audit_depth_e2e_ms + llm_calls * cfg.llm_audit_call_ms
    return ttft_ms, e2e_ms


def _input_details(value: ArbitrationInput) -> dict[str, Any]:
    return {
        "p_perception": value.p_perception,
        "s_semantic": value.s_semantic,
        "p_logic": value.p_logic,
        "p_leakage": value.p_leakage,
        "tool_risk": value.tool_risk,
        "session_risk": value.session_risk,
        "expert_failures": value.expert_failures,
    }


def _run_case(policy: str, sample: ArbitrationSample, args: argparse.Namespace, cfg: LatencyConfig) -> dict[str, Any]:
    started = time.perf_counter()
    verdict = _policy_verdict(policy, sample, args)
    raw_latency_ms = (time.perf_counter() - started) * 1000
    row = {
        "policy": policy,
        "case_id": sample.case_id,
        "risk_level": sample.risk_level,
        "verdict_action": verdict.action,
        "risk_score": verdict.risk_score,
        "audit_depth": verdict.audit_depth,
        "audit_count": _audit_count(verdict),
        "llm_audit_calls": _llm_audit_calls(verdict),
        "raw_latency_ms": raw_latency_ms,
        "expected_min_depth": sample.expected_min_depth,
        "expected_max_depth": sample.expected_max_depth,
        "input": _input_details(sample.input),
        "reasons": verdict.reasons,
        "metadata": verdict.metadata,
    }
    ttft_ms, e2e_latency_ms = _latency(row, cfg)
    row["ttft_ms"] = ttft_ms
    row["e2e_latency_ms"] = e2e_latency_ms
    row["meets_expected_depth"] = sample.expected_min_depth <= verdict.audit_depth <= sample.expected_max_depth
    return row


def _rate(rows: list[dict[str, Any]], field_name: str, value: Any) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[field_name] == value) / len(rows)


def _mean(rows: list[dict[str, Any]], field_name: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[field_name]) for row in rows) / len(rows)


def _metric_row(
    row_type: str,
    policy: str,
    risk_level: str,
    rows: list[dict[str, Any]],
    stage_pass: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    ttfts = [float(row["ttft_ms"]) for row in rows]
    e2es = [float(row["e2e_latency_ms"]) for row in rows]
    raw_latencies = [float(row["raw_latency_ms"]) for row in rows]
    return {
        "row_type": row_type,
        "policy": policy,
        "risk_level": risk_level,
        "sample_count": len(rows),
        "avg_risk_score": _round(_mean(rows, "risk_score")),
        "allow_rate": _round(_rate(rows, "verdict_action", "allow")),
        "sanitize_rate": _round(_rate(rows, "verdict_action", "sanitize")),
        "audit_action_rate": _round(_rate(rows, "verdict_action", "audit")),
        "block_rate": _round(_rate(rows, "verdict_action", "block")),
        "audit_rate": _round(_mean(rows, "audit_count")),
        "avg_audit_depth": _round(_mean(rows, "audit_depth")),
        "max_audit_depth": max((int(row["audit_depth"]) for row in rows), default=0),
        "avg_audit_count": _round(_mean(rows, "audit_count")),
        "avg_llm_audit_calls": _round(_mean(rows, "llm_audit_calls")),
        "avg_ttft_ms": _round(sum(ttfts) / len(ttfts)) if ttfts else 0.0,
        "p95_ttft_ms": _round(_p95(ttfts)),
        "avg_e2e_latency_ms": _round(sum(e2es) / len(e2es)) if e2es else 0.0,
        "p95_e2e_latency_ms": _round(_p95(e2es)),
        "avg_raw_latency_ms": _round(sum(raw_latencies) / len(raw_latencies)) if raw_latencies else 0.0,
        "p95_raw_latency_ms": _round(_p95(raw_latencies)),
        "stage_pass": stage_pass,
        "notes": notes,
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _stage_pass(metric_rows: dict[tuple[str, str], dict[str, Any]]) -> bool:
    dynamic_low = metric_rows.get(("dynamic_arbitration", "low"))
    static_low = metric_rows.get(("static_threshold", "low"))
    dynamic_high = metric_rows.get(("dynamic_arbitration", "high"))
    static_high = metric_rows.get(("static_threshold", "high"))
    dynamic_critical = metric_rows.get(("dynamic_arbitration", "critical"))
    static_critical = metric_rows.get(("static_threshold", "critical"))
    if not all([dynamic_low, static_low, dynamic_high, static_high, dynamic_critical, static_critical]):
        return False
    return bool(
        dynamic_low["avg_audit_count"] < static_low["avg_audit_count"]
        and dynamic_low["avg_ttft_ms"] < static_low["avg_ttft_ms"]
        and dynamic_high["avg_audit_depth"] > static_high["avg_audit_depth"]
        and dynamic_critical["avg_audit_depth"] > static_critical["avg_audit_depth"]
        and dynamic_critical["block_rate"] >= 1.0
    )


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    selected_policies = POLICIES if args.policy == "all" else [args.policy]
    cfg = LatencyConfig(
        base_ttft_ms=args.base_ttft_ms,
        base_e2e_ms=args.base_e2e_ms,
        audit_step_ttft_ms=args.audit_step_ttft_ms,
        audit_depth_e2e_ms=args.audit_depth_e2e_ms,
        llm_audit_call_ms=args.llm_audit_call_ms,
        stage=args.stage,
    )
    log_path = Path(args.log_output)
    if log_path.exists() and not args.append:
        log_path.unlink()

    samples = _samples()
    rows: list[dict[str, Any]] = []
    for policy in selected_policies:
        for sample in samples:
            row = _run_case(policy, sample, args, cfg)
            rows.append(row)
            append_jsonl(
                log_path,
                ExperimentLogRecord(
                    case_id=str(row["case_id"]),
                    stage=cfg.stage,
                    model=policy,
                    latency_ms=row["e2e_latency_ms"],
                    risk_score=row["risk_score"],
                    verdict=row["verdict_action"],
                    success=bool(row["meets_expected_depth"]),
                    details=row,
                ),
            )

    metric_rows: list[dict[str, Any]] = []
    metric_index: dict[tuple[str, str], dict[str, Any]] = {}
    for policy in selected_policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        metric_rows.append(_metric_row("policy", policy, "ALL", policy_rows))
        for risk_level in RISK_LEVELS:
            group = [row for row in policy_rows if row["risk_level"] == risk_level]
            metric = _metric_row("risk_level", policy, risk_level, group)
            metric_index[(policy, risk_level)] = metric
            metric_rows.append(metric)

    stage_pass = _stage_pass(metric_index) if set(POLICIES).issubset(set(selected_policies)) else bool(rows)
    if "dynamic_arbitration" in selected_policies:
        dynamic_rows = [row for row in rows if row["policy"] == "dynamic_arbitration"]
        metric_rows.append(
            _metric_row(
                "stage_gate",
                "dynamic_arbitration",
                "ALL",
                dynamic_rows,
                stage_pass=stage_pass,
                notes="P7.6 arbitration latency passed" if stage_pass else "P7.6 arbitration latency below threshold",
            )
        )

    _write_csv(args.metrics_output, metric_rows)
    return {
        "stage_pass": stage_pass,
        "sample_count": len(samples),
        "policy_count": len(selected_policies),
        "row_count": len(rows),
        "metrics_output": args.metrics_output,
        "log_output": args.log_output,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P7.6 dynamic arbitration latency evaluation")
    parser.add_argument("--metrics-output", default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--log-output", default=DEFAULT_LOG_OUTPUT)
    parser.add_argument("--policy", choices=["all", *POLICIES], default="all")
    parser.add_argument("--stage", default="p7_arbitration_latency")
    parser.add_argument("--base-ttft-ms", type=float, default=12.0)
    parser.add_argument("--base-e2e-ms", type=float, default=36.0)
    parser.add_argument("--audit-step-ttft-ms", type=float, default=5.0)
    parser.add_argument("--audit-depth-e2e-ms", type=float, default=6.0)
    parser.add_argument("--llm-audit-call-ms", type=float, default=80.0)
    parser.add_argument("--static-sanitize-threshold", type=float, default=0.2)
    parser.add_argument("--static-audit-threshold", type=float, default=0.2)
    parser.add_argument("--static-block-threshold", type=float, default=0.85)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--require-stage-pass", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_eval(args)
    print(
        "Arbitration latency eval: "
        f"stage={args.stage}, "
        f"samples={result['sample_count']}, "
        f"policies={result['policy_count']}, "
        f"rows={result['row_count']}, "
        f"stage_pass={result['stage_pass']}"
    )
    if args.require_stage_pass and not result["stage_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
