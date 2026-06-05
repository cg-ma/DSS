"""Fixed-threshold arbitration baseline for P5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.dss_guard.arbitration.dynamic_policy import ArbitrationInput, normalize_arbitration_input
from src.dss_guard.schemas import DefenseVerdict


@dataclass(frozen=True)
class StaticPolicyConfig:
    sanitize_threshold: float = 0.35
    audit_threshold: float = 0.6
    block_threshold: float = 0.88


def _max_risk(inputs: ArbitrationInput) -> float:
    return max(
        inputs.p_perception,
        1.0 - inputs.s_semantic,
        inputs.p_logic,
        inputs.p_leakage,
        inputs.session_risk,
    )


def arbitrate_static(
    value: ArbitrationInput | dict[str, Any] | None,
    config: StaticPolicyConfig | None = None,
) -> DefenseVerdict:
    cfg = config or StaticPolicyConfig()
    inputs = normalize_arbitration_input(value)
    score = round(_max_risk(inputs), 6)
    if score >= cfg.block_threshold:
        action = "block"
        audit_depth = 2
        level = "static_block"
    elif score >= cfg.audit_threshold:
        action = "audit"
        audit_depth = 1
        level = "static_audit"
    elif score >= cfg.sanitize_threshold:
        action = "sanitize"
        audit_depth = 0
        level = "static_sanitize"
    else:
        action = "allow"
        audit_depth = 0
        level = "static_allow"

    reasons = [level]
    if inputs.expert_failures:
        reasons.extend(f"expert_failure:{failure}" for failure in inputs.expert_failures)
    return DefenseVerdict(
        case_id=inputs.case_id,
        action=action,  # type: ignore[arg-type]
        risk_score=score,
        reasons=reasons,
        stage="p5_static_arbitration",
        audit_depth=audit_depth,
        metadata={
            "policy": "static",
            "thresholds": {
                "sanitize": cfg.sanitize_threshold,
                "audit": cfg.audit_threshold,
                "block": cfg.block_threshold,
            },
            "input": {
                "p_perception": inputs.p_perception,
                "s_semantic": inputs.s_semantic,
                "p_logic": inputs.p_logic,
                "p_leakage": inputs.p_leakage,
                "tool_risk": inputs.tool_risk,
                "session_risk": inputs.session_risk,
                "expert_failures": inputs.expert_failures,
            },
        },
    )
