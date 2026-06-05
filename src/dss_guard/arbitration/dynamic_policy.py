"""Dynamic risk arbitration policy for P5."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.dss_guard.schemas import DefenseVerdict, ToolRisk

TOOL_RISK_SCORE: dict[str, float] = {
    "low": 0.05,
    "medium": 0.32,
    "high": 0.68,
    "critical": 0.95,
}


@dataclass(frozen=True)
class ArbitrationInput:
    case_id: str = ""
    p_perception: float = 0.0
    s_semantic: float = 1.0
    p_logic: float = 0.0
    p_leakage: float = 0.0
    tool_risk: ToolRisk = "low"
    session_risk: float = 0.0
    expert_failures: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicPolicyConfig:
    l1_threshold: float = 0.35
    l2_threshold: float = 0.6
    l3_threshold: float = 0.82
    l4_threshold: float = 0.9
    failure_floor: float = 0.55


def _clamp(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def normalize_arbitration_input(value: ArbitrationInput | dict[str, Any] | None) -> ArbitrationInput:
    """Normalize missing or partially failed expert outputs conservatively."""

    if value is None:
        return ArbitrationInput(expert_failures=["missing_input"])
    if isinstance(value, ArbitrationInput):
        return ArbitrationInput(
            case_id=value.case_id,
            p_perception=_clamp(value.p_perception, 0.0),
            s_semantic=_clamp(value.s_semantic, 1.0),
            p_logic=_clamp(value.p_logic, 0.0),
            p_leakage=_clamp(value.p_leakage, 0.0),
            tool_risk=value.tool_risk if value.tool_risk in TOOL_RISK_SCORE else "medium",
            session_risk=_clamp(value.session_risk, 0.0),
            expert_failures=list(value.expert_failures),
            metadata=dict(value.metadata),
        )
    failures = list(value.get("expert_failures") or [])
    for field_name in ("p_perception", "s_semantic", "p_logic", "p_leakage"):
        if field_name not in value or value.get(field_name) is None:
            failures.append(f"missing:{field_name}")
    tool_risk = value.get("tool_risk", "low")
    if tool_risk not in TOOL_RISK_SCORE:
        failures.append("invalid:tool_risk")
        tool_risk = "medium"
    return ArbitrationInput(
        case_id=str(value.get("case_id", "")),
        p_perception=_clamp(value.get("p_perception"), 0.0),
        s_semantic=_clamp(value.get("s_semantic"), 1.0),
        p_logic=_clamp(value.get("p_logic"), 0.0),
        p_leakage=_clamp(value.get("p_leakage"), 0.0),
        tool_risk=tool_risk,
        session_risk=_clamp(value.get("session_risk"), 0.0),
        expert_failures=list(dict.fromkeys(failures)),
        metadata=dict(value.get("metadata") or {}),
    )


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _risk_score(inputs: ArbitrationInput, config: DynamicPolicyConfig) -> tuple[float, dict[str, float]]:
    semantic_risk = 1.0 - inputs.s_semantic
    tool_score = TOOL_RISK_SCORE.get(inputs.tool_risk, 0.32)
    failure_score = min(1.0, len(inputs.expert_failures) * 0.22)
    linear = (
        2.0 * inputs.p_perception
        + 2.3 * semantic_risk
        + 2.6 * inputs.p_logic
        + 2.9 * inputs.p_leakage
        + 1.4 * tool_score
        + 1.2 * inputs.session_risk
        + 1.1 * failure_score
        - 2.2
    )
    score = _sigmoid(linear)
    if inputs.p_leakage >= 0.85:
        score = max(score, config.l4_threshold)
    if inputs.p_logic >= 0.75 and tool_score >= 0.68:
        score = max(score, config.l3_threshold)
    if inputs.p_perception >= 0.35 or semantic_risk >= 0.35:
        score = max(score, config.l1_threshold)
    if inputs.expert_failures:
        score = max(score, config.failure_floor)
    components = {
        "p_perception": inputs.p_perception,
        "semantic_risk": semantic_risk,
        "p_logic": inputs.p_logic,
        "p_leakage": inputs.p_leakage,
        "tool_risk_score": tool_score,
        "session_risk": inputs.session_risk,
        "failure_score": failure_score,
    }
    return round(min(1.0, score), 6), components


def _level_and_action(score: float, inputs: ArbitrationInput, config: DynamicPolicyConfig) -> tuple[str, str, int]:
    if inputs.p_leakage >= 0.88 or score >= config.l4_threshold:
        return "L4", "block", 4
    if score >= config.l3_threshold:
        return "L3", "audit", 3
    if score >= config.l2_threshold:
        return "L2", "audit", 2
    if score >= config.l1_threshold:
        return "L1", "sanitize", 1
    return "L0", "allow", 0


def _reasons(inputs: ArbitrationInput, components: dict[str, float], level: str) -> list[str]:
    reasons = [f"audit_level:{level}"]
    if inputs.p_perception >= 0.35:
        reasons.append("perception_risk")
    if components["semantic_risk"] >= 0.35:
        reasons.append("semantic_mismatch")
    if inputs.p_logic >= 0.35:
        reasons.append("logic_deviation")
    if inputs.p_leakage >= 0.35:
        reasons.append("leakage_risk")
    if components["tool_risk_score"] >= 0.32:
        reasons.append(f"tool_risk:{inputs.tool_risk}")
    if inputs.session_risk >= 0.35:
        reasons.append("session_risk")
    for failure in inputs.expert_failures:
        reasons.append(f"expert_failure:{failure}")
    return reasons


def arbitrate_dynamic(
    value: ArbitrationInput | dict[str, Any] | None,
    config: DynamicPolicyConfig | None = None,
) -> DefenseVerdict:
    cfg = config or DynamicPolicyConfig()
    inputs = normalize_arbitration_input(value)
    score, components = _risk_score(inputs, cfg)
    level, action, audit_depth = _level_and_action(score, inputs, cfg)
    return DefenseVerdict(
        case_id=inputs.case_id,
        action=action,  # type: ignore[arg-type]
        risk_score=score,
        reasons=_reasons(inputs, components, level),
        stage="p5_dynamic_arbitration",
        audit_depth=audit_depth,
        metadata={
            "policy": "dynamic",
            "audit_level": level,
            "components": components,
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
