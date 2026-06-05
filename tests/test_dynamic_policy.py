from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dss_guard.arbitration import ArbitrationInput, arbitrate_dynamic, arbitrate_static, normalize_arbitration_input


def test_dynamic_policy_allows_low_risk_clean_without_audit() -> None:
    verdict = arbitrate_dynamic(ArbitrationInput(case_id="clean", p_perception=0.05, s_semantic=0.98))

    assert verdict.action == "allow"
    assert verdict.audit_depth == 0
    assert verdict.metadata["audit_level"] == "L0"


def test_dynamic_policy_sanitizes_medium_perception_risk() -> None:
    verdict = arbitrate_dynamic(ArbitrationInput(case_id="medium", p_perception=0.45, s_semantic=0.9))

    assert verdict.action in {"sanitize", "audit"}
    assert verdict.audit_depth >= 1
    assert "perception_risk" in verdict.reasons


def test_dynamic_policy_sends_logic_trap_to_deep_audit() -> None:
    verdict = arbitrate_dynamic(
        ArbitrationInput(
            case_id="logic",
            p_perception=0.3,
            s_semantic=0.58,
            p_logic=0.9,
            p_leakage=0.0,
            tool_risk="high",
        )
    )

    assert verdict.action in {"audit", "block"}
    assert verdict.audit_depth >= 3
    assert verdict.metadata["audit_level"] in {"L3", "L4"}
    assert "logic_deviation" in verdict.reasons


def test_dynamic_policy_blocks_critical_leakage() -> None:
    verdict = arbitrate_dynamic(
        ArbitrationInput(
            case_id="leakage",
            p_perception=0.4,
            s_semantic=0.68,
            p_logic=0.0,
            p_leakage=0.92,
            tool_risk="critical",
        )
    )

    assert verdict.action == "block"
    assert verdict.audit_depth == 4
    assert verdict.metadata["audit_level"] == "L4"
    assert "leakage_risk" in verdict.reasons


def test_dynamic_policy_handles_missing_and_failed_experts_conservatively() -> None:
    normalized = normalize_arbitration_input({"case_id": "partial", "p_perception": 0.1, "tool_risk": "unknown"})
    verdict = arbitrate_dynamic({"case_id": "partial", "p_perception": 0.1, "tool_risk": "unknown"})

    assert normalized.expert_failures
    assert verdict.action in {"sanitize", "audit"}
    assert verdict.audit_depth >= 1
    assert any(reason.startswith("expert_failure:") for reason in verdict.reasons)


def test_static_policy_uses_same_input_interface_but_fixed_depth() -> None:
    inputs = ArbitrationInput(case_id="compare", p_perception=0.3, s_semantic=0.58, p_logic=0.9, tool_risk="high")

    dynamic = arbitrate_dynamic(inputs)
    static = arbitrate_static(inputs)

    assert dynamic.case_id == static.case_id
    assert static.metadata["policy"] == "static"
    assert dynamic.metadata["policy"] == "dynamic"
    assert dynamic.audit_depth > static.audit_depth
