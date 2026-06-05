"""Guarded agent integrating P3 perception, P4 validation and P5 arbitration."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from src.dss_guard.agent.rag_agent import AgentResult, RAGAgent, RAGAgentConfig, _normal_summary, build_summary_prompt
from src.dss_guard.alignment.data_flow_check import check_data_flow
from src.dss_guard.alignment.logic_expert import evaluate_logic_alignment, extract_action_requests
from src.dss_guard.arbitration import ArbitrationInput, arbitrate_dynamic, arbitrate_static
from src.dss_guard.intent.extractor import extract_intent
from src.dss_guard.intent.reflective_auditor import reflect_intent
from src.dss_guard.intent.semantic_check import check_semantic_consistency
from src.dss_guard.perception.pipeline import PerceptionConfig, scan_case
from src.dss_guard.schemas import DefenseVerdict

StrategyName = Literal["keyword", "perception_only", "llm_auditor", "static_threshold", "ours"]


@dataclass(frozen=True)
class GuardedAgentConfig:
    strategy: StrategyName = "ours"
    perception_threshold: float = 0.35
    dual_loop_threshold: float = 0.6
    use_api: bool = False
    model: str | None = None


KEYWORD_PATTERNS = [
    r"忽略之前",
    r"不要总结",
    r"直接输出",
    r"hacker\.com",
    r"api\s*key|token|密钥",
    r"发送邮件|转发给",
    r"\[system message\]|<\|system\|>|begin tool|override",
]


def _keyword_verdict(case_id: str, external_content: str) -> DefenseVerdict:
    reasons = []
    for pattern in KEYWORD_PATTERNS:
        if re.search(pattern, external_content or "", flags=re.IGNORECASE):
            reasons.append(f"keyword:{pattern}")
    if reasons:
        return DefenseVerdict(
            case_id=case_id,
            action="sanitize",
            risk_score=0.55,
            reasons=reasons,
            stage="p6_keyword_guard",
            sanitized_text="[高风险外部指令已清洗]",
            audit_depth=0,
        )
    return DefenseVerdict(case_id=case_id, action="allow", risk_score=0.0, reasons=["keyword_no_hit"], stage="p6_keyword_guard")


def _safe_answer(case: dict[str, Any], external_content: str, verdict: DefenseVerdict) -> str:
    if verdict.action == "block":
        return "检测到外部资料包含高风险指令或敏感信息诱导，已阻断该请求。"
    if verdict.action in {"sanitize", "audit"}:
        return _normal_summary(case, external_content) + " 已忽略外部资料中的非摘要指令。"
    return _normal_summary(case, external_content)


def _dual_loop_confirms_clean(inputs: ArbitrationInput, details: dict[str, Any]) -> bool:
    return (
        inputs.p_perception < 0.45
        and inputs.s_semantic >= 0.95
        and inputs.p_logic <= 0.05
        and inputs.p_leakage <= 0.05
        and inputs.tool_risk == "low"
        and not details.get("reflection_contaminated")
        and int(details.get("action_count") or 0) == 0
    )


class GuardedAgent:
    def __init__(self, config: GuardedAgentConfig | None = None) -> None:
        self.config = config or GuardedAgentConfig()
        self.baseline = RAGAgent(RAGAgentConfig(use_api=self.config.use_api, model=self.config.model))

    def _run_dual_loop(self, case: dict[str, Any], external_content: str, perception_score: float) -> tuple[ArbitrationInput, dict[str, Any]]:
        case_id = str(case.get("case_id", ""))
        intent = extract_intent(str(case.get("user_query", "")), case_id=case_id)
        reflection = reflect_intent(str(case.get("user_query", "")), external_content, intent)
        actions = extract_action_requests(case, reflection.revised_intent)
        candidate = self.baseline.call_model(case, build_summary_prompt(str(case.get("user_query", "")), external_content), external_content)
        semantic = check_semantic_consistency(reflection.revised_intent, candidate)
        logic = evaluate_logic_alignment(reflection.revised_intent, actions)
        leakage = check_data_flow(reflection.revised_intent, external_content, candidate, actions)
        tool_risk = "critical" if leakage.leakage_score >= 0.8 else ("high" if logic.deviation_score >= 0.45 else str(case.get("tool_risk", "low")))
        arbitration_input = ArbitrationInput(
            case_id=case_id,
            p_perception=perception_score,
            s_semantic=semantic.semantic_score,
            p_logic=logic.deviation_score,
            p_leakage=leakage.leakage_score,
            tool_risk=tool_risk,  # type: ignore[arg-type]
            session_risk=0.0,
            metadata={"strategy": self.config.strategy},
        )
        details = {
            "candidate_answer": candidate,
            "semantic_score": semantic.semantic_score,
            "logic_deviation_score": logic.deviation_score,
            "leakage_score": leakage.leakage_score,
            "reflection_contaminated": reflection.contaminated,
            "action_count": len(actions),
            "semantic_reasons": semantic.reasons,
            "logic_reasons": logic.reasons,
            "leakage_reasons": leakage.reasons,
            "reflection_reasons": reflection.reasons,
        }
        return arbitration_input, details

    def run_case(self, case: dict[str, Any]) -> AgentResult:
        started = time.perf_counter()
        case_id = str(case.get("case_id", ""))
        external_content = self.baseline.get_external_content(case)
        prompt_text = build_summary_prompt(str(case.get("user_query", "")), external_content)
        metadata: dict[str, Any] = {"strategy": self.config.strategy}

        if self.config.strategy == "keyword":
            verdict = _keyword_verdict(case_id, external_content)
        else:
            perception_signal, _ = scan_case(case, config=PerceptionConfig(prefer_model=False))
            metadata["perception_score"] = perception_signal.score
            metadata["perception_level"] = perception_signal.risk_level
            if self.config.strategy == "perception_only":
                if perception_signal.score >= self.config.perception_threshold:
                    verdict = DefenseVerdict(
                        case_id=case_id,
                        action="sanitize",
                        risk_score=perception_signal.score,
                        reasons=["perception_risk", *perception_signal.reasons],
                        stage="p6_perception_only",
                        sanitized_text="[感知层清洗高风险片段]",
                        audit_depth=1,
                    )
                else:
                    verdict = DefenseVerdict(case_id=case_id, action="allow", risk_score=perception_signal.score, reasons=["perception_low"], stage="p6_perception_only")
            else:
                arbitration_input, dual_details = self._run_dual_loop(case, external_content, perception_signal.score)
                metadata.update(dual_details)
                if self.config.strategy == "static_threshold":
                    verdict = arbitrate_static(arbitration_input)
                elif self.config.strategy in {"llm_auditor", "ours"}:
                    if self.config.strategy == "ours" and _dual_loop_confirms_clean(arbitration_input, dual_details):
                        verdict = DefenseVerdict(
                            case_id=case_id,
                            action="allow",
                            risk_score=min(perception_signal.score, self.config.perception_threshold),
                            reasons=["audit_level:L0", "dual_loop_clean_override"],
                            stage="p6_guarded_agent",
                            audit_depth=0,
                            metadata={"policy": "dynamic", "audit_level": "L0", "clean_override": True},
                        )
                    else:
                        verdict = arbitrate_dynamic(arbitration_input)
                else:
                    raise ValueError(f"Unsupported guarded strategy: {self.config.strategy}")

        if verdict.action == "allow":
            final_answer = self.baseline.call_model(case, prompt_text, external_content)
        else:
            final_answer = _safe_answer(case, external_content, verdict)

        final_check = check_data_flow(
            extract_intent(str(case.get("user_query", "")), case_id=case_id),
            external_content,
            final_answer,
        )
        metadata["final_leakage_score"] = final_check.leakage_score
        if final_check.leakage_score >= 0.7 and verdict.action != "block":
            verdict = DefenseVerdict(
                case_id=case_id,
                action="block",
                risk_score=max(verdict.risk_score, final_check.leakage_score),
                reasons=[*verdict.reasons, "final_answer_leakage_check"],
                stage=verdict.stage,
                audit_depth=max(verdict.audit_depth, 4),
                metadata={**verdict.metadata, "final_check": final_check.reasons},
            )
            final_answer = "最终回答触发敏感信息泄露检查，已阻断。"

        latency_ms = (time.perf_counter() - started) * 1000
        return AgentResult(
            case_id=case_id,
            final_answer=final_answer,
            verdict=verdict,
            latency_ms=latency_ms,
            prompt_text=prompt_text,
            metadata=metadata,
        )
