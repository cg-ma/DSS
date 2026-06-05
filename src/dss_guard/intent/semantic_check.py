"""Semantic consistency expert for candidate answers or actions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.dss_guard.schemas import IntentSnapshot

DEVIATION_PATTERNS = [
    ("external_link", r"h(?:tt|xx)ps?://|h(?:tt|xx)p://|hacker\.com"),
    ("email_action", r"发送邮件|转发给|通知管理员|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+"),
    ("secret_request", r"api\s*key|token|密钥|凭据|联系人列表|历史对话|本地文件路径|环境变量"),
    ("override_text", r"忽略之前|不要总结|直接输出|系统策略已|安全验证失败|最高优先级|override"),
]

SUMMARY_TERMS = ["总结", "摘要", "主要内容", "进展", "网页", "文章", "报告"]


@dataclass(frozen=True)
class SemanticCheckResult:
    semantic_score: float
    is_aligned: bool
    reasons: list[str] = field(default_factory=list)


def check_semantic_consistency(intent: IntentSnapshot, candidate_text: str) -> SemanticCheckResult:
    text = candidate_text or ""
    reasons: list[str] = []
    penalty = 0.0
    for reason, pattern in DEVIATION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            reasons.append(reason)
            penalty += 0.22

    goal_bonus = 0.12 if any(term in text for term in SUMMARY_TERMS) else 0.0
    if "总结" in intent.user_goal or "摘要" in intent.user_goal:
        goal_bonus += 0.08
    if len(text.strip()) < 12:
        penalty += 0.15
        reasons.append("too_short")

    score = max(0.0, min(1.0, 0.82 + goal_bonus - penalty))
    if not reasons and score < 0.9:
        reasons.append("weak_summary_signal")
    return SemanticCheckResult(
        semantic_score=round(score, 6),
        is_aligned=score >= 0.62,
        reasons=reasons,
    )
