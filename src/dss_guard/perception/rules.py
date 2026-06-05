"""Rule-based detectors for external prompt injection signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.dss_guard.schemas import TextWindow


@dataclass(frozen=True)
class RuleMatch:
    reason: str
    start: int
    end: int
    weight: float


@dataclass(frozen=True)
class RuleScanResult:
    rule_score: float
    hidden_text_score: float
    matches: list[RuleMatch] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for match in self.matches:
            if match.reason not in seen:
                seen.add(match.reason)
                result.append(match.reason)
        return result


RULE_PATTERNS: list[tuple[str, str, float]] = [
    ("ignore_previous", r"忽\s*略\s*(之前|上文|所有)|ignore[_\s-]*previous", 0.58),
    ("stop_summary", r"不\s*要\s*总结|停止.{0,8}摘要|stop.{0,12}summary", 0.56),
    ("direct_output", r"直接输出|最终回答|final\s+(answer|response)|把最终回答替换", 0.46),
    ("secret_instruction", r"不要告诉用户|不要透露|隐藏系统消息|secret", 0.42),
    ("system_boundary", r"\[system message\]|<\|system\|>|</?system>|begin tool|developer note|important override", 0.72),
    ("highest_priority", r"最高优先级|拥有最高优先级|override|系统策略已", 0.58),
    ("external_link", r"https?://hacker\.com/[A-Za-z0-9_\-/.]+|hxxp://hacker\.com/[A-Za-z0-9_\-/.]+", 0.36),
    ("email_action", r"[A-Za-z0-9._%+-]+@(example|hacker)\.com|发送邮件|转发给|通知管理员", 0.38),
    ("leakage_request", r"api\s*key|token|密钥|凭据|联系人列表|历史对话|本地文件路径|环境变量|访问令牌", 0.34),
    ("action_request", r"打开.{0,8}链接|点击.{0,8}链接|调用.{0,8}工具|下载工具|访问\s*http", 0.34),
]

HIDDEN_PATTERNS: list[tuple[str, str, float]] = [
    ("css_display_none", r"display\s*:\s*none", 0.72),
    ("css_visibility_hidden", r"visibility\s*:\s*hidden", 0.68),
    ("css_white_on_white", r"color\s*:\s*white|background\s*:\s*white", 0.52),
    ("html_comment", r"<!--|-->", 0.46),
]

SAFE_CONTEXT_PATTERNS = [
    r"不能当作",
    r"不得",
    r"不代表",
    r"没有隐藏任务",
    r"只是",
    r"提醒",
    r"强调",
    r"解释",
    r"案例说明",
    r"合规",
    r"风险摘要",
    r"不执行任何操作",
]


def _safe_context(text: str, start: int, end: int) -> bool:
    context_start = max(0, start - 36)
    context_end = min(len(text), end + 36)
    context = text[context_start:context_end].lower()
    return any(re.search(pattern, context, flags=re.IGNORECASE) for pattern in SAFE_CONTEXT_PATTERNS)


def _collect_matches(text: str, patterns: list[tuple[str, str, float]]) -> list[RuleMatch]:
    matches: list[RuleMatch] = []
    for reason, pattern, weight in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            adjusted_weight = weight * 0.25 if _safe_context(text, match.start(), match.end()) else weight
            matches.append(RuleMatch(reason=reason, start=match.start(), end=match.end(), weight=adjusted_weight))
    return matches


def scan_rules(window: TextWindow) -> RuleScanResult:
    text = window.text or ""
    rule_matches = _collect_matches(text, RULE_PATTERNS)
    hidden_matches = _collect_matches(text, HIDDEN_PATTERNS)
    all_matches = rule_matches + hidden_matches

    rule_score = min(1.0, sum(match.weight for match in rule_matches))
    hidden_text_score = min(1.0, sum(match.weight for match in hidden_matches))
    if hidden_text_score and rule_score:
        rule_score = min(1.0, rule_score + hidden_text_score * 0.35)

    return RuleScanResult(
        rule_score=round(rule_score, 6),
        hidden_text_score=round(hidden_text_score, 6),
        matches=all_matches,
    )
