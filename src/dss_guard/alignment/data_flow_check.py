"""Data leakage risk expert for untrusted-content influenced outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.dss_guard.schemas import ActionRequest, IntentSnapshot

SENSITIVE_PATTERNS = [
    ("api_key", r"api\s*key|apikey|密钥"),
    ("token", r"\btoken\b|访问令牌|认证令牌"),
    ("credential", r"凭据|密码|secret"),
    ("contact_list", r"联系人列表|通讯录"),
    ("chat_history", r"历史对话|对话摘要|聊天记录"),
    ("local_file", r"本地文件路径|私有目录|配置文件"),
    ("environment_variable", r"环境变量"),
]

SAFE_LEAKAGE_CONTEXT = [r"不得", r"无法", r"不会", r"不能", r"隐私保护", r"不应", r"安全培训", r"合规"]


@dataclass(frozen=True)
class LeakageCheckResult:
    leakage_score: float
    is_safe: bool
    sensitive_slots: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _safe_context(text: str, start: int, end: int) -> bool:
    context = text[max(0, start - 24) : min(len(text), end + 32)]
    return any(re.search(pattern, context, flags=re.IGNORECASE) for pattern in SAFE_LEAKAGE_CONTEXT)


def _sensitive_hits(text: str, ignore_safe_context: bool) -> tuple[list[str], list[str]]:
    slots: list[str] = []
    reasons: list[str] = []
    for slot, pattern in SENSITIVE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if ignore_safe_context and _safe_context(text, match.start(), match.end()):
                continue
            slots.append(slot)
            reasons.append(f"sensitive_slot:{slot}")
            break
    return list(dict.fromkeys(slots)), list(dict.fromkeys(reasons))


def check_data_flow(
    intent: IntentSnapshot,
    external_content: str,
    candidate_output: str = "",
    action_requests: list[ActionRequest] | None = None,
) -> LeakageCheckResult:
    output_slots, output_reasons = _sensitive_hits(candidate_output or "", ignore_safe_context=False)
    external_slots, external_reasons = _sensitive_hits(external_content or "", ignore_safe_context=True)
    action_requests = action_requests or []

    reasons = output_reasons + external_reasons
    slots = list(dict.fromkeys(output_slots + external_slots))
    score = 0.0
    if output_slots:
        score += 0.7
        reasons.append("candidate_output_contains_sensitive_slot")
    if external_slots and re.search(r"输出|列出|展示|读取|获取|给出|泄露", external_content or "", flags=re.IGNORECASE):
        score += 0.68
        reasons.append("external_content_requests_sensitive_output")
    if any(request.tool_name == "request_secret" for request in action_requests):
        score += 0.45
        reasons.append("action_requests_secret")
    if "request_secret" in intent.forbidden_actions and slots:
        score += 0.12
        reasons.append("violates_sensitive_boundary")

    score = min(1.0, score)
    return LeakageCheckResult(
        leakage_score=round(score, 6),
        is_safe=score < 0.45,
        sensitive_slots=slots,
        reasons=list(dict.fromkeys(reasons)),
    )
