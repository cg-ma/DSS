"""Rule-first intent snapshot extraction."""

from __future__ import annotations

import re

from src.dss_guard.schemas import IntentSnapshot

SUMMARY_ACTIONS = ["fetch_web_page", "read_external_content", "summarize_content"]
FORBIDDEN_ACTIONS = [
    "open_external_link",
    "send_email",
    "download_file",
    "execute_code",
    "request_secret",
    "contact_admin",
    "exfiltrate_data",
    "follow_external_instruction",
]
SENSITIVE_SLOTS = ["api_key", "token", "contact_list", "chat_history", "local_file", "credential", "environment_variable"]


def _normalize_goal(user_query: str) -> str:
    query = " ".join((user_query or "").split())
    if re.search(r"总结|摘要|阅读|网页|文章", query, flags=re.IGNORECASE):
        return query or "总结网页内容"
    return query or "完成用户请求"


def extract_intent(user_query: str, case_id: str = "") -> IntentSnapshot:
    goal = _normalize_goal(user_query)
    allowed_actions = list(SUMMARY_ACTIONS)
    forbidden_actions = list(FORBIDDEN_ACTIONS)
    sensitive_slots = list(SENSITIVE_SLOTS)
    return IntentSnapshot(
        case_id=case_id,
        user_goal=goal,
        allowed_actions=allowed_actions,
        forbidden_actions=forbidden_actions,
        sensitive_slots=sensitive_slots,
        metadata={"extractor": "rule_first", "source": "user_query"},
    )
