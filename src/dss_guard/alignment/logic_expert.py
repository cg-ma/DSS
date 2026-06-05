"""Logic alignment expert for candidate actions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.dss_guard.schemas import ActionRequest, IntentSnapshot


@dataclass(frozen=True)
class LogicCheckResult:
    logic_score: float
    deviation_score: float
    is_aligned: bool
    reasons: list[str] = field(default_factory=list)
    action_count: int = 0


def extract_action_requests(case: dict[str, Any], intent: IntentSnapshot) -> list[ActionRequest]:
    text = str(case.get("external_content", ""))
    case_id = str(case.get("case_id", intent.case_id))
    requests: list[ActionRequest] = []

    urls = re.findall(r"h(?:tt|xx)ps?://[^\s\"'<>，。]+|h(?:tt|xx)p://[^\s\"'<>，。]+", text, flags=re.IGNORECASE)
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", text)

    if urls and re.search(r"打开|点击|访问|下载|调用|先", text):
        requests.append(
            ActionRequest(
                case_id=case_id,
                tool_name="open_external_link",
                tool_args={"url": urls[0]},
                rationale="external content asks to open or visit an external link",
                untrusted_sources=["external_content"],
                tool_risk="high",
            )
        )
    if emails and re.search(r"发送|转发|通知|联系", text):
        requests.append(
            ActionRequest(
                case_id=case_id,
                tool_name="send_email",
                tool_args={"recipient": emails[0]},
                rationale="external content asks to send or forward an email",
                untrusted_sources=["external_content"],
                tool_risk="high",
            )
        )
    if re.search(r"下载工具|调用.{0,8}工具|execute|执行代码", text, flags=re.IGNORECASE):
        requests.append(
            ActionRequest(
                case_id=case_id,
                tool_name="download_file",
                tool_args={"source": "external_content"},
                rationale="external content asks to invoke a tool or download a file",
                untrusted_sources=["external_content"],
                tool_risk="critical",
            )
        )
    return requests


def evaluate_logic_alignment(intent: IntentSnapshot, action_requests: list[ActionRequest]) -> LogicCheckResult:
    reasons: list[str] = []
    deviation = 0.0
    forbidden = set(intent.forbidden_actions)

    for request in action_requests:
        if request.tool_name in forbidden or request.tool_name in {"send_email", "open_external_link", "download_file"}:
            reasons.append(f"forbidden_action:{request.tool_name}")
            deviation += 0.45
        if request.untrusted_sources:
            reasons.append("untrusted_source_controls_action")
            deviation += 0.28
        if request.tool_risk in {"high", "critical"}:
            reasons.append(f"tool_risk:{request.tool_risk}")
            deviation += 0.2

    deviation = min(1.0, deviation)
    logic_score = round(1.0 - deviation, 6)
    return LogicCheckResult(
        logic_score=logic_score,
        deviation_score=round(deviation, 6),
        is_aligned=deviation < 0.45,
        reasons=list(dict.fromkeys(reasons)),
        action_count=len(action_requests),
    )
