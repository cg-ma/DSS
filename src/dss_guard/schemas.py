"""DSS 防御流水线共享数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RiskLevel = Literal["low", "medium", "high", "critical"]
VerdictAction = Literal["allow", "sanitize", "audit", "block"]
ToolRisk = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class ExternalContent:
    """从网页、文档或搜索结果中检索到的不可信外部内容。"""

    case_id: str
    source_id: str
    source_type: str
    raw_text: str
    clean_text: str
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextWindow:
    """由重叠滑动窗口切片得到的外部内容片段。"""

    case_id: str
    window_id: str
    source_id: str
    start: int
    end: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskSignal:
    """敏捷感知层输出的风险评分。"""

    case_id: str
    risk_level: RiskLevel
    score: float
    matched_spans: list[tuple[int, int]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    source_id: str | None = None


@dataclass(frozen=True)
class IntentSnapshot:
    """用户原始目标和行为边界的结构化表示。"""

    case_id: str
    user_goal: str
    allowed_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    sensitive_slots: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionRequest:
    """智能体提出的候选工具调用或候选动作。"""

    case_id: str
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    untrusted_sources: list[str] = field(default_factory=list)
    tool_risk: ToolRisk = "low"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DefenseVerdict:
    """防御层输出的最终或中间裁决。"""

    case_id: str
    action: VerdictAction
    risk_score: float
    reasons: list[str] = field(default_factory=list)
    stage: str = "unknown"
    sanitized_text: str | None = None
    audit_depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
