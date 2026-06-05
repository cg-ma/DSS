"""Risk fusion for P3 perception signals."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.dss_guard.schemas import RiskSignal, TextWindow


@dataclass(frozen=True)
class PerceptionWindowResult:
    window: TextWindow
    rule_score: float
    hidden_text_score: float
    ppl_anomaly_score: float
    classifier_score: float
    fused_score: float
    reasons: list[str] = field(default_factory=list)
    matched_spans: list[tuple[int, int]] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)


def risk_level(score: float) -> str:
    if score >= 0.82:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def fuse_scores(
    classifier_score: float,
    rule_score: float,
    ppl_anomaly_score: float,
    hidden_text_score: float,
) -> float:
    weighted = (
        classifier_score * 0.42
        + rule_score * 0.36
        + ppl_anomaly_score * 0.16
        + hidden_text_score * 0.06
    )
    max_signal = max(classifier_score, rule_score, ppl_anomaly_score, hidden_text_score)
    if rule_score >= 0.7 or hidden_text_score >= 0.7:
        weighted = max(weighted, 0.72)
    if ppl_anomaly_score >= 0.55 and classifier_score >= 0.45:
        weighted = max(weighted, 0.62)
    return round(min(1.0, max(weighted, max_signal * 0.88)), 6)


def fuse_window_results(case_id: str, window_results: list[PerceptionWindowResult]) -> RiskSignal:
    if not window_results:
        return RiskSignal(case_id=case_id, risk_level="low", score=0.0)

    top_result = max(window_results, key=lambda result: result.fused_score)
    matched_spans: list[tuple[int, int]] = []
    reasons: list[str] = []
    features = {
        "max_classifier_score": 0.0,
        "max_rule_score": 0.0,
        "max_hidden_text_score": 0.0,
        "max_ppl_anomaly_score": 0.0,
        "windows_count": float(len(window_results)),
    }

    for result in window_results:
        features["max_classifier_score"] = max(features["max_classifier_score"], result.classifier_score)
        features["max_rule_score"] = max(features["max_rule_score"], result.rule_score)
        features["max_hidden_text_score"] = max(features["max_hidden_text_score"], result.hidden_text_score)
        features["max_ppl_anomaly_score"] = max(features["max_ppl_anomaly_score"], result.ppl_anomaly_score)
        for reason in result.reasons:
            if reason not in reasons:
                reasons.append(reason)
        if result.fused_score >= 0.35:
            matched_spans.extend(result.matched_spans or [(result.window.start, result.window.end)])

    return RiskSignal(
        case_id=case_id,
        risk_level=risk_level(top_result.fused_score),
        score=top_result.fused_score,
        matched_spans=matched_spans,
        reasons=reasons,
        features=features,
        source_id="external_content",
    )
