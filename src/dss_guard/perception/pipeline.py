"""End-to-end P3 perception scanning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dss_guard.content.segmenter import SegmentConfig, segment_text
from src.dss_guard.perception.classifier import DEFAULT_DISTILBERT_MODEL_PATH, build_classifier
from src.dss_guard.perception.ppl_filter import compute_anomaly
from src.dss_guard.perception.risk_fusion import PerceptionWindowResult, fuse_scores, fuse_window_results
from src.dss_guard.perception.rules import scan_rules
from src.dss_guard.schemas import RiskSignal, TextWindow


@dataclass(frozen=True)
class PerceptionConfig:
    window_size: int = 384
    overlap_ratio: float = 0.5
    max_windows: int = 64
    attack_threshold: float = 0.35
    classifier_model_path: str = DEFAULT_DISTILBERT_MODEL_PATH
    prefer_model: bool = True


def _absolute_spans(window: TextWindow, local_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(window.start + start, window.start + end) for start, end in local_spans]


def scan_content(
    text: str,
    case_id: str,
    config: PerceptionConfig | None = None,
) -> tuple[RiskSignal, list[PerceptionWindowResult]]:
    cfg = config or PerceptionConfig()
    segment_config = SegmentConfig(
        window_size=cfg.window_size,
        overlap_ratio=cfg.overlap_ratio,
        max_windows=cfg.max_windows,
    )
    windows = segment_text(text=text, case_id=case_id, config=segment_config)
    classifier = build_classifier(model_path=Path(cfg.classifier_model_path), prefer_model=cfg.prefer_model)

    window_results: list[PerceptionWindowResult] = []
    for window in windows:
        rule_result = scan_rules(window)
        anomaly_result = compute_anomaly(window.text)
        classifier_result = classifier.score_window(window)
        fused = fuse_scores(
            classifier_score=classifier_result.classifier_score,
            rule_score=rule_result.rule_score,
            ppl_anomaly_score=anomaly_result.ppl_anomaly_score,
            hidden_text_score=rule_result.hidden_text_score,
        )
        local_spans = [(match.start, match.end) for match in rule_result.matches]
        reasons = rule_result.reasons + anomaly_result.reasons + classifier_result.reasons
        features = {
            "classifier_score": classifier_result.classifier_score,
            "rule_score": rule_result.rule_score,
            "hidden_text_score": rule_result.hidden_text_score,
            "ppl_anomaly_score": anomaly_result.ppl_anomaly_score,
            "classifier_backend": classifier_result.backend,
            **classifier_result.features,
            **anomaly_result.features,
        }
        window_results.append(
            PerceptionWindowResult(
                window=window,
                rule_score=rule_result.rule_score,
                hidden_text_score=rule_result.hidden_text_score,
                ppl_anomaly_score=anomaly_result.ppl_anomaly_score,
                classifier_score=classifier_result.classifier_score,
                fused_score=fused,
                reasons=reasons,
                matched_spans=_absolute_spans(window, local_spans),
                features=features,
            )
        )

    return fuse_window_results(case_id=case_id, window_results=window_results), window_results


def scan_case(case: dict[str, Any], config: PerceptionConfig | None = None) -> tuple[RiskSignal, list[PerceptionWindowResult]]:
    return scan_content(
        text=str(case.get("external_content", "")),
        case_id=str(case.get("case_id", "")),
        config=config,
    )
