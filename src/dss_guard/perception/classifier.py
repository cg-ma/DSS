"""Lightweight window classifier with optional local model configuration."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.dss_guard.perception.ppl_filter import compute_anomaly
from src.dss_guard.perception.rules import scan_rules
from src.dss_guard.schemas import TextWindow

DEFAULT_DISTILBERT_MODEL_PATH = os.environ.get("DSS_DISTILBERT_MODEL_PATH", r"E:\distilbert-base-multilingual-cased")


@dataclass(frozen=True)
class ClassifierResult:
    classifier_score: float
    backend: str
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


class LightweightWindowClassifier:
    """Pure-Python classifier used when no fine-tuned local model is available."""

    def __init__(self, model_path: str | Path | None = None, prefer_model: bool = True) -> None:
        self.model_path = Path(model_path or DEFAULT_DISTILBERT_MODEL_PATH)
        self.prefer_model = prefer_model
        self.backend = "feature_fallback"
        self.model_available = False
        self._try_enable_model_backend()

    def _try_enable_model_backend(self) -> None:
        if not self.prefer_model:
            return
        if not str(self.model_path) or not self.model_path.exists():
            return
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return
        # The local DistilBERT base model is not necessarily fine-tuned for PII.
        # Keep deterministic feature fallback until a sequence-classification
        # checkpoint is available.
        self.model_available = True
        self.backend = "distilbert_configured_feature_fallback"

    def score_window(self, window: TextWindow) -> ClassifierResult:
        text = window.text or ""
        rules = scan_rules(window)
        anomaly = compute_anomaly(text)
        lower_text = text.lower()
        features = {
            "rule_score": rules.rule_score,
            "hidden_text_score": rules.hidden_text_score,
            "ppl_anomaly_score": anomaly.ppl_anomaly_score,
            "url_count": min(3, len(re.findall(r"h(?:tt|xx)ps?://|h(?:tt|xx)p://", lower_text))) / 3,
            "email_count": min(3, len(re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", text))) / 3,
            "imperative_count": min(5, len(re.findall(r"请|必须|直接|输出|打开|访问|发送|ignore|override", lower_text))) / 5,
            "boundary_count": min(4, len(re.findall(r"system|tool|developer|override|系统|最高优先级", lower_text))) / 4,
            "secret_count": min(4, len(re.findall(r"api\s*key|token|密钥|凭据|联系人|本地文件|环境变量", lower_text))) / 4,
        }
        linear = (
            features["rule_score"] * 2.4
            + features["hidden_text_score"] * 0.65
            + features["ppl_anomaly_score"] * 1.1
            + features["url_count"] * 0.8
            + features["email_count"] * 0.55
            + features["imperative_count"] * 0.65
            + features["boundary_count"] * 0.75
            + features["secret_count"] * 0.6
            - 1.15
        )
        score = 1 / (1 + math.exp(-linear))
        reasons = [f"classifier:{name}" for name, value in features.items() if value >= 0.45]
        return ClassifierResult(
            classifier_score=round(score, 6),
            backend=self.backend,
            features=features,
            reasons=reasons,
        )

    def score_windows(self, windows: list[TextWindow]) -> list[ClassifierResult]:
        return [self.score_window(window) for window in windows]


def build_classifier(model_path: str | Path | None = None, prefer_model: bool = True) -> LightweightWindowClassifier:
    return LightweightWindowClassifier(model_path=model_path, prefer_model=prefer_model)
