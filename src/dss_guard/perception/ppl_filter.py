"""Statistical anomaly fallback for the PPL filter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnomalyResult:
    ppl_anomaly_score: float
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _ratio(count: int, total: int) -> float:
    return count / total if total else 0.0


def _repeat_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        normalized = token.lower()
        counts[normalized] = counts.get(normalized, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(tokens)


def _mixed_script_score(text: str) -> float:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_symbols = bool(re.search(r"[_<>/\\{}\[\]|`~]", text))
    return min(1.0, (0.35 if has_cjk and has_latin else 0.0) + (0.35 if has_symbols else 0.0))


def compute_anomaly(text: str) -> AnomalyResult:
    content = text or ""
    total = len(content)
    if total == 0:
        return AnomalyResult(ppl_anomaly_score=0.0, features={}, reasons=[])

    non_natural = sum(1 for char in content if not (char.isalnum() or char.isspace() or "\u4e00" <= char <= "\u9fff"))
    delimiter_chars = sum(1 for char in content if char in "_-=/\\|<>[]{}:;#*`~")
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+|[^\s]", content)
    spaced_cjk = len(re.findall(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", content))
    obfuscated = len(
        re.findall(
            r"i\s*[-_ ]?\s*g\s*[-_ ]?\s*n\s*[-_ ]?\s*o\s*[-_ ]?\s*r\s*[-_ ]?\s*e|忽\s*略|系\s*[- ]?\s*统\s*[- ]?\s*提\s*[- ]?\s*示",
            content,
            flags=re.IGNORECASE,
        )
    )

    features = {
        "non_natural_char_ratio": _ratio(non_natural, total),
        "delimiter_density": _ratio(delimiter_chars, total),
        "repeat_token_ratio": _repeat_token_ratio(tokens),
        "mixed_script_score": _mixed_script_score(content),
        "spaced_cjk_density": min(1.0, spaced_cjk / 8),
        "obfuscated_keyword_score": min(1.0, obfuscated * 0.45),
    }

    score = (
        features["non_natural_char_ratio"] * 0.35
        + features["delimiter_density"] * 1.4
        + features["repeat_token_ratio"] * 0.35
        + features["mixed_script_score"] * 0.28
        + features["spaced_cjk_density"] * 0.32
        + features["obfuscated_keyword_score"]
    )
    score = min(1.0, score)

    reasons: list[str] = []
    if features["obfuscated_keyword_score"] >= 0.45:
        reasons.append("obfuscated_keyword")
    if features["delimiter_density"] >= 0.08:
        reasons.append("high_delimiter_density")
    if features["spaced_cjk_density"] >= 0.25:
        reasons.append("spaced_cjk_tokens")
    if features["mixed_script_score"] >= 0.35:
        reasons.append("mixed_script")

    return AnomalyResult(ppl_anomaly_score=round(score, 6), features=features, reasons=reasons)
