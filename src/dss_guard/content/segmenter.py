"""Sliding-window segmentation for untrusted external content."""

from __future__ import annotations

from dataclasses import dataclass

from src.dss_guard.schemas import TextWindow


@dataclass(frozen=True)
class SegmentConfig:
    window_size: int = 384
    overlap_ratio: float = 0.5
    mode: str = "char"
    max_windows: int = 64


def _step_size(window_size: int, overlap_ratio: float) -> int:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")
    return max(1, int(window_size * (1 - overlap_ratio)))


def segment_text(
    text: str,
    case_id: str,
    source_id: str = "external_content",
    config: SegmentConfig | None = None,
) -> list[TextWindow]:
    """Split text into overlapping windows while preserving character spans.

    The current tokenizer mode is intentionally approximate. For Chinese-heavy
    web snippets, character windows give stable span mapping and avoid adding a
    tokenizer dependency to P3.
    """

    cfg = config or SegmentConfig()
    content = text or ""
    if not content:
        return [
            TextWindow(
                case_id=case_id,
                window_id=f"{case_id}:w0000",
                source_id=source_id,
                start=0,
                end=0,
                text="",
                metadata={"mode": cfg.mode, "window_index": 0},
            )
        ]

    step = _step_size(cfg.window_size, cfg.overlap_ratio)
    windows: list[TextWindow] = []
    start = 0
    while start < len(content) and len(windows) < cfg.max_windows:
        end = min(len(content), start + cfg.window_size)
        windows.append(
            TextWindow(
                case_id=case_id,
                window_id=f"{case_id}:w{len(windows):04d}",
                source_id=source_id,
                start=start,
                end=end,
                text=content[start:end],
                metadata={
                    "mode": cfg.mode,
                    "window_index": len(windows),
                    "window_size": cfg.window_size,
                    "overlap_ratio": cfg.overlap_ratio,
                },
            )
        )
        if end == len(content):
            break
        start += step
    return windows
