"""DSS 实验运行配置辅助工具。

密钥只从环境变量读取。真实值应放在本地 `.env` 文件或终端环境变量中，
仓库中只提交 `.env.example`。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    base_url: str
    model: str
    num_trials: int
    delay_between_trials: float


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值为 {raw_value!r}") from exc


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字，当前值为 {raw_value!r}") from exc


def get_llm_settings(require_api_key: bool = True) -> LLMSettings:
    settings = LLMSettings(
        api_key=os.getenv("DSS_LLM_API_KEY"),
        base_url=os.getenv("DSS_LLM_BASE_URL", DEFAULT_BASE_URL),
        model=os.getenv("DSS_LLM_MODEL", DEFAULT_MODEL),
        num_trials=_read_int("DSS_NUM_TRIALS", 3),
        delay_between_trials=_read_float("DSS_DELAY_BETWEEN_TRIALS", 1.0),
    )
    if require_api_key and not settings.api_key:
        raise RuntimeError("请先设置环境变量 DSS_LLM_API_KEY。可参考 .env.example。")
    return settings


def create_openai_client():
    settings = get_llm_settings(require_api_key=True)
    from openai import OpenAI

    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)
