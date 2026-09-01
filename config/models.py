"""LLM factory for OpenRouter-compatible chat models.

The LangChain OpenAI integration is imported lazily so Streamlit can render
configuration/help screens even before optional dependencies are installed.
"""
from __future__ import annotations

from typing import Any

from config.settings import Settings, get_settings


AVAILABLE_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct:free",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5",
]


def create_llm(settings: Settings | None = None, model: str | None = None, temperature: float | None = None) -> Any:
    """Create an OpenRouter-backed LangChain chat model."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Missing dependency 'langchain-openai'. Install project dependencies with "
            "`uv pip install -r requirements.txt` or `pip install -r requirements.txt`."
        ) from exc

    cfg = settings or get_settings()
    return ChatOpenAI(
        model=model or cfg.llm_model,
        temperature=cfg.llm_temperature if temperature is None else temperature,
        api_key=cfg.require_openrouter_key(),
        base_url=cfg.openrouter_base_url,
        timeout=cfg.request_timeout,
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": cfg.app_name},
    )