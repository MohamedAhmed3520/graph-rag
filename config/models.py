from __future__ import annotations

from typing import Any

from config.settings import Settings, get_settings


AVAILABLE_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct:free",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5",
]


def create_llm(
    settings: Settings | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Create an OpenRouter-backed LangChain chat model."""

    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'langchain-openrouter'. "
            "Run: pip install -U langchain-openrouter"
        ) from exc

    cfg = settings or get_settings()

    return ChatOpenRouter(
        model=model or cfg.llm_model,
        temperature=(
            cfg.llm_temperature
            if temperature is None
            else temperature
        ),
        api_key=cfg.require_openrouter_key(),
        max_tokens=700,
        max_retries=2,
    )
