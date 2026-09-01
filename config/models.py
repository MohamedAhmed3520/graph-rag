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
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'langchain-openai'."
        ) from exc

    cfg = settings or get_settings()

    llm = ChatOpenAI(
        model=model or cfg.llm_model,
        temperature=(
            cfg.llm_temperature
            if temperature is None
            else temperature
        ),
        api_key=cfg.require_openrouter_key(),
        base_url=cfg.openrouter_base_url,
        timeout=cfg.request_timeout,
        max_retries=2,
        default_headers={
            "HTTP-Referer": "https://your-streamlit-app.streamlit.app",
            "X-Title": cfg.app_name,
        },
    )

    return llm
