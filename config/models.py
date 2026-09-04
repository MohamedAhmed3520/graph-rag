from __future__ import annotations

from typing import Any

from config.settings import Settings, get_settings


AVAILABLE_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct:free",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5",
]


class OpenRouterAuthError(RuntimeError):
    """Raised when the OpenRouter API key is missing or rejected (HTTP 401/403)."""


# Markers OpenRouter / OpenAI-compatible clients include in auth failures.
_AUTH_MARKERS: tuple[str, ...] = (
    "unauthorized",
    "user not found",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "no api key",
    "missing api key",
    "authentication",
)


def is_openrouter_auth_error(exc: BaseException) -> bool:
    """Return True when ``exc`` is an OpenRouter authentication failure.

    Detects both the dedicated exception classes raised by the openrouter /
    langchain-openrouter clients and the textual signature OpenRouter returns
    for a bad key (HTTP 401 ``"User not found"``).
    """

    class_name = type(exc).__name__.lower()
    if "unauthorized" in class_name or "authentication" in class_name:
        return True

    text = str(exc).lower()
    return any(marker in text for marker in _AUTH_MARKERS)


def create_llm(
    settings: Settings | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Create an OpenRouter-backed LangChain chat model.

    Raises:
        OpenRouterAuthError: if no usable API key is configured.
    """

    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'langchain-openrouter'. "
            "Run: pip install -U langchain-openrouter"
        ) from exc

    cfg = settings or get_settings()

    try:
        api_key = cfg.require_openrouter_key()
    except Exception as exc:
        raise OpenRouterAuthError(str(exc)) from exc

    return ChatOpenRouter(
        model=model or cfg.llm_model,
        temperature=(
            cfg.llm_temperature
            if temperature is None
            else temperature
        ),
        api_key=api_key,
        max_tokens=700,
        max_retries=2,
    )
