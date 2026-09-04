"""Centralized runtime configuration for local and Streamlit deployments."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Production Graph RAG"

    # OpenRouter
    openrouter_api_key: str | None = Field(
        default=None,
        alias="OPENROUTER_API_KEY",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    llm_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="LLM_MODEL",
    )
    llm_temperature: float = Field(
        default=0.1,
        alias="LLM_TEMPERATURE",
    )
    max_tokens: int = Field(
        default=700,
        alias="MAX_TOKENS",
    )

    # Embeddings / retrieval
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    reranker_model: str | None = Field(
        default=None,
        alias="RERANKER_MODEL",
    )
    chunk_size: int = Field(
        default=800,
        alias="CHUNK_SIZE",
    )
    chunk_overlap: int = Field(
        default=120,
        alias="CHUNK_OVERLAP",
    )
    retrieval_top_k: int = Field(
        default=8,
        alias="TOP_K",
    )
    similarity_threshold: float = Field(
        default=0.0,
        alias="SIMILARITY_THRESHOLD",
    )
    hybrid_alpha: float = Field(
        default=0.6,
        alias="HYBRID_ALPHA",
    )
    enable_async: bool = Field(
        default=True,
        alias="ENABLE_ASYNC",
    )

    # Neo4j
    neo4j_uri: str | None = Field(
        default=None,
        alias="NEO4J_URI",
    )
    neo4j_username: str | None = Field(
        default=None,
        alias="NEO4J_USERNAME",
    )
    neo4j_password: str | None = Field(
        default=None,
        alias="NEO4J_PASSWORD",
    )
    neo4j_database: str = Field(
        default="neo4j",
        alias="NEO4J_DATABASE",
    )

    # Storage
    faiss_index_path: Path = Field(
        default=Path("data/processed/faiss.index"),
        alias="FAISS_INDEX_PATH",
    )
    faiss_docstore_path: Path = Field(
        default=Path("data/processed/docstore.json"),
        alias="FAISS_DOCSTORE_PATH",
    )

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")

    request_timeout: int = 90

    def _streamlit_secret(self, name: str) -> str | None:
        """Read a single value from Streamlit secrets when running under Streamlit."""
        try:
            import streamlit as st

            value = st.secrets.get(name)
            if value and str(value).strip():
                return str(value).strip()
        except Exception:
            pass
        return None

    def resolve_openrouter_key(self) -> tuple[str, str]:
        """Return ``(api_key, source)`` for the OpenRouter API key.

        Resolution order (highest precedence first):

        1. Streamlit secrets (``st.secrets["OPENROUTER_API_KEY"]``) — the
           intended mechanism on Streamlit Cloud. This deliberately wins over a
           committed ``.env`` so an outdated key baked into the repo can never
           shadow the secret configured in the Streamlit Cloud dashboard.
        2. A real OS environment variable (``OPENROUTER_API_KEY``) — used when
           deploying locally or on other hosts that inject env vars.
        3. The value loaded from the ``.env`` file by pydantic-settings.

        The returned ``source`` is one of ``"streamlit"``, ``"env"``,
        ``"env_file"`` and is used for diagnostics so it is obvious which key
        is active.
        """

        secret = self._streamlit_secret("OPENROUTER_API_KEY")
        if secret:
            return secret, "streamlit"

        env_value = os.environ.get("OPENROUTER_API_KEY")
        if env_value and env_value.strip():
            # If os.environ mirrors the .env file pydantic already loaded,
            # attribute it to the env file; otherwise it is a real env var.
            source = "env"
            if (
                self.openrouter_api_key
                and env_value.strip() == self.openrouter_api_key.strip()
            ):
                source = "env_file"
            return env_value.strip(), source

        if self.openrouter_api_key and self.openrouter_api_key.strip():
            return self.openrouter_api_key.strip(), "env_file"

        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to Streamlit Secrets "
            "(App settings -> Secrets) or to a local .env file."
        )

    def require_openrouter_key(self) -> str:
        """Return the resolved OpenRouter API key (raises if none is configured)."""
        return self.resolve_openrouter_key()[0]

    def verify_openrouter_key(
        self,
        timeout: float = 10.0,
    ) -> dict[str, object]:
        """Validate the configured OpenRouter key against the /key endpoint.

        Returns a dict with:
            status:  "ok" | "missing" | "invalid" | "error"
            source:  where the key came from
                     ("streamlit"/"env"/"env_file")
            message: human-readable description
            detail:  extra info from OpenRouter (e.g. limit/usage) when present
        """

        try:
            api_key, source = self.resolve_openrouter_key()
        except Exception as exc:
            return {
                "status": "missing",
                "source": None,
                "message": str(exc),
                "detail": None,
            }

        url = f"{self.openrouter_base_url.rstrip('/')}/key"
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(
                    response.read().decode("utf-8", errors="replace")
                )

            detail = payload.get("data", payload)
            return {
                "status": "ok",
                "source": source,
                "message": "OpenRouter API key is valid.",
                "detail": detail,
            }

        except urllib.error.HTTPError as exc:
            # 401 -> bad/unknown key ("User not found"), 402 -> out of credit,
            # 429 -> rate limited.
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""

            if exc.code in (401, 403):
                message = (
                    "OpenRouter rejected the API key (HTTP "
                    f"{exc.code}). The key is invalid, revoked, or copied "
                    "incorrectly. Create a fresh key at "
                    "https://openrouter.ai/keys and update it in Streamlit "
                    "Secrets (key name: OPENROUTER_API_KEY)."
                )
                status = "invalid"
            elif exc.code == 402:
                message = (
                    "OpenRouter reports the account is out of credits "
                    "(HTTP 402). Add credits at "
                    "https://openrouter.ai/credits."
                )
                status = "error"
            else:
                message = f"OpenRouter returned HTTP {exc.code}."
                status = "error"

            return {
                "status": status,
                "source": source,
                "message": message,
                "detail": body[:500] or None,
            }

        except urllib.error.URLError as exc:
            return {
                "status": "error",
                "source": source,
                "message": (
                    f"Could not reach OpenRouter to verify the key: {exc.reason}. "
                    "Check the network/proxy configuration."
                ),
                "detail": None,
            }

    def resolve_neo4j(
        self,
        key: str,
        default: str | None = None,
    ) -> str | None:
        """Resolve Neo4j configuration from settings or Streamlit secrets."""

        mapping = {
            "NEO4J_URI": "neo4j_uri",
            "NEO4J_USERNAME": "neo4j_username",
            "NEO4J_PASSWORD": "neo4j_password",
            "NEO4J_DATABASE": "neo4j_database",
        }

        attr = mapping[key]
        value = getattr(self, attr, None)

        if value:
            return str(value).strip()

        secret = self._streamlit_secret(key)
        if secret:
            return secret

        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
