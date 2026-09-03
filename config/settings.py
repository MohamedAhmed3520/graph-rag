"""Centralized runtime configuration for local and Streamlit deployments."""
from __future__ import annotations

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

    def require_openrouter_key(self) -> str:
        """Return OpenRouter key from env/.env or Streamlit secrets."""
        if self.openrouter_api_key and self.openrouter_api_key.strip():
            return self.openrouter_api_key.strip()

        try:
            import streamlit as st

            secret = st.secrets.get("OPENROUTER_API_KEY")
            if secret:
                return str(secret).strip()
        except Exception:
            pass

        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. "
            "Configure it in Streamlit Secrets or .env."
        )

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

        try:
            import streamlit as st

            secret = st.secrets.get(key)
            if secret:
                return str(secret).strip()
        except Exception:
            pass

        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
