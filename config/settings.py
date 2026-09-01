"""Application settings loaded from environment and .env."""
import os
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized runtime configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Production Graph RAG"
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    llm_model: str = Field(default="openai/gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 6
    mmr_lambda: float = 0.5
    similarity_threshold: float = 0.2
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    faiss_index_path: Path = Field(default=Path("data/processed/faiss.index"), alias="FAISS_INDEX_PATH")
    faiss_docstore_path: Path = Field(default=Path("data/processed/docstore.json"), alias="FAISS_DOCSTORE_PATH")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    request_timeout: int = 60

    def require_openrouter_key(self) -> str:
        """Load OPENROUTER_API_KEY from .env, environment, or Streamlit secrets."""
        # First check if already loaded from .env or environment
        if self.openrouter_api_key:
            return self.openrouter_api_key
        
        # Try Streamlit secrets (for Streamlit Cloud deployment)
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
                return st.secrets["OPENROUTER_API_KEY"]
        except (ImportError, Exception):
            pass
        
        # Fallback error message
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. "
            "Configure it in:\n"
            "  1. Local .env file (development)\n"
            "  2. Environment variable OPENROUTER_API_KEY\n"
            "  3. Streamlit Secrets (Streamlit Cloud)"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()