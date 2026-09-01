"""Streamlit sidebar controls for the Graph RAG application."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from config.models import AVAILABLE_MODELS
from config.settings import Settings


@dataclass(frozen=True)
class SidebarOptions:
    """Runtime options selected by the user."""

    model: str
    temperature: float
    top_k: int
    similarity_threshold: float
    use_mmr: bool
    relation_filter: list[str]


def render_sidebar(settings: Settings, relation_types: list[str] | None = None) -> SidebarOptions:
    """Render sidebar and return user-selected runtime options."""
    st.sidebar.title("⚙️ Graph RAG Controls")
    model = st.sidebar.selectbox(
        "OpenRouter model",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(settings.llm_model) if settings.llm_model in AVAILABLE_MODELS else 0,
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, float(settings.llm_temperature), 0.05)
    top_k = st.sidebar.slider("Top-k retrieval", 1, 20, int(settings.retrieval_top_k), 1)
    similarity_threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, float(settings.similarity_threshold), 0.01)
    use_mmr = st.sidebar.toggle("Use MMR diversification", value=True)
    relation_filter = st.sidebar.multiselect("Filter relation types", options=relation_types or [])
    st.sidebar.divider()
    st.sidebar.caption("Neo4j")
    st.sidebar.code(settings.neo4j_uri, language="text")
    st.sidebar.caption("Embeddings")
    st.sidebar.code(settings.embedding_model, language="text")
    return SidebarOptions(model, temperature, top_k, similarity_threshold, use_mmr, relation_filter)