"""Reusable Streamlit components."""
from __future__ import annotations

from typing import Any

import streamlit as st


def metric_row(graph_stats: dict[str, Any], vector_stats: dict[str, Any]) -> None:
    """Render graph and vector statistics."""
    cols = st.columns(4)
    cols[0].metric("Graph nodes", graph_stats.get("nodes", 0))
    cols[1].metric("Graph edges", graph_stats.get("relationships", 0))
    cols[2].metric("Vector chunks", vector_stats.get("documents", 0))
    cols[3].metric("Embedding dim", vector_stats.get("dimension", "—"))


def render_context(vector_context: list[dict[str, Any]], graph_context: list[dict[str, Any]]) -> None:
    """Render retrieved vector and graph contexts in expanders."""
    with st.expander("📚 Retrieved vector context", expanded=False):
        if not vector_context:
            st.info("No vector context retrieved yet.")
        for item in vector_context:
            st.markdown(f"**Score:** `{item.get('score', 'n/a')}` | **Source:** `{item.get('source', 'unknown')}`")
            st.write(item.get("content", ""))
            st.divider()
    with st.expander("🕸️ Graph traversal results", expanded=False):
        if not graph_context:
            st.info("No graph context retrieved yet.")
        for item in graph_context:
            st.json(item)


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render source citations."""
    if not citations:
        return
    with st.expander("🔖 Citations", expanded=True):
        for idx, citation in enumerate(citations, start=1):
            st.markdown(f"{idx}. `{citation.get('source', 'unknown')}` — chunk `{citation.get('chunk_id', 'n/a')}`")


def workflow_status(status: str, errors: list[str] | None = None) -> None:
    """Render current workflow status and errors."""
    st.status(status or "Idle", state="error" if errors else "complete")
    if errors:
        for error in errors:
            st.error(error)