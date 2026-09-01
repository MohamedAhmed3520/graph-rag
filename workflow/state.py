"""Typed LangGraph workflow state."""
from __future__ import annotations

from typing import Any, TypedDict


class GraphRAGState(TypedDict, total=False):
    """Shared state for ingestion and question-answering workflows."""

    file_paths: list[str]
    documents: list[Any]
    chunks: list[Any]
    graph_documents: list[Any]
    question: str
    rewritten_question: str
    vector_context: list[dict[str, Any]]
    graph_context: list[dict[str, Any]]
    merged_context: str
    grade: str
    answer: str
    citations: list[dict[str, Any]]
    errors: list[str]
    status: str
