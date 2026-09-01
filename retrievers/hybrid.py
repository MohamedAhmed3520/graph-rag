"""Hybrid retrieval combining vector evidence and graph traversal."""
from __future__ import annotations

from typing import Any
from config.settings import get_settings
from retrievers.vector_retriever import VectorRetriever
from retrievers.graph_retriever import GraphRetriever
from retrievers.reranker import SimpleReranker


class HybridRetriever:
    """Retrieve, merge, and format graph plus vector contexts."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector: VectorRetriever | None = None
        self.graph: GraphRetriever | None = None
        self.reranker = SimpleReranker()

    def _vector(self) -> VectorRetriever:
        if self.vector is None:
            self.vector = VectorRetriever()
        return self.vector

    def _graph(self) -> GraphRetriever:
        if self.graph is None:
            self.graph = GraphRetriever()
        return self.graph

    def retrieve(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        k = top_k or self.settings.retrieval_top_k
        vector_hits = self.reranker.rerank(self._vector().retrieve(query, k, filters=filters), k)
        graph_hits = self._graph().retrieve(query, top_k=k)
        merged = self.format_context(vector_hits, graph_hits)
        return {"vector_context": vector_hits, "graph_context": graph_hits, "merged_context": merged}

    @staticmethod
    def format_context(vector_hits: list[dict[str, Any]], graph_hits: list[dict[str, Any]]) -> str:
        vector_text = "\n".join(f"[Chunk {i+1}] {h.get('text','')} SOURCE={h.get('metadata',{}).get('source','unknown')}" for i, h in enumerate(vector_hits))
        graph_text = "\n".join(f"({h.get('subject')}) -[{h.get('relation')}]-> ({h.get('object')}) confidence={h.get('confidence', '')}" for h in graph_hits)
        return f"<vector_context>\n{vector_text}\n</vector_context>\n<graph_context>\n{graph_text}\n</graph_context>"
