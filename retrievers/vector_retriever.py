"""Vector retrieval."""
from __future__ import annotations

from typing import Any

from config.settings import get_settings
from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_store import VectorStore


class VectorRetriever:

    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Any | None = None,
    ):
        self.settings = get_settings()
        self.store = store
        self.embedder = embedder

    def _embedder(self):
        if self.embedder is None:
            self.embedder = EmbeddingModel(
                self.settings.embedding_model
            )

        return self.embedder

    def _store(self):
        if self.store is None:
            self.store = VectorStore(
                self.settings
            )

        return self.store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ):
        query = (
            query or ""
        ).strip()

        if not query:
            return []

        vector = (
            self._embedder()
            .embed_query(query)
        )

        return self._store().search(
            query_vector=vector,
            top_k=(
                top_k
                or self.settings.retrieval_top_k
            ),
            threshold=(
                self.settings.similarity_threshold
                if threshold is None
                else threshold
            ),
            filters=filters,
        )
