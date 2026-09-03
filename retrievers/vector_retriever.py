"""Vector retrieval."""
from __future__ import annotations

from typing import Any

from config.settings import Settings, get_settings
from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_store import VectorStore


class VectorRetriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.embedder = embedder

    def _embedder(self) -> EmbeddingModel:
        if self.embedder is None:
            self.embedder = EmbeddingModel(
                self.settings.embedding_model
            )

        return self.embedder

    def _store(self) -> VectorStore:
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
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()

        if not query:
            return []

        embedding = self._embedder().embed_query(
            query
        )

        return self._store().search(
            query_vector=embedding,
            top_k=(
                top_k
                if top_k is not None
                else self.settings.retrieval_top_k
            ),
            threshold=(
                threshold
                if threshold is not None
                else self.settings.similarity_threshold
            ),
            filters=filters,
        )
