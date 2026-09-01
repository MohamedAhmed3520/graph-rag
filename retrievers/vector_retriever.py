"""Vector retriever backed by FAISS."""
from __future__ import annotations

from typing import Any
from embeddings.vector_store import VectorStore
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """Retrieve semantically similar chunks with optional metadata filtering."""

    def __init__(self, store: VectorStore | None = None, embedder: Any | None = None) -> None:
        self.settings = get_settings()
        self.embedder = embedder
        self.store = store

    def _embedder(self) -> Any:
        if self.embedder is None:
            from embeddings.embedding_model import EmbeddingModel

            self.embedder = EmbeddingModel()
        return self.embedder

    def _store(self) -> VectorStore:
        if self.store is None:
            self.store = VectorStore(self.settings)
        return self.store

    def retrieve(self, query: str, top_k: int | None = None, threshold: float | None = None, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            vector = self._embedder().embed_query(query)
            return self._store().search(vector, top_k or self.settings.retrieval_top_k, threshold if threshold is not None else self.settings.similarity_threshold, filters)
        except Exception as exc:
            logger.warning("Vector retrieval unavailable: %s", exc)
            return []
