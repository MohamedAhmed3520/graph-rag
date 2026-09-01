"""Embedding model factory and compatibility wrapper."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from config.settings import get_settings


class SentenceTransformerEmbeddings:
    """Small LangChain-compatible wrapper around Sentence Transformers."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "sentence-transformers is required for embeddings. Install requirements.txt."
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts with normalized vectors for inner-product FAISS search."""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with normalized vectors."""
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str | None = None) -> Any:
    """Return a cached embedding model.

    The factory first uses ``langchain_huggingface`` when installed, then falls
    back to a direct Sentence Transformers wrapper. This avoids hard startup
    failures in environments where the optional LangChain integration package is
    not installed but ``sentence-transformers`` is available.
    """
    settings = get_settings()
    resolved_model = model_name or settings.embedding_model
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None  # type: ignore[assignment]

    try:
        if HuggingFaceEmbeddings is not None:
            return HuggingFaceEmbeddings(
                model_name=resolved_model,
                encode_kwargs={"normalize_embeddings": True},
            )
        return SentenceTransformerEmbeddings(resolved_model)
    except Exception as exc:  # pragma: no cover - model/network/runtime dependent
        raise RuntimeError(f"Failed to load embedding model '{resolved_model}': {exc}") from exc


class EmbeddingModel:
    """Thin compatibility wrapper used across the workflow and vector store.

    The project originally expected a class-based embedding API. Keeping this
    wrapper avoids touching multiple consumers while still delegating the heavy
    lifting to the cached lazy factory above.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name
        self._model = get_embedding_model(model_name)

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        probe = self._model.embed_query("dimension probe")
        return len(probe)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents with normalized vectors."""
        return self._model.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._model.embed_query(text)

    @property
    def langchain(self) -> Any:
        """Return the underlying LangChain embeddings implementation."""
        return self._model
