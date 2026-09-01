"""FAISS vector index management."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from config.settings import Settings
from embeddings.embedding_model import EmbeddingModel
from ingestion.splitter import Chunk
from utils.helpers import ensure_dirs, read_json, write_json


@dataclass(slots=True)
class VectorRecord:
    chunk_id: str
    source_id: str
    text: str
    metadata: dict[str, str]


class FaissVectorStore:
    """Persisted dense vector index plus document store."""

    def __init__(
        self,
        index_path: Path | Settings,
        docstore_path: Path | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        """Initialize the vector store.

        The production UI and workflow pass a ``Settings`` instance, while some
        lower-level tests/tools may pass explicit paths. Supporting both keeps
        the storage layer convenient without leaking path construction into UI
        code.
        """
        if isinstance(index_path, Settings):
            settings = index_path
            self.index_path = Path(settings.faiss_index_path)
            self.docstore_path = Path(settings.faiss_docstore_path)
            self.embedding_model_name = embedding_model_name or settings.embedding_model
        else:
            if docstore_path is None:
                raise ValueError("docstore_path is required when index_path is not a Settings instance.")
            self.index_path = Path(index_path)
            self.docstore_path = Path(docstore_path)
            self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.dimension: int | None = None
        self.index = self._load_index()
        self.docstore: dict[str, VectorRecord] = self._load_docstore()

    def _get_embedding_model(self) -> EmbeddingModel:
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel(self.embedding_model_name)
            self.dimension = self.embedding_model.dimension
        return self.embedding_model

    def _ensure_dimension(self) -> int:
        """Resolve and cache the embedding dimension."""
        if self.dimension is None:
            self._get_embedding_model()
        if self.dimension is None:
            raise RuntimeError("Embedding dimension could not be determined.")
        return self.dimension

    def _create_empty_index(self) -> faiss.Index:
        """Create an empty FAISS index using the configured embedding dimension."""
        return faiss.IndexFlatIP(self._ensure_dimension())

    def _load_index(self) -> faiss.Index:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        # Default to loading the embedding model lazily only when a new index
        # must be created. Existing persisted indexes can be loaded without it.
        return self._create_empty_index()

    def _load_docstore(self) -> dict[str, VectorRecord]:
        raw = read_json(self.docstore_path, default={})
        return {k: VectorRecord(**v) for k, v in raw.items()}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        model = self._get_embedding_model()
        texts = [chunk.text for chunk in chunks]
        embeddings = model.embed_documents(texts)
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.index is None or self.index.d != vectors.shape[1]:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.dimension = vectors.shape[1]
        self.index.add(vectors)
        for chunk in chunks:
            self.docstore[chunk.chunk_id] = VectorRecord(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
        self.persist()

    def persist(self) -> None:
        ensure_dirs(self.index_path.parent, self.docstore_path.parent)
        faiss.write_index(self.index, str(self.index_path))
        write_json(self.docstore_path, {k: vars(v) for k, v in self.docstore.items()})

    def clear(self) -> None:
        """Delete persisted vector artifacts and reset the in-memory store."""
        if self.index_path.exists():
            self.index_path.unlink()
        if self.docstore_path.exists():
            self.docstore_path.unlink()
        self.embedding_model = None
        self.dimension = None
        self.index = self._create_empty_index()
        self.docstore = {}

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 6,
        threshold: float = 0.2,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, str | float]]:
        """Search the index for similar vectors.
        
        Returns a list of dicts with 'text', 'metadata', etc.
        Handles empty index gracefully.
        """
        if not self.docstore or self.index is None or self.index.ntotal == 0:
            return []
        
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        distances, indices = self.index.search(vector, min(top_k, self.index.ntotal))
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or dist < threshold:
                continue
            
            # Find the record by index position
            doc_id = None
            for i, (k, v) in enumerate(self.docstore.items()):
                if i == idx:
                    doc_id = k
                    break
            
            if doc_id is None:
                continue
            
            record = self.docstore[doc_id]
            
            # Apply filters if provided
            if filters:
                skip = False
                for key, value in filters.items():
                    if record.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            results.append({
                "id": record.chunk_id,
                "text": record.text,
                "metadata": {
                    "source_id": record.source_id,
                    "chunk_id": record.chunk_id,
                    **record.metadata,
                },
                "distance": float(dist),
            })
        
        return results

    def stats(self) -> dict[str, int | str]:
        """Return vector index statistics for health panels."""
        return {
            "documents": len(self.docstore),
            "vectors": int(self.index.ntotal) if self.index is not None else 0,
            "dimension": int(self.index.d) if self.index is not None else "—",
        }


class VectorStore(FaissVectorStore):
    """Backward-compatible alias for the FAISS vector store implementation."""

    pass
