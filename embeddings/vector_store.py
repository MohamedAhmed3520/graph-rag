"""Persistent FAISS index with explicit vector-to-chunk mapping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    def __init__(
        self,
        index_path: Path | Settings,
        docstore_path: Path | None = None,
        embedding_model_name: str | None = None,
    ):
        if isinstance(index_path, Settings):
            cfg = index_path
            self.index_path = Path(cfg.faiss_index_path)
            self.docstore_path = Path(cfg.faiss_docstore_path)
            self.embedding_model_name = (
                embedding_model_name or cfg.embedding_model
            )
        else:
            if docstore_path is None:
                raise ValueError(
                    "docstore_path is required when index_path is not Settings"
                )

            self.index_path = Path(index_path)
            self.docstore_path = Path(docstore_path)
            self.embedding_model_name = embedding_model_name

        self.embedding_model: EmbeddingModel | None = None
        self.dimension: int | None = None

        self.docstore: dict[str, VectorRecord] = {}
        self.index_to_chunk_id: list[str] = []

        self.index = self._load_index()

        self._load_docstore()
        self._validate_mapping()

    def _get_embedding_model(self) -> EmbeddingModel:
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel(
                self.embedding_model_name
            )
            self.dimension = self.embedding_model.dimension

        return self.embedding_model

    def _create_empty_index(self) -> faiss.Index:
        dimension = (
            self.dimension
            or self._get_embedding_model().dimension
        )

        return faiss.IndexFlatIP(dimension)

    def _load_index(self) -> faiss.Index:
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            self.dimension = int(index.d)
            return index

        return self._create_empty_index()

    def _load_docstore(self) -> None:
        raw = read_json(self.docstore_path, {})

        if isinstance(raw, dict) and "records" in raw:
            records = raw.get("records", {})
            self.index_to_chunk_id = list(
                raw.get("index_to_chunk_id", [])
            )
        else:
            # Old projects may have only records.
            # We do NOT trust dictionary order for FAISS mapping.
            records = raw if isinstance(raw, dict) else {}

        self.docstore = {
            key: VectorRecord(**value)
            for key, value in records.items()
        }

    def _validate_mapping(self) -> None:
        if self.index.ntotal == 0 and not self.docstore:
            return

        if len(self.index_to_chunk_id) != self.index.ntotal:
            raise RuntimeError(
                "FAISS index/docstore mapping is inconsistent. "
                "Delete/rebuild the FAISS database."
            )

        missing = [
            chunk_id
            for chunk_id in self.index_to_chunk_id
            if chunk_id not in self.docstore
        ]

        if missing:
            raise RuntimeError(
                "FAISS mapping references missing chunks. "
                "Delete/rebuild the FAISS database."
            )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        # Avoid duplicate vectors.
        new_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_id not in self.docstore
        ]

        if not new_chunks:
            return

        model = self._get_embedding_model()

        vectors = np.asarray(
            model.embed_documents(
                [chunk.text for chunk in new_chunks]
            ),
            dtype=np.float32,
        )

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.index.d:
            if self.index.ntotal == 0:
                self.index = faiss.IndexFlatIP(
                    vectors.shape[1]
                )
                self.dimension = vectors.shape[1]
            else:
                raise RuntimeError(
                    "Embedding dimension changed. "
                    "Delete/rebuild the FAISS database."
                )

        self.index.add(vectors)

        for chunk in new_chunks:
            self.docstore[chunk.chunk_id] = VectorRecord(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )

            self.index_to_chunk_id.append(
                chunk.chunk_id
            )

        self.persist()

    def persist(self) -> None:
        ensure_dirs(
            self.index_path.parent,
            self.docstore_path.parent,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        write_json(
            self.docstore_path,
            {
                "records": {
                    key: {
                        "chunk_id": value.chunk_id,
                        "source_id": value.source_id,
                        "text": value.text,
                        "metadata": value.metadata,
                    }
                    for key, value in self.docstore.items()
                },
                "index_to_chunk_id": self.index_to_chunk_id,
            },
        )

    def clear(self) -> None:
        self.index_path.unlink(missing_ok=True)
        self.docstore_path.unlink(missing_ok=True)

        self.docstore = {}
        self.index_to_chunk_id = []

        self.embedding_model = None
        self.dimension = None

        self.index = self._create_empty_index()

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 8,
        threshold: float = 0.0,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:

        if self.index.ntotal == 0:
            return []

        if not self.docstore:
            return []

        self._validate_mapping()

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if vector.shape[1] != self.index.d:
            raise RuntimeError(
                f"Query embedding dimension "
                f"{vector.shape[1]} != FAISS dimension "
                f"{self.index.d}"
            )

        k = min(
            max(1, int(top_k)),
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            vector,
            k,
        )

        results: list[dict[str, Any]] = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):
            idx = int(idx)
            score = float(score)

            if idx < 0:
                continue

            if score < threshold:
                continue

            chunk_id = self.index_to_chunk_id[idx]

            record = self.docstore.get(chunk_id)

            if record is None:
                continue

            if filters:
                if any(
                    record.metadata.get(key) != value
                    for key, value in filters.items()
                ):
                    continue

            source = record.metadata.get(
                "source",
                record.metadata.get(
                    "source_path",
                    "unknown",
                ),
            )

            results.append(
                {
                    "id": record.chunk_id,
                    "chunk_id": record.chunk_id,
                    "text": record.text,
                    "score": score,
                    "metadata": {
                        "source": source,
                        "source_path": record.metadata.get(
                            "source_path",
                            source,
                        ),
                        "source_id": record.source_id,
                        "chunk_id": record.chunk_id,
                        **record.metadata,
                    },
                }
            )

        return results

    def stats(self) -> dict[str, int | str]:
        return {
            "documents": len(self.docstore),
            "vectors": int(self.index.ntotal),
            "dimension": int(self.index.d),
        }


class VectorStore(FaissVectorStore):
    pass
