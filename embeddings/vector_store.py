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
    """
    Persistent FAISS vector store.

    Important:
    FAISS vector positions are mapped explicitly to chunk IDs using
    index_to_chunk_id instead of relying on dictionary ordering.
    """

    def __init__(
        self,
        index_path: Path | Settings,
        docstore_path: Path | None = None,
        embedding_model_name: str | None = None,
    ) -> None:

        # --------------------------------------------------
        # Settings-based initialization
        # --------------------------------------------------

        if isinstance(index_path, Settings):

            settings = index_path

            self.index_path = Path(
                settings.faiss_index_path
            )

            self.docstore_path = Path(
                settings.faiss_docstore_path
            )

            self.embedding_model_name = (
                embedding_model_name
                or settings.embedding_model
            )

        # --------------------------------------------------
        # Explicit paths
        # --------------------------------------------------

        else:

            if docstore_path is None:
                raise ValueError(
                    "docstore_path is required when "
                    "index_path is not a Settings instance."
                )

            self.index_path = Path(
                index_path
            )

            self.docstore_path = Path(
                docstore_path
            )

            self.embedding_model_name = (
                embedding_model_name
            )

        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------

        self.embedding_model: EmbeddingModel | None = None

        self.dimension: int | None = None

        self.docstore: dict[
            str,
            VectorRecord,
        ] = {}

        # IMPORTANT:
        # FAISS index position -> chunk ID
        self.index_to_chunk_id: list[str] = []

        # Load persisted data
        self.index = self._load_index()

        self._load_docstore()

        self._validate_mapping()

    # ======================================================
    # Embedding model
    # ======================================================

    def _get_embedding_model(
        self,
    ) -> EmbeddingModel:

        if self.embedding_model is None:

            self.embedding_model = EmbeddingModel(
                self.embedding_model_name
            )

            self.dimension = (
                self.embedding_model.dimension
            )

        return self.embedding_model

    # ======================================================
    # FAISS index creation
    # ======================================================

    def _create_empty_index(
        self,
    ) -> faiss.Index:

        dimension = (
            self.dimension
            or self._get_embedding_model().dimension
        )

        return faiss.IndexFlatIP(
            dimension
        )

    # ======================================================
    # Load FAISS
    # ======================================================

    def _load_index(
        self,
    ) -> faiss.Index:

        if self.index_path.exists():

            index = faiss.read_index(
                str(self.index_path)
            )

            self.dimension = int(
                index.d
            )

            return index

        return self._create_empty_index()

    # ======================================================
    # Load document store
    # ======================================================

    def _load_docstore(
        self,
    ) -> None:

        raw = read_json(
            self.docstore_path,
            {},
        )

        # --------------------------------------------------
        # New format
        # --------------------------------------------------

        if (
            isinstance(raw, dict)
            and "records" in raw
        ):

            records = raw.get(
                "records",
                {},
            )

            self.index_to_chunk_id = list(
                raw.get(
                    "index_to_chunk_id",
                    [],
                )
            )

        # --------------------------------------------------
        # Old format
        # --------------------------------------------------

        else:

            records = (
                raw
                if isinstance(raw, dict)
                else {}
            )

            # Old files do not have reliable
            # explicit FAISS mappings.
            self.index_to_chunk_id = []

        self.docstore = {}

        for key, value in records.items():

            self.docstore[key] = (
                VectorRecord(**value)
            )

    # ======================================================
    # Validate FAISS ↔ docstore mapping
    # ======================================================

    def _validate_mapping(
        self,
    ) -> None:

        # Completely empty database
        if (
            self.index.ntotal == 0
            and not self.docstore
        ):
            return

        # Mapping must exist for every vector
        if (
            len(
                self.index_to_chunk_id
            )
            != self.index.ntotal
        ):

            raise RuntimeError(
                "FAISS index/docstore mapping is inconsistent. "
                "Clear and rebuild the vector database."
            )

        # Every mapped chunk must exist
        missing = [
            chunk_id
            for chunk_id
            in self.index_to_chunk_id
            if chunk_id
            not in self.docstore
        ]

        if missing:

            raise RuntimeError(
                "FAISS mapping references missing chunks. "
                "Clear and rebuild the vector database."
            )

    # ======================================================
    # Add chunks
    # ======================================================

    def add_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:

        if not chunks:
            return

        # --------------------------------------------------
        # Prevent duplicate vectors
        # --------------------------------------------------

        new_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_id
            not in self.docstore
        ]

        if not new_chunks:
            return

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        model = (
            self._get_embedding_model()
        )

        embeddings = model.embed_documents(
            [
                chunk.text
                for chunk in new_chunks
            ]
        )

        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if vectors.ndim == 1:

            vectors = vectors.reshape(
                1,
                -1,
            )

        # --------------------------------------------------
        # Dimension validation
        # --------------------------------------------------

        if vectors.shape[1] != self.index.d:

            # Empty FAISS index:
            # recreate it with actual dimension.
            if self.index.ntotal == 0:

                self.index = (
                    faiss.IndexFlatIP(
                        vectors.shape[1]
                    )
                )

                self.dimension = (
                    vectors.shape[1]
                )

            else:

                raise RuntimeError(
                    "Embedding dimension changed. "
                    "Clear and rebuild the FAISS database."
                )

        # --------------------------------------------------
        # Add vectors to FAISS
        # --------------------------------------------------

        self.index.add(
            vectors
        )

        # --------------------------------------------------
        # Add documents + explicit mapping
        # --------------------------------------------------

        for chunk in new_chunks:

            self.docstore[
                chunk.chunk_id
            ] = VectorRecord(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )

            self.index_to_chunk_id.append(
                chunk.chunk_id
            )

        # --------------------------------------------------
        # Persist
        # --------------------------------------------------

        self.persist()

    # ======================================================
    # Persist
    # ======================================================

    def persist(
        self,
    ) -> None:

        ensure_dirs(
            self.index_path.parent,
            self.docstore_path.parent,
        )

        # Save FAISS index
        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        # Save document store + mapping
        records = {}

        for key, record in (
            self.docstore.items()
        ):

            records[key] = {
                "chunk_id": record.chunk_id,
                "source_id": record.source_id,
                "text": record.text,
                "metadata": record.metadata,
            }

        write_json(
            self.docstore_path,
            {
                "records": records,
                "index_to_chunk_id":
                    self.index_to_chunk_id,
            },
        )

    # ======================================================
    # Clear
    # ======================================================

    def clear(
        self,
    ) -> None:

        self.index_path.unlink(
            missing_ok=True
        )

        self.docstore_path.unlink(
            missing_ok=True
        )

        self.docstore = {}

        self.index_to_chunk_id = []

        self.embedding_model = None

        self.dimension = None

        self.index = (
            self._create_empty_index()
        )

    # ======================================================
    # SEARCH
    # ======================================================

    def search(
        self,
        query_vector: list[float]
        | np.ndarray,
        top_k: int = 8,
        threshold: float = 0.0,
        filters: dict[str, str]
        | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search FAISS and return matching chunks.

        Because embeddings are normalized and the index is
        IndexFlatIP, the returned inner product is used as
        the similarity score.
        """

        # --------------------------------------------------
        # Empty index
        # --------------------------------------------------

        if (
            self.index is None
            or self.index.ntotal == 0
        ):

            return []

        if not self.docstore:

            return []

        # --------------------------------------------------
        # Ensure FAISS mapping is valid
        # --------------------------------------------------

        self._validate_mapping()

        # --------------------------------------------------
        # Prepare query vector
        # --------------------------------------------------

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if vector.ndim == 1:

            vector = vector.reshape(
                1,
                -1,
            )

        # --------------------------------------------------
        # Validate dimensions
        # --------------------------------------------------

        if vector.shape[1] != self.index.d:

            raise RuntimeError(
                f"Query embedding dimension "
                f"{vector.shape[1]} does not match "
                f"FAISS dimension {self.index.d}."
            )

        # --------------------------------------------------
        # Search
        # --------------------------------------------------

        k = min(
            max(1, int(top_k)),
            self.index.ntotal,
        )

        scores, indices = (
            self.index.search(
                vector,
                k,
            )
        )

        results: list[
            dict[str, Any]
        ] = []

        # --------------------------------------------------
        # Resolve vector positions → chunk IDs
        # --------------------------------------------------

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            idx = int(idx)

            score = float(
                score
            )

            if idx < 0:
                continue

            # Similarity threshold
            if score < threshold:
                continue

            # Explicit mapping
            chunk_id = (
                self.index_to_chunk_id[
                    idx
                ]
            )

            record = self.docstore.get(
                chunk_id
            )

            if record is None:
                continue

            # --------------------------------------------------
            # Optional metadata filters
            # --------------------------------------------------

            if filters:

                skipped = False

                for (
                    key,
                    value,
                ) in filters.items():

                    if (
                        record.metadata.get(
                            key
                        )
                        != value
                    ):
                        skipped = True
                        break

                if skipped:
                    continue

            # --------------------------------------------------
            # Source
            # --------------------------------------------------

            source = record.metadata.get(
                "source",
                record.metadata.get(
                    "source_path",
                    "unknown",
                ),
            )

            # --------------------------------------------------
            # Standardized hit format
            # --------------------------------------------------

            results.append(
                {
                    "id": record.chunk_id,

                    "chunk_id":
                        record.chunk_id,

                    "text":
                        record.text,

                    "score":
                        score,

                    "metadata":
                        {
                            "source":
                                source,

                            "source_path":
                                record.metadata.get(
                                    "source_path",
                                    source,
                                ),

                            "source_id":
                                record.source_id,

                            "chunk_id":
                                record.chunk_id,

                            **record.metadata,
                        },
                }
            )

        return results

    # ======================================================
    # Statistics
    # ======================================================

    def stats(
        self,
    ) -> dict[str, int | str]:

        return {
            "documents":
                len(self.docstore),

            "vectors":
                int(
                    self.index.ntotal
                ),

            "dimension":
                int(
                    self.index.d
                ),
        }


class VectorStore(
    FaissVectorStore
):
    """Backward-compatible alias."""

    pass
