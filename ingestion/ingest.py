"""High-level ingestion orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from config.settings import Settings, get_settings
from ingestion.loader import load_documents
from ingestion.parser import parse_document
from ingestion.splitter import Chunk, split_text
from utils.helpers import stable_id
from utils.logger import get_logger

logger = get_logger(__name__)


def ingest_paths(paths: list[Path], settings: Settings) -> list[Chunk]:
    """Load, parse, and chunk documents into normalized chunks."""
    loaded = load_documents(paths)
    chunks: list[Chunk] = []
    for doc in loaded:
        text = parse_document(doc)
        if not text.strip():
            logger.warning("Empty document skipped: %s", doc.source_path)
            continue
        source_id = stable_id(str(doc.source_path))
        for index, chunk_text in enumerate(split_text(text, settings.chunk_size, settings.chunk_overlap)):
            chunk_id = stable_id(f"{source_id}:{index}:{chunk_text[:128]}")
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=source_id,
                    text=chunk_text,
                    index=index,
                    metadata={"source_path": str(doc.source_path), "suffix": doc.suffix},
                )
            )
    return chunks


def ingest_documents(paths: Iterable[Path | str], settings: Settings | None = None) -> list[Chunk]:
    """Public ingestion entrypoint used by the UI and workflow layer.

    This wrapper accepts a broad iterable of path-like values, resolves a
    runtime settings object if one is not provided, and delegates to the core
    ingestion pipeline.
    """
    runtime_settings = settings or get_settings()
    normalized_paths = [Path(path) for path in paths]
    if not normalized_paths:
        logger.warning("No document paths were provided to ingestion")
        return []
    return ingest_paths(normalized_paths, runtime_settings)


class IngestionPipeline:
    """Backward-compatible ingestion pipeline adapter.

    The workflow layer in this codebase expects a class-based ingestion API
    with ``load`` and ``split`` methods. This adapter keeps that interface
    available while delegating to the production ingestion functions defined in
    this module and adjacent ingestion helpers.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        """Return the active ingestion settings."""
        return self._settings

    def load(self, paths: Iterable[Path | str]) -> list[Chunk]:
        """Load and chunk documents from the provided paths."""
        return ingest_documents(paths, self._settings)

    def split(self, documents: Iterable[Chunk]) -> list[Chunk]:
        """Return chunk objects unchanged for workflow compatibility.

        The workflow currently passes already-ingested chunk objects between the
        ``load`` and ``split`` nodes. This method intentionally preserves that
        contract so the pipeline remains stable while the broader architecture
        continues to use the chunk abstraction.
        """
        return list(documents)


__all__ = ["IngestionPipeline", "ingest_paths", "ingest_documents"]