"""Parse normalized loaded documents into text."""
from __future__ import annotations

from pathlib import Path

from ingestion.loader import LoadedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_document(document: LoadedDocument) -> str:
    """Parse a single loaded document into clean text.

    ``load_documents`` already delegates file parsing to LangChain Community
    loaders and stores UTF-8 normalized text in ``content``. This parser remains
    as a validation/cleanup boundary for ingestion and backward compatibility.
    """
    suffix = document.suffix
    try:
        if suffix in {".pdf", ".txt", ".docx", ".md", ".markdown"}:
            return document.content.decode("utf-8", errors="ignore").strip()
        raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to parse %s", document.source_path)
        raise RuntimeError(f"Failed to parse {document.source_path}: {exc}") from exc