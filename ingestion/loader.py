"""Document loading utilities backed by LangChain Community loaders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from utils.logger import get_logger

logger = get_logger(__name__)


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx", ".md", ".markdown"}


@dataclass(slots=True)
class LoadedDocument:
    """Raw document payload before parsing."""

    source_path: Path
    content: bytes
    suffix: str


def load_documents(paths: Iterable[Path]) -> list[LoadedDocument]:
    """Load supported files from the provided paths."""
    documents: list[LoadedDocument] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            logger.warning("Skipping missing file: %s", path)
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            logger.warning("Unsupported file type skipped: %s", path)
            continue
        documents.append(LoadedDocument(source_path=path, content=path.read_bytes(), suffix=suffix))
    return documents