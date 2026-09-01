"""Document splitting utilities."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    """A chunk of text with metadata."""

    chunk_id: str
    source_id: str
    text: str
    index: int
    metadata: dict[str, str]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Simple deterministic chunking with overlap."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - chunk_overlap)
    return [chunk for chunk in chunks if chunk]