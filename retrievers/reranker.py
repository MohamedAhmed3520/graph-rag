"""Lightweight retrieval reranking utilities."""
from __future__ import annotations

from typing import Any


class SimpleReranker:
    """Rerank by score while preserving citation diversity."""

    def rerank(self, items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        seen: set[str] = set()
        ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        result: list[dict[str, Any]] = []
        for item in ranked:
            source = str(item.get("metadata", {}).get("source", item.get("source", "")))
            key = f"{source}:{item.get('chunk_id', item.get('id', ''))}"
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
            if len(result) >= top_k:
                break
        return result
