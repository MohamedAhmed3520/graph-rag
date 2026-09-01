"""Graph build orchestration."""
from __future__ import annotations

from graph.extractor import GraphExtractor
from graph.neo4j_store import Neo4jStore


class GraphBuilder:
    def __init__(self, store: Neo4jStore | None = None, extractor: GraphExtractor | None = None) -> None:
        self.store = store or Neo4jStore()
        self.extractor = extractor or GraphExtractor()

    def build_from_text(self, text: str, chunk_id: str) -> dict:
        data = self.extractor.extract(text)
        self.store.upsert_entities(data["entities"], chunk_id)
        self.store.upsert_relationships(data["relationships"], chunk_id)
        return data
