"""Graph build orchestration."""
from __future__ import annotations

from typing import Any

from graph.extractor import GraphExtractor
from graph.neo4j_store import Neo4jStore


class GraphBuilder:
    """Build and persist knowledge-graph facts in Neo4j."""

    def __init__(
        self,
        store: Neo4jStore | None = None,
        extractor: GraphExtractor | None = None,
    ) -> None:
        self.store = store or Neo4jStore()
        self.extractor = extractor

    def build_from_text(
        self,
        text: str,
        chunk_id: str,
    ) -> dict[str, Any]:
        """Extract graph facts from one chunk and store them."""

        extractor = (
            self.extractor
            or GraphExtractor()
        )

        data = extractor.extract(text)

        self.store.upsert_entities(
            data["entities"],
            chunk_id,
        )

        self.store.upsert_relationships(
            data["relationships"],
            chunk_id,
        )

        return data

    def build(
        self,
        graph_documents: list[dict[str, Any]],
    ) -> dict[str, int]:
        """
        Persist already-extracted graph documents.

        This method is used by the LangGraph ingestion workflow after
        GraphExtractor.extract_documents() has already produced the facts.
        It does NOT call the LLM again.
        """

        entity_count = 0
        relationship_count = 0

        if not graph_documents:
            return {
                "entities": 0,
                "relationships": 0,
            }

        for document in graph_documents:

            chunk_id = str(
                document.get(
                    "chunk_id",
                    "",
                )
            )

            entities = document.get(
                "entities",
                [],
            )

            relationships = document.get(
                "relationships",
                [],
            )

            if entities:
                self.store.upsert_entities(
                    entities,
                    chunk_id,
                )

                entity_count += len(
                    entities
                )

            if relationships:
                self.store.upsert_relationships(
                    relationships,
                    chunk_id,
                )

                relationship_count += len(
                    relationships
                )

        return {
            "entities": entity_count,
            "relationships": relationship_count,
        }
