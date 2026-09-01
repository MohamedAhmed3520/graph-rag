"""Graph search utilities for traversal-based retrieval and visualization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graph.neo4j_store import Neo4jStore
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class GraphSearchResult:
    """Structured graph visualization result."""

    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


class GraphSearch:
    """Search graph neighborhoods using Neo4j when available."""

    def __init__(self, store: Neo4jStore | None = None) -> None:
        self.store = store

    def _store(self) -> Neo4jStore:
        if self.store is None:
            self.store = Neo4jStore()
        return self.store

    def search(self, query: str, limit: int = 10, relation_types: list[str] | None = None) -> list[dict[str, Any]]:
        """Return triples relevant to a user query for Graph RAG context."""
        store = self._store()
        if not store.is_available():
            return []
        cypher = """
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE ($query = '' OR toLower(s.name) CONTAINS toLower($query)
               OR toLower(o.name) CONTAINS toLower($query)
               OR toLower(r.relation) CONTAINS toLower($query))
          AND (size($relation_types) = 0 OR r.relation IN $relation_types)
        RETURN s.name AS subject, r.relation AS relation, o.name AS object,
               coalesce(r.confidence, 0.0) AS confidence, coalesce(r.evidence, '') AS evidence,
               coalesce(r.chunk_id, '') AS chunk_id
        LIMIT $limit
        """
        try:
            return store.query(cypher, {"query": query, "limit": limit, "relation_types": relation_types or []})
        except Exception as exc:
            logger.warning("Graph search failed: %s", exc)
            return []

    def subgraph(self, query: str = "", limit: int = 80) -> dict[str, list[dict[str, Any]]]:
        """Return nodes and edges for interactive graph visualization."""
        triples = self.search(query=query, limit=limit)
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for triple in triples:
            subject = str(triple.get("subject", ""))
            obj = str(triple.get("object", ""))
            if subject:
                nodes.setdefault(subject, {"id": subject, "label": subject, "type": "Entity"})
            if obj:
                nodes.setdefault(obj, {"id": obj, "label": obj, "type": "Entity"})
            if subject and obj:
                edges.append({"source": subject, "target": obj, "label": triple.get("relation", "RELATION"), "type": triple.get("relation", "RELATION")})
        return {"nodes": list(nodes.values()), "edges": edges}


class GraphSearcher(GraphSearch):
    """Backward-compatible alias for older imports."""

    def search_result(self, query: str, limit: int = 10) -> GraphSearchResult:
        graph = self.subgraph(query=query, limit=limit)
        return GraphSearchResult(nodes=graph["nodes"], relationships=graph["edges"])