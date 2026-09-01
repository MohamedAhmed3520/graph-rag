"""Knowledge-graph retriever."""
from __future__ import annotations

from graph.graph_search import GraphSearch
from graph.neo4j_store import Neo4jStore
from utils.logger import get_logger

logger = get_logger(__name__)


class GraphRetriever:
    """Retrieve graph neighborhoods relevant to a natural-language query."""

    def __init__(self, store: Neo4jStore | None = None) -> None:
        self.store = store
        self.searcher: GraphSearch | None = GraphSearch(store) if store is not None else None

    def _searcher(self) -> GraphSearch:
        if self.searcher is None:
            self.store = Neo4jStore()
            self.searcher = GraphSearch(self.store)
        return self.searcher

    def retrieve(self, query: str, top_k: int = 8, relation_types: list[str] | None = None) -> list[dict[str, str | float]]:
        try:
            return self._searcher().search(query=query, limit=top_k, relation_types=relation_types)
        except Exception as exc:
            logger.warning("Graph retrieval unavailable: %s", exc)
            return []
