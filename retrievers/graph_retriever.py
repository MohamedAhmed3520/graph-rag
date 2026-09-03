"""Knowledge-graph retriever."""
from __future__ import annotations

from typing import Any

from config.settings import Settings, get_settings
from graph.graph_search import GraphSearch
from graph.neo4j_store import Neo4jStore
from utils.logger import get_logger


logger = get_logger(__name__)


class GraphRetriever:
    """Retrieve graph evidence relevant to a natural-language query."""

    def __init__(
        self,
        store: Neo4jStore | None = None,
        settings: Settings | None = None,
    ) -> None:

        self.settings = settings or get_settings()

        self.store = store
        self.searcher: GraphSearch | None = (
            GraphSearch(store)
            if store is not None
            else None
        )

    def _searcher(self) -> GraphSearch:
        """Return a graph searcher with the configured Neo4j connection."""

        if self.searcher is None:
            self.store = Neo4jStore(
                self.settings
            )

            self.searcher = GraphSearch(
                self.store
            )

        return self.searcher

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        relation_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:

        query = (query or "").strip()

        if not query:
            return []

        try:
            return self._searcher().search(
                query=query,
                limit=top_k,
                relation_types=relation_types,
            )

        except Exception as exc:
            logger.exception(
                "Graph retrieval failed: %s",
                exc,
            )

            raise
