"""Entity-term graph lookup followed by neighborhood traversal."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from graph.neo4j_store import Neo4jStore


@dataclass(slots=True)
class GraphSearchResult:
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


class GraphSearch:
    def __init__(
        self,
        store: Neo4jStore | None = None,
    ):
        self.store = store

    def _store(self) -> Neo4jStore:
        if self.store is None:
            self.store = Neo4jStore()

        return self.store

    @staticmethod
    def _terms(query: str) -> list[str]:
        stopwords = {
            "what",
            "which",
            "who",
            "where",
            "when",
            "why",
            "how",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "in",
            "on",
            "and",
            "or",
            "with",
            "from",
            "about",
            "explain",
            "tell",
            "me",
            "does",
            "do",
            "can",
            "could",
            "would",
            "different",
            "types",
            "give",
            "please",
        }

        words = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]{1,}",
            query.lower(),
        )

        return [
            word
            for word in words
            if word not in stopwords
        ]

    def _sample(
        self,
        limit: int = 10,
        relation_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return a general sample of the graph when no query terms match.

        This is used so the graph visualization has something to show even
        before the user types a search term (the term-based ``search`` returns
        nothing for an empty query, which used to make the graph appear
        "undetected"/empty).
        """
        store = self._store()

        if not store.is_available():
            return []

        return store.query(
            """
            MATCH (s:Entity)-[r:RELATION]->(o:Entity)

            WHERE
                size($relation_types) = 0
                OR r.relation IN $relation_types

            RETURN
                s.name AS seed,
                s.name AS subject,
                r.relation AS relation,
                o.name AS object,
                coalesce(r.confidence, 0.0)
                    AS confidence,
                coalesce(r.evidence, '')
                    AS evidence,
                coalesce(r.chunk_id, '')
                    AS chunk_id

            ORDER BY confidence DESC

            LIMIT $limit
            """,
            {
                "limit": max(1, limit),
                "relation_types": relation_types or [],
            },
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        relation_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:

        store = self._store()

        if not store.is_available():
            return []

        terms = self._terms(query)

        if not terms:
            return []

        return store.query(
            """
            MATCH (seed:Entity)

            WHERE any(
                term IN $terms
                WHERE toLower(seed.name) CONTAINS term
            )

            MATCH (seed)-[r:RELATION]-(neighbor:Entity)

            WHERE
                size($relation_types) = 0
                OR r.relation IN $relation_types

            RETURN
                seed.name AS seed,

                CASE
                    WHEN startNode(r) = seed
                    THEN seed.name
                    ELSE neighbor.name
                END AS subject,

                r.relation AS relation,

                CASE
                    WHEN startNode(r) = seed
                    THEN neighbor.name
                    ELSE seed.name
                END AS object,

                coalesce(r.confidence, 0.0)
                    AS confidence,

                coalesce(r.evidence, '')
                    AS evidence,

                coalesce(r.chunk_id, '')
                    AS chunk_id

            ORDER BY confidence DESC

            LIMIT $limit
            """,
            {
                "terms": terms,
                "limit": max(1, limit),
                "relation_types": relation_types or [],
            },
        )

    def subgraph(
        self,
        query: str = "",
        limit: int = 80,
    ) -> dict[str, list[dict[str, Any]]]:

        triples = self.search(
            query,
            limit,
        )

        # When the query is empty (or none of its terms match an entity) fall
        # back to a general sample so the graph is visible instead of blank.
        if not triples:
            triples = self._sample(
                limit,
            )

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        for triple in triples:
            subject = str(
                triple.get("subject", "")
            )
            obj = str(
                triple.get("object", "")
            )

            if subject:
                nodes.setdefault(
                    subject,
                    {
                        "id": subject,
                        "name": subject,
                        "label": subject,
                        "type": "Entity",
                    },
                )

            if obj:
                nodes.setdefault(
                    obj,
                    {
                        "id": obj,
                        "name": obj,
                        "label": obj,
                        "type": "Entity",
                    },
                )

            if subject and obj:
                edges.append(
                    {
                        "source": subject,
                        "target": obj,
                        "relation": triple.get(
                            "relation",
                            "RELATION",
                        ),
                        "confidence": triple.get(
                            "confidence",
                            0.0,
                        ),
                    }
                )

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
        }


class GraphSearcher(GraphSearch):
    def search_result(
        self,
        query: str,
        limit: int = 10,
    ) -> GraphSearchResult:

        graph = self.subgraph(
            query,
            limit,
        )

        return GraphSearchResult(
            nodes=graph["nodes"],
            relationships=graph["edges"],
        )
