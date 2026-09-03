"""Neo4j persistence and connection handling."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from config.settings import Settings, get_settings
from graph.entities import Entity, Relationship
from utils.logger import get_logger


logger = get_logger(__name__)


class Neo4jStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        self.driver: Any | None = None
        self._available = False

        self.neo4j_uri = self.settings.resolve_neo4j(
            "NEO4J_URI"
        )
        self.neo4j_username = self.settings.resolve_neo4j(
            "NEO4J_USERNAME",
            "neo4j",
        )
        self.neo4j_password = self.settings.resolve_neo4j(
            "NEO4J_PASSWORD"
        )
        self.neo4j_database = self.settings.resolve_neo4j(
            "NEO4J_DATABASE",
            "neo4j",
        )

        self._connect()

    def _connect(self) -> None:
        if not self.neo4j_uri:
            logger.warning(
                "NEO4J_URI is missing. Graph retrieval disabled."
            )
            return

        if not self.neo4j_password:
            logger.warning(
                "NEO4J_PASSWORD is missing. Graph retrieval disabled."
            )
            return

        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(
                    self.neo4j_username or "neo4j",
                    self.neo4j_password,
                ),
            )

            self.driver.verify_connectivity()

            with self.driver.session(
                database=self.neo4j_database
            ) as session:
                session.run("RETURN 1").consume()

            self._available = True

            # Do not print credentials/passwords.
            logger.info(
                "Neo4j connected successfully: %s",
                self.neo4j_uri,
            )

        except Exception as exc:
            logger.exception(
                "Neo4j connection failed: %s",
                exc,
            )

            self.driver = None
            self._available = False

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

        self._available = False

    def is_available(self) -> bool:
        return self._available

    @contextmanager
    def session(self) -> Iterator[Any]:
        if self.driver is None:
            raise RuntimeError(
                "Neo4j is not available."
            )

        with self.driver.session(
            database=self.neo4j_database
        ) as session:
            yield session

    def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._available:
            return []

        with self.session() as session:
            result = session.run(
                cypher,
                parameters or {},
            )

            return [dict(record) for record in result]

    def stats(self) -> dict[str, int]:
        if not self._available:
            return {
                "nodes": 0,
                "relationships": 0,
            }

        rows = self.query(
            """
            MATCH (n:Entity)
            OPTIONAL MATCH ()-[r:RELATION]->()
            RETURN
                count(DISTINCT n) AS nodes,
                count(DISTINCT r) AS relationships
            """
        )

        return (
            rows[0]
            if rows
            else {
                "nodes": 0,
                "relationships": 0,
            }
        )

    def upsert_entities(
        self,
        entities: list[Entity],
        chunk_id: str,
    ) -> None:
        if not self._available or not entities:
            return

        with self.session() as session:
            session.run(
                """
                UNWIND $entities AS entity

                MERGE (e:Entity {name: entity.name})

                SET e.type = entity.type,
                    e.description = entity.description,
                    e.confidence = entity.confidence,
                    e.chunk_id = $chunk_id
                """,
                entities=[
                    entity.model_dump()
                    for entity in entities
                ],
                chunk_id=chunk_id,
            ).consume()

    def upsert_relationships(
        self,
        relationships: list[Relationship],
        chunk_id: str,
    ) -> None:
        if not self._available or not relationships:
            return

        with self.session() as session:
            session.run(
                """
                UNWIND $relationships AS rel

                MERGE (s:Entity {name: rel.subject})
                MERGE (o:Entity {name: rel.object})

                MERGE (
                    s
                )-[r:RELATION {
                    relation: rel.relation
                }]->(
                    o
                )

                SET r.confidence = rel.confidence,
                    r.evidence = rel.evidence,
                    r.chunk_id = $chunk_id
                """,
                relationships=[
                    relationship.model_dump()
                    for relationship in relationships
                ],
                chunk_id=chunk_id,
            ).consume()
