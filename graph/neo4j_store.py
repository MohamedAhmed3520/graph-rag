"""Neo4j persistence layer.

Supports local environment variables and Streamlit Cloud secrets.
Neo4j is optional: if unavailable, the application continues with
vector-only RAG.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
import os

from config.settings import Settings, get_settings
from graph.entities import Entity, Relationship
from utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jStore:
    """Production wrapper around Neo4j with graceful unavailability handling."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.driver: Any | None = None
        self._available = False

        self.neo4j_uri = self._get_secret("NEO4J_URI")
        self.neo4j_username = self._get_secret("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = self._get_secret("NEO4J_PASSWORD")
        self.neo4j_database = self._get_secret("NEO4J_DATABASE", "neo4j")

        self._connect()

    @staticmethod
    def _get_secret(key: str, default: str | None = None) -> str | None:
        """Read a setting from Streamlit secrets, then environment variables."""
        try:
            import streamlit as st

            value = st.secrets.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        except Exception:
            pass

        value = os.getenv(key)

        if value is not None and value.strip():
            return value.strip()

        return default

    def _connect(self) -> None:
        if not self.neo4j_uri:
            logger.info("Neo4j is not configured. Using vector-only RAG.")
            return

        if not self.neo4j_password:
            logger.warning(
                "NEO4J_PASSWORD is missing. Neo4j will be disabled."
            )
            return

        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            logger.warning("Neo4j driver is not installed: %s", exc)
            return

        try:
            logger.info(
                "Connecting to Neo4j: %s (database=%s)",
                self.neo4j_uri,
                self.neo4j_database,
            )

            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(
                    self.neo4j_username,
                    self.neo4j_password,
                ),
            )

            # Verify network/authentication
            self.driver.verify_connectivity()

            # Verify database access
            with self.driver.session(
                database=self.neo4j_database
            ) as session:
                session.run("RETURN 1").consume()

            self._available = True

            logger.info(
                "Neo4j connected successfully: database=%s",
                self.neo4j_database,
            )

        except Exception as exc:
            logger.warning(
                "Neo4j is unavailable: %s",
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
        """Return whether Neo4j is installed and reachable."""
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

    def verify(self) -> bool:
        return self.is_available()

    def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read/write query and return records as dictionaries."""
        if not self.is_available():
            return []

        with self.session() as session:
            records = session.run(
                cypher,
                parameters or {},
            )

            return [dict(record) for record in records]

    def upsert_entities(
        self,
        entities: list[Entity],
        chunk_id: str,
    ) -> None:
        if not self.is_available() or not entities:
            return

        query = """
        UNWIND $entities AS entity

        MERGE (e:Entity {name: entity.name})

        SET e.type = entity.type,
            e.description = entity.description,
            e.confidence = entity.confidence,
            e.chunk_id = $chunk_id
        """

        with self.session() as session:
            session.run(
                query,
                entities=[
                    e.model_dump()
                    for e in entities
                ],
                chunk_id=chunk_id,
            )

    def upsert_relationships(
        self,
        relationships: list[Relationship],
        chunk_id: str,
    ) -> None:
        if not self.is_available() or not relationships:
            return

        query = """
        UNWIND $relationships AS rel

        MERGE (s:Entity {name: rel.subject})
        MERGE (o:Entity {name: rel.object})

        MERGE (s)-[r:RELATION {
            relation: rel.relation
        }]->(o)

        SET r.confidence = rel.confidence,
            r.evidence = rel.evidence,
            r.chunk_id = $chunk_id
        """

        with self.session() as session:
            session.run(
                query,
                relationships=[
                    r.model_dump()
                    for r in relationships
                ],
                chunk_id=chunk_id,
            )
