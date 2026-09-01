"""Neo4j persistence layer.

The module intentionally imports the Neo4j Python driver lazily so the
Streamlit UI can still launch before optional infrastructure is installed or
while the database is offline.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

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
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            logger.warning("Neo4j driver is not installed: %s", exc)
            return
        try:
            username = getattr(self.settings, "neo4j_username", getattr(self.settings, "neo4j_user", "neo4j"))
            self.driver = GraphDatabase.driver(self.settings.neo4j_uri, auth=(username, self.settings.neo4j_password))
            with self.driver.session() as session:
                session.run("RETURN 1").consume()
            self._available = True
        except Exception as exc:
            logger.warning("Neo4j is unavailable: %s", exc)
            self.driver = None
            self._available = False

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def is_available(self) -> bool:
        """Return whether the Neo4j driver is installed and reachable."""
        return self._available

    @contextmanager
    def session(self) -> Iterator[Any]:
        if self.driver is None:
            raise RuntimeError("Neo4j is not available. Install neo4j and start the database.")
        with self.driver.session() as session:
            yield session

    def verify(self) -> bool:
        return self.is_available()

    def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a read/write query and return records as dictionaries."""
        if not self.is_available():
            return []
        with self.session() as session:
            records = session.run(cypher, parameters or {})
            return [dict(record) for record in records]

    def upsert_entities(self, entities: list[Entity], chunk_id: str) -> None:
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
            session.run(query, entities=[e.model_dump() for e in entities], chunk_id=chunk_id)

    def upsert_relationships(self, relationships: list[Relationship], chunk_id: str) -> None:
        if not self.is_available() or not relationships:
            return
        query = """
        UNWIND $relationships AS rel
        MERGE (s:Entity {name: rel.subject})
        MERGE (o:Entity {name: rel.object})
        MERGE (s)-[r:RELATION {relation: rel.relation}]->(o)
        SET r.confidence = rel.confidence,
            r.evidence = rel.evidence,
            r.chunk_id = $chunk_id
        """
        with self.session() as session:
            session.run(query, relationships=[r.model_dump() for r in relationships], chunk_id=chunk_id)
