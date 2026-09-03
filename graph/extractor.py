"""LLM-based knowledge graph extraction."""
from __future__ import annotations

import json
from typing import Any

from config.models import create_llm
from graph.entities import Entity, Relationship
from prompts.extraction import GRAPH_EXTRACTION_PROMPT
from utils.logger import get_logger


logger = get_logger(__name__)


class GraphExtractor:
    """Extract entities and relationships from text chunks."""

    def __init__(self) -> None:
        self.llm = create_llm()

    def extract(
        self,
        text: str,
    ) -> dict[str, list[Any]]:
        """Extract graph information from one text block."""

        if not text or not text.strip():
            return {
                "entities": [],
                "relationships": [],
                "concepts": [],
                "keywords": [],
            }

        prompt = GRAPH_EXTRACTION_PROMPT.format(
            text=text
        )

        response = self.llm.invoke(prompt)

        content = str(
            response.content
        ).strip()

        payload = self._parse_json(content)

        entities: list[Entity] = []
        relationships: list[Relationship] = []

        # -------------------------------
        # Entities
        # -------------------------------

        for item in payload.get(
            "entities",
            [],
        ):
            try:
                if not isinstance(item, dict):
                    continue

                entities.append(
                    Entity(**item)
                )

            except Exception as exc:
                logger.warning(
                    "Skipping invalid entity: %s",
                    exc,
                )

        # -------------------------------
        # Relationships
        # -------------------------------

        for item in payload.get(
            "relationships",
            [],
        ):
            try:
                if not isinstance(item, dict):
                    continue

                relationships.append(
                    Relationship(**item)
                )

            except Exception as exc:
                logger.warning(
                    "Skipping invalid relationship: %s",
                    exc,
                )

        return {
            "entities": entities,
            "relationships": relationships,
            "concepts": [
                str(x)
                for x in payload.get(
                    "concepts",
                    [],
                )
            ],
            "keywords": [
                str(x)
                for x in payload.get(
                    "keywords",
                    [],
                )
            ],
        }

    def extract_documents(
        self,
        chunks: list[Any],
    ) -> list[dict[str, Any]]:
        """
        Extract graph facts from every chunk.

        Each returned item keeps the original chunk_id so the
        graph relationships can be traced back to the source chunk.
        """

        results: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks):

            chunk_id = getattr(
                chunk,
                "chunk_id",
                None,
            )

            text = getattr(
                chunk,
                "text",
                "",
            )

            if not text:
                logger.warning(
                    "Skipping empty chunk %s",
                    index,
                )
                continue

            try:
                extracted = self.extract(
                    text
                )

                results.append(
                    {
                        "chunk_id": chunk_id or f"chunk-{index}",
                        "entities": extracted[
                            "entities"
                        ],
                        "relationships": extracted[
                            "relationships"
                        ],
                        "concepts": extracted[
                            "concepts"
                        ],
                        "keywords": extracted[
                            "keywords"
                        ],
                    }
                )

                logger.info(
                    "Graph extraction completed for chunk %s",
                    index + 1,
                )

            except Exception as exc:
                logger.exception(
                    "Graph extraction failed for chunk %s: %s",
                    index + 1,
                    exc,
                )

                # Do not destroy the entire ingestion job because
                # one chunk failed.
                results.append(
                    {
                        "chunk_id": chunk_id or f"chunk-{index}",
                        "entities": [],
                        "relationships": [],
                        "concepts": [],
                        "keywords": [],
                    }
                )

        return results

    @staticmethod
    def _parse_json(
        content: str,
    ) -> dict[str, Any]:
        """Parse JSON from normal output or Markdown code fences."""

        cleaned = (
            content
            .replace(
                "```json",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

        # Find the JSON object even if the model added
        # explanatory text before/after it.
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start >= 0 and end > start:
            cleaned = cleaned[
                start : end + 1
            ]

        try:
            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:
            logger.error(
                "Invalid graph extraction JSON: %s",
                cleaned[:1000],
            )

            raise ValueError(
                "Graph extraction model returned invalid JSON"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Graph extraction output must be a JSON object"
            )

        return data
