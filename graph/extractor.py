"""LLM-based graph extraction."""
from __future__ import annotations

import json
from typing import Any

from config.models import create_llm
from graph.entities import Entity, Relationship
from prompts.extraction import GRAPH_EXTRACTION_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


class GraphExtractor:
    def __init__(self) -> None:
        self.llm = create_llm()

    def extract(self, text: str) -> dict[str, list[Any]]:
        prompt = GRAPH_EXTRACTION_PROMPT.format(text=text)
        response = self.llm.invoke(prompt)
        payload = self._parse_json(response.content)
        entities = [Entity(**item) for item in payload.get("entities", [])]
        relationships = [Relationship(**item) for item in payload.get("relationships", [])]
        concepts = [str(x) for x in payload.get("concepts", [])]
        keywords = [str(x) for x in payload.get("keywords", [])]
        return {"entities": entities, "relationships": relationships, "concepts": concepts, "keywords": keywords}

    def _parse_json(self, content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.exception("Graph extraction JSON parsing failed")
            raise ValueError(f"Invalid graph extraction output: {content}") from exc
