"""Graph entity models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    type: str = "Unknown"
    description: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Relationship(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str | None = None
