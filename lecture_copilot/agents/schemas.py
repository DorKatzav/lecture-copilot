"""Structured outputs of the agents (PLAN.md §3.4). Every Ollama JSON call validates against one of these."""

from typing import Literal

from pydantic import BaseModel, Field


class Concept(BaseModel):
    term: str
    explanation: str
    canonical_key: str


class Claim(BaseModel):
    text: str
    normalized: str
    importance: int = Field(ge=0, le=100)


class Item(BaseModel):
    kind: Literal["question", "action", "highlight", "note", "decision"]
    text: str
    owner: str | None = None
    due: str | None = None


class ExtractResult(BaseModel):
    chunk_summary: str
    concepts: list[Concept]
    claims: list[Claim]
    items: list[Item]
