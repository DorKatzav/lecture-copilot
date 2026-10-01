"""Structured outputs of the agents (PLAN.md §3.4). Every Ollama JSON call validates against one of these."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Concept(BaseModel):
    term: str
    explanation: str
    canonical_key: str


class Claim(BaseModel):
    text: str
    normalized: str
    importance: int = Field(ge=0, le=100)
    contradicts: str | None = None   # the earlier claim (from the recalled list) this one contradicts (M3)


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


Text = Annotated[str, Field(min_length=1)]


class DigestSection(BaseModel):
    """Map step of the Digest: a block of chunk summaries → paragraphs of the full summary."""
    paragraphs: list[Text] = Field(min_length=1, max_length=3)


class Continuation(BaseModel):
    new: list[str]
    repeated: list[str]
    contradicts: list[str]


class DigestExec(BaseModel):
    """Reduce step of the Digest."""
    exec_summary: list[Text] = Field(min_length=5, max_length=5)
    continuation: Continuation | None = None


class DigestExecWithPrevious(DigestExec):
    """When a previous lecture is given the comparison is required: a null answer is retried (D-M3-5)."""
    continuation: Continuation
