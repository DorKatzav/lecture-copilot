"""Embeddings through Ollama (bge-m3, 1024 dims): one batched call per chunk (budget 2 s). Vectors are stored as
little-endian float32 blobs in SQLite; a provider failure is an EmbedError the caller logs and moves past."""

import asyncio
import struct

import httpx

from lecture_copilot.config import BUDGET_S, EMBED_MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_LOAD_OPTIONS


class EmbedError(RuntimeError):
    pass


async def embed(texts: list[str], *, client: httpx.AsyncClient, model: str = EMBED_MODEL,
                timeout_s: float = BUDGET_S["embed"] * 5, backoff_s: float = 1.0) -> list[list[float]]:
    if not texts:
        return []
    body = {"model": model, "input": texts, "keep_alive": OLLAMA_KEEP_ALIVE, "options": OLLAMA_LOAD_OPTIONS}
    last = ""
    for attempt in (1, 2):
        try:
            r = await client.post("/api/embed", json=body, timeout=timeout_s)
            r.raise_for_status()
            vecs = r.json()["embeddings"]
            if len(vecs) != len(texts):
                raise EmbedError(f"{len(vecs)} embeddings for {len(texts)} texts")
            return vecs
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            last = f"{type(e).__name__}: {e}"
            if attempt == 1:
                await asyncio.sleep(backoff_s)
    raise EmbedError(last)


def pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
