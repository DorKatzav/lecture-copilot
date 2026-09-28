"""The one Ollama JSON call: schema-constrained output, independent pydantic validation, one retry.

A provider failure (HTTP error, timeout, malformed body, invalid JSON twice) is returned as `JsonCall.error`, never
raised — the caller marks the item failed and the lecture continues. `check` is an extra validator on the parsed
value (e.g. the foreign-script rule): a message back means invalid, and it is retried like a schema error.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from lecture_copilot.config import OLLAMA_NUM_CTX

T = TypeVar("T", bound=BaseModel)
MAX_ATTEMPTS = 2


@dataclass
class JsonCall(Generic[T]):
    value: T | None = None
    attempts: int = 0
    first_valid: bool = False
    error: str | None = None
    raw: list[str] = field(default_factory=list)
    ms: list[float] = field(default_factory=list)
    load_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


async def chat_json(
    model: str,
    system: str,
    user: str,
    schema: type[T],
    *,
    client: httpx.AsyncClient,
    think: bool | None = None,
    options: dict | None = None,
    check: Callable[[T], str | None] | None = None,
    keep_alive: str | None = None,
    timeout_s: float = 60.0,
    backoff_s: float = 1.0,
) -> JsonCall[T]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    call: JsonCall[T] = JsonCall()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        call.attempts = attempt
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"num_ctx": OLLAMA_NUM_CTX, **(options or {})},
        }
        if think is not None:
            body["think"] = think
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        t0 = time.perf_counter()
        try:
            resp = await client.post("/api/chat", json=body, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message") if isinstance(data, dict) else None
            if not isinstance(message, dict):
                raise ValueError(f"malformed response body: {str(data)[:80]}")
        except (httpx.HTTPError, ValueError) as e:
            call.ms.append((time.perf_counter() - t0) * 1000)
            call.error = f"{type(e).__name__}: {e}"
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(backoff_s)
            continue
        call.ms.append((time.perf_counter() - t0) * 1000)
        if not call.raw:  # first answered attempt: the one that paid for loading the model
            call.load_ms = (data.get("load_duration") or 0) / 1e6
        call.tokens_in += data.get("prompt_eval_count") or 0
        call.tokens_out += data.get("eval_count") or 0
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        call.raw.append(content)
        try:
            value = schema.model_validate_json(content)
            problem = check(value) if check else None
        except ValidationError as e:
            problem = str(e)
        if problem:
            call.error = f"invalid output: {problem}"
            messages = [*messages, {"role": "assistant", "content": content},
                        {"role": "user", "content": f"Your output was invalid:\n{problem}\nReturn ONLY valid JSON "
                                                    "matching the schema."}]
            continue
        call.value = value
        call.first_valid = attempt == 1
        call.error = None
        return call
    return call
