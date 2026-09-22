import asyncio
import json

import httpx

from lecture_copilot.agents.schemas import ExtractResult
from lecture_copilot.llm import chat_json
from tests.stubs import FakeOllama

GOOD = json.dumps({"chunk_summary": "סיכום.", "concepts": [], "claims": [], "items": []})
BAD_RANGE = json.dumps({"chunk_summary": "x", "concepts": [], "items": [],
                        "claims": [{"text": "t", "normalized": "n", "importance": 400}]})


def run(fake: FakeOllama, **kw):
    async def go():
        async with fake.async_client() as client:
            return await chat_json("qwen3:8b", "SYS", "USER", ExtractResult, client=client, backoff_s=0, **kw)
    return asyncio.run(go())


def test_valid_first_attempt():
    fake = FakeOllama([GOOD])
    call = run(fake)
    assert call.value.chunk_summary == "סיכום."
    assert call.attempts == 1 and call.first_valid and call.error is None


def test_request_is_schema_constrained_with_explicit_context():
    fake = FakeOllama([GOOD])
    run(fake, think=False)
    req = fake.requests[0]
    assert req["format"] == ExtractResult.model_json_schema()
    assert req["options"]["num_ctx"] == 4096
    assert req["think"] is False and req["stream"] is False
    assert [m["role"] for m in req["messages"]] == ["system", "user"]


def test_think_omitted_when_not_given():
    fake = FakeOllama([GOOD])
    run(fake)
    assert "think" not in fake.requests[0]


def test_invalid_then_valid_retries_once_with_the_error():
    fake = FakeOllama([BAD_RANGE, GOOD])
    call = run(fake)
    assert call.value is not None and call.attempts == 2 and not call.first_valid
    retry_msgs = fake.requests[1]["messages"]
    assert retry_msgs[-2] == {"role": "assistant", "content": BAD_RANGE}
    assert "importance" in retry_msgs[-1]["content"]  # the validation error is appended


def test_not_json_twice_returns_error_without_raising():
    fake = FakeOllama(["Sure! Here is the JSON:", "{broken"])
    call = run(fake)
    assert call.value is None and call.attempts == 2 and call.error


def test_http_error_then_success():
    fake = FakeOllama([500, GOOD])
    call = run(fake)
    assert call.value is not None and call.attempts == 2


def test_connection_refused_twice_is_a_result_not_an_exception():
    fake = FakeOllama([httpx.ConnectError("refused"), httpx.ConnectError("refused")])
    call = run(fake)
    assert call.value is None and "refused" in call.error


def test_tokens_and_timings_are_recorded():
    fake = FakeOllama([GOOD])
    call = run(fake)
    assert (call.tokens_in, call.tokens_out) == (100, 20)
    assert call.load_ms == 5.0 and len(call.ms) == 1
