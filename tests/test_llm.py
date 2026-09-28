import asyncio
import json

import httpx
import pytest

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


def test_load_ms_kept_when_the_first_attempt_is_an_http_error():
    fake = FakeOllama([500, GOOD])
    assert run(fake).load_ms == 5.0


@pytest.mark.parametrize("body", [
    [],                                                        # list body
    {"message": None},                                         # message: null
    {"message": {"content": None}},                            # content: null
    {"message": {"content": GOOD}, "load_duration": None,      # null counters on an otherwise good reply
     "prompt_eval_count": None, "eval_count": None},
])
def test_malformed_response_shapes_never_raise(body):
    fake = FakeOllama([body, body])
    call = run(fake)
    if call.value is None:
        assert call.attempts == 2 and call.error
    else:
        assert call.tokens_in == 0 and call.load_ms == 0.0


def test_extra_check_failure_retries_with_its_message():
    fake = FakeOllama([GOOD, GOOD])
    seen = []

    def check(v):
        seen.append(v)
        return "chunk_summary contains Cyrillic letters" if len(seen) == 1 else None

    call = run(fake, check=check)
    assert call.value is not None and call.attempts == 2 and not call.first_valid
    assert "Cyrillic" in fake.requests[1]["messages"][-1]["content"]


def test_extra_check_failing_twice_is_an_error_not_a_value():
    fake = FakeOllama([GOOD, GOOD])
    call = run(fake, check=lambda v: "contains Arabic letters")
    assert call.value is None and call.attempts == 2 and "Arabic" in call.error


def test_keep_alive_is_sent_only_when_given():
    fake = FakeOllama([GOOD, GOOD])
    run(fake, keep_alive="30m")
    run(fake)
    assert fake.requests[0]["keep_alive"] == "30m" and "keep_alive" not in fake.requests[1]
