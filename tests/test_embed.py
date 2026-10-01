import asyncio

import httpx
import pytest

from lecture_copilot.config import EMBED_MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_LOAD_OPTIONS
from lecture_copilot.store.embed import EmbedError, cosine, embed, pack, unpack
from tests.stubs import EMBED_DIMS, FakeOllama, fake_embedding


def run(fake, texts, **kw):
    async def go():
        async with fake.async_client() as client:
            return await embed(texts, client=client, **kw)
    return asyncio.run(go())


def test_one_vector_per_text_in_one_batch():
    fake = FakeOllama()
    vecs = run(fake, ["CAC", "LTV", "Churn"])
    assert len(vecs) == 3 and len(vecs[0]) == EMBED_DIMS and vecs[0] == fake_embedding("CAC")
    assert len(fake.embed_calls) == 1 and fake.loads[0][1]["model"] == EMBED_MODEL


def test_request_keeps_the_model_resident_with_the_load_options():
    fake = FakeOllama()
    run(fake, ["x"])
    body = fake.loads[0][1]
    assert body["keep_alive"] == OLLAMA_KEEP_ALIVE and body["options"] == OLLAMA_LOAD_OPTIONS


def test_empty_input_makes_no_call():
    fake = FakeOllama()
    assert run(fake, []) == [] and fake.embed_calls == []


def test_a_provider_failure_is_an_embed_error():
    fake = FakeOllama(fail_loads=True)
    with pytest.raises(EmbedError, match="500"):
        run(fake, ["x"], backoff_s=0)


def test_a_wrong_count_is_an_embed_error():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")

    async def go():
        async with client:
            await embed(["a", "b"], client=client, backoff_s=0)
    with pytest.raises(EmbedError, match="2"):
        asyncio.run(go())


def test_pack_and_unpack_round_trip():
    v = [0.25, -1.0, 3.5]
    assert unpack(pack(v)) == pytest.approx(v) and len(pack(v)) == 12


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0) and cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 0]) == 0.0
