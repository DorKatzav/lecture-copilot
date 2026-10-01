import asyncio
import json

import pytest

from lecture_copilot import config
from lecture_copilot.store.db import Store
from lecture_copilot.store.net import Net, NetError, Offline


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    yield s
    s.close()


def test_every_outbound_call_is_logged_with_host_bytes_and_cost(store):
    net = Net(store)

    async def fake_call():
        return {"ok": True}, {"bytes_out": 820, "bytes_in": 1400, "tokens_in": 170, "tokens_out": 150,
                              "cost_usd": 0.00069}
    out = asyncio.run(net.call("generativelanguage.googleapis.com", fake_call, lecture_id="L", ref="C1"))
    assert out == {"ok": True}
    (row,) = store.con.execute("select * from decisions where node = 'net'").fetchall()
    o = json.loads(row["output_json"])
    assert o["host"] == "generativelanguage.googleapis.com" and o["bytes_out"] == 820 and o["status"] == "ok"
    assert row["cost_usd"] == pytest.approx(0.00069) and row["tokens_in"] == 170 and row["input_ref"] == "C1"


def test_a_connection_failure_is_offline_and_logged(store):
    net = Net(store)

    async def dead():
        raise ConnectionError("Network is unreachable")
    with pytest.raises(Offline):
        asyncio.run(net.call("h", dead, lecture_id="L", ref="C1"))
    o = json.loads(store.con.execute("select output_json from decisions where node = 'net'").fetchone()[0])
    assert o["status"] == "offline" and "unreachable" in o["error"]


def test_other_failures_are_net_errors_after_one_retry(store):
    net = Net(store, backoff_s=0)
    calls = []

    async def flaky():
        calls.append(1)
        raise RuntimeError("429 quota")
    with pytest.raises(NetError, match="quota"):
        asyncio.run(net.call("h", flaky, lecture_id="L", ref="C1"))
    assert len(calls) == 2
    rows = store.con.execute("select output_json from decisions where node = 'net'").fetchall()
    assert len(rows) == 1 and json.loads(rows[0][0])["attempts"] == 2


def test_cost_per_lecture_is_summed(store):
    net = Net(store)

    async def one():
        return None, {"bytes_out": 1, "bytes_in": 1, "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.01}
    for _ in range(3):
        asyncio.run(net.call("h", one, lecture_id="L", ref="x"))
    asyncio.run(net.call("h", one, lecture_id="OTHER", ref="x"))
    assert net.cost("L") == pytest.approx(0.03) and net.calls("L") == 3


def test_gemini_cost_from_tokens():
    from lecture_copilot.store.net import gemini_cost
    assert gemini_cost(1_000_000, 0) == pytest.approx(config.GEMINI_PRICE_PER_M["in"])
    assert gemini_cost(170, 150) == pytest.approx(170 * 0.75e-6 + 150 * 3.75e-6)


# ---------- .env ----------

def test_load_env_reads_the_file_without_overriding_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="k1"\n# comment\nNOTION_TOKEN=t1\nEMPTY=\n', encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("NOTION_TOKEN", "already")
    loaded = config.load_env(env)
    assert loaded == {"GEMINI_API_KEY", "EMPTY"}
    import os
    assert os.environ["GEMINI_API_KEY"] == "k1" and os.environ["NOTION_TOKEN"] == "already"


def test_load_env_requires_the_gemini_key_only_when_fact_checking(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config.load_env(tmp_path / "missing.env", fact_check=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        config.load_env(tmp_path / "missing.env", fact_check=True)
