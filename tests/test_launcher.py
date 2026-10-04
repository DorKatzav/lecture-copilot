import httpx

from lecture_copilot.web.launcher import readiness


def test_readiness_reports_each_dependency(monkeypatch, tmp_path):
    def fake_mw(cmd, **kw):
        class P:
            returncode, stdout, stderr = 0, "MacWhisper 15.2.1 (1521)\n", ""
        return P()
    monkeypatch.setattr("subprocess.run", fake_mw)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"version": "0.34.2"})),
                          base_url="http://ollama")
    r = readiness(ollama_client=client, env_file=tmp_path / "none.env")
    assert r == {"ollama": "0.34.2", "mw": "MacWhisper 15.2.1 (1521)", "gemini": True, "notion": "off", "ready": True,
                 "fix": ["Notion is off — NOTION_TOKEN missing (.env); the folder is written as usual"]}


def test_readiness_says_what_to_fix(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: (_ for _ in ()).throw(FileNotFoundError("mw")))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("down"))),
                          base_url="http://ollama")
    r = readiness(ollama_client=client, env_file=tmp_path / "none.env")
    assert r["ready"] is False and r["gemini"] is False
    assert any("ollama_serve.sh" in f for f in r["fix"]) and any("Install CLI" in f for f in r["fix"])
    assert any("GEMINI_API_KEY" in f for f in r["fix"])


def test_readiness_tells_the_notion_state(monkeypatch, tmp_path):
    """M7: the checklist asks "is Notion on?" — the launcher answers, like it does for Gemini."""
    from lecture_copilot.output.notion import NotionIds
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: type("P", (), {"returncode": 0, "stdout": "mw 1\n",
                                                                           "stderr": ""})())
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"version": "0.34.2"})),
                          base_url="http://ollama")
    env = tmp_path / ".env"
    env.write_text("NOTION_TOKEN=t\n", encoding="utf-8")
    r = readiness(ollama_client=client, env_file=env)
    assert r["notion"] == "off" and any("notion-init" in f for f in r["fix"])
    monkeypatch.setenv("NOTION_TOKEN", "t")
    for k, v in NotionIds("c", "l", "g", "k", "t").as_env().items():
        monkeypatch.setenv(k, v)
    r = readiness(ollama_client=client, env_file=env)
    assert r["notion"] == "on" and not any("Notion" in f for f in r["fix"])
