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
    assert r == {"ollama": "0.34.2", "mw": "MacWhisper 15.2.1 (1521)", "gemini": True, "ready": True, "fix": []}


def test_readiness_says_what_to_fix(monkeypatch, tmp_path):
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: (_ for _ in ()).throw(FileNotFoundError("mw")))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("down"))),
                          base_url="http://ollama")
    r = readiness(ollama_client=client, env_file=tmp_path / "none.env")
    assert r["ready"] is False and r["gemini"] is False
    assert any("ollama_serve.sh" in f for f in r["fix"]) and any("Install CLI" in f for f in r["fix"])
    assert any("GEMINI_API_KEY" in f for f in r["fix"])
