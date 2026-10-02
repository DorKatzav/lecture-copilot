""""copilot": the one command for a lecture day (DESIGN_HE §flow 1). Starts Ollama the way the project needs it,
checks MacWhisper's CLI and the Gemini key, serves the page and opens it in the browser. "Ready" or "fix this"."""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import httpx

from lecture_copilot.config import MW_BIN, OLLAMA_URL, ROOT, load_env

PORT = 8770


def readiness(*, ollama_client: httpx.Client | None = None, env_file: Path = ROOT / ".env",
              mw_bin: str = MW_BIN) -> dict:
    out: dict = {"ollama": None, "mw": None, "gemini": False, "ready": False, "fix": []}
    client = ollama_client or httpx.Client(base_url=OLLAMA_URL, timeout=3)
    try:
        out["ollama"] = client.get("/api/version").json().get("version")
    except (httpx.HTTPError, ValueError):
        out["fix"].append("Ollama is not running — scripts/ollama_serve.sh")
    try:
        p = subprocess.run([mw_bin, "version"], capture_output=True, text=True, timeout=20)
        out["mw"] = (p.stdout.strip().splitlines() or [""])[0] if p.returncode == 0 else None
        if p.returncode != 0:
            out["fix"].append(f"mw version failed: {(p.stderr or p.stdout).strip()[:80]}")
    except (FileNotFoundError, subprocess.SubprocessError):
        out["fix"].append("mw not on PATH — MacWhisper → Settings → Advanced → Install CLI")
    try:
        load_env(env_file, fact_check=False)
    except RuntimeError:
        pass
    out["gemini"] = bool(os.environ.get("GEMINI_API_KEY"))
    if not out["gemini"]:
        out["fix"].append("no GEMINI_API_KEY in .env — fact-checking will be off")
    out["ready"] = out["ollama"] is not None and out["mw"] is not None
    return out


def ensure_ollama() -> bool:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=2)
        return True
    except httpx.HTTPError:
        p = subprocess.run([str(ROOT / "scripts" / "ollama_serve.sh")], capture_output=True, text=True)
        return p.returncode == 0


def main(open_browser: bool = True, port: int = PORT) -> int:
    import uvicorn

    from lecture_copilot.web.app import create_app
    from lecture_copilot.web.session import Session

    ensure_ollama()
    checks = readiness()
    for line in checks["fix"]:
        print(f"fix: {line}", file=sys.stderr)
    if not checks["ready"]:
        return 1
    session = Session()
    session.checks = checks
    app = create_app(session)
    url = f"http://127.0.0.1:{port}/"
    print(f"ready · ollama {checks['ollama']} · {checks['mw']} · fact-checking {'on' if checks['gemini'] else 'off'}"
          f" · {url}")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0
