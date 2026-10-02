""""copilot": the one command for a lecture day (DESIGN_HE §flow 1). Starts Ollama the way the project needs it,
checks MacWhisper's CLI and the Gemini key, serves the page and opens it in the browser. "Ready" or "fix this"."""

import json
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


def main(open_browser: bool = True, port: int = PORT, db: Path | None = None, courses_root: Path | None = None) -> int:
    import uvicorn

    from lecture_copilot.web.app import create_app
    from lecture_copilot.web.session import Session

    ensure_ollama()
    checks = readiness()
    for line in checks["fix"]:
        print(f"fix: {line}", file=sys.stderr)
    if not checks["ready"]:
        return 1
    kwargs = {k: v for k, v in (("db", db), ("courses_root", courses_root)) if v is not None}
    session = Session(**kwargs)
    session.checks = checks
    app = create_app(session)
    url = f"http://127.0.0.1:{port}/"
    print(f"ready · ollama {checks['ollama']} · {checks['mw']} · fact-checking {'on' if checks['gemini'] else 'off'}"
          f" · {url}")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


def mic_report(levels_db: list[float], block_s: float) -> dict:
    """From per-block levels: the peak, and when the meter first passed −40 dB (the spec: within 10 s)."""
    first = next((i for i, v in enumerate(levels_db) if v > -40), None)
    return {"peak_db": round(max(levels_db), 1) if levels_db else None,
            "seconds_to_minus40": round((first + 1) * block_s, 2) if first is not None else None,
            "seconds": round(len(levels_db) * block_s, 2), "readable": first is not None}


def miccheck(seconds: float = 10.0, out: Path = ROOT / "runs" / "m5" / "mic_check.json") -> dict:
    """Ten seconds from the microphone, from Dor's own Terminal (where macOS shows the permission prompt)."""
    import numpy as np
    import sounddevice as sd
    block = 0.25
    levels: list[float] = []

    def cb(indata, frames, t, status):
        rms = float(np.sqrt(np.mean(np.square(indata[:, 0]))))
        levels.append(20 * float(np.log10(rms + 1e-12)))
    with sd.InputStream(samplerate=16000, channels=1, blocksize=int(16000 * block), callback=cb):
        sd.sleep(int(seconds * 1000))
    report = mic_report(levels, block)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return report
