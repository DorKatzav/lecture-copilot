"""M7: the first lecture, rehearsed. Numbers land in eval/m7.json; the checklist page reads them.

    python scripts/m7.py cold-boot          # kill Ollama, start `copilot`, time the "ready" line, walk the checklist
    python scripts/m7.py report --page docs/notes/FIRST_LECTURE_HE.html
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from lecture_copilot.config import ROOT
from lecture_copilot.metrics_page import fill_metrics

RESULTS = ROOT / "eval" / "m7.json"
WORK = ROOT / "runs" / "m7"
DEMO = ROOT / "runs" / "m5" / "demo" / "demo_lecture.vtt"
PORT = 8772

# The checklist, in order. `who` says whether `copilot` does it (and the walk measures it) or Dor does it by hand.
STEPS = [
    {"id": "power", "who": "dor", "he": "המחשב מחובר לחשמל (או מעל 60% סוללה) ו-wifi דולק"},
    {"id": "seat", "who": "dor", "he": "יושבים במושב הרגיל, עד 5 מטר מהמרצה; בלי מיקרופון חיצוני"},
    {"id": "mic", "who": "dor", "he": "פעם אחת: `miccheck` מ-Terminal — macOS מאשר את המיקרופון"},
    {"id": "ready", "who": "copilot", "he": "`copilot` מדפיס \"ready\" ופותח את הדף (מהפעלה קרה: Ollama עולה לבד)"},
    {"id": "page", "who": "copilot",
     "he": "הדף אומר \"מוכן\" ושורת המצב מראה Ollama, MacWhisper, בדיקת עובדות, Notion"},
    {"id": "course", "who": "dor", "he": "בוחרים קורס (או מקלידים חדש) ושם להרצאה; מקטינים את החלון"},
    {"id": "start", "who": "copilot", "he": "\"הקלט\" — הסטטוס עובר ל\"מקליט\", המד זז"},
    {"id": "first_chunk", "who": "copilot", "he": "הקטע הראשון מגיע ללוח (תקציב 30 שנ' לקטע)"},
    {"id": "mark", "who": "copilot", "he": "\"★ סמן את זה\" — ההדגשה מופיעה בלוח"},
    {"id": "note", "who": "copilot", "he": "הערה עם חותמת זמן"},
    {"id": "recap", "who": "copilot", "he": "\"מה פספסתי\" עונה בעברית"},
    {"id": "stop", "who": "copilot", "he": "\"סיום\" — Digest תוך שתי דקות"},
    {"id": "folder", "who": "copilot",
     "he": "תיקיית ההרצאה: digest.md, digest.html, transcript.txt, claims.json; course.html של הקורס"},
    {"id": "notion", "who": "copilot", "he": "עמוד ההרצאה ב-Notion (או דילוג עם סיבה כשאין טוקן)"},
    {"id": "cost", "who": "copilot", "he": "עלות ההרצאה מתחת ל-$0.20; שום אודיו לא יצא"},
    {"id": "views", "who": "dor", "he": "פעם אחת: תצוגות ב-Notion (גלריה, ★, פסיקה ≠ נכון, לוח משימות)"},
]


def is_ready_line(line: str) -> bool:
    return line.strip().startswith("ready ·")


def walk_summary(steps: list[dict]) -> dict:
    failed = [s["id"] for s in steps if s["status"] == "failed"]
    by = {s["id"]: s for s in steps}
    return {"n": len(steps), "ok": sum(s["status"] == "ok" for s in steps),
            "skipped": sum(s["status"] == "skipped" for s in steps), "failed": len(failed), "failed_ids": failed,
            "all_ok": not failed, "first_chunk_s": by.get("first_chunk", {}).get("s"),
            "digest_s": by.get("stop", {}).get("s")}


# ---------- live ----------

def _kill_ollama() -> None:
    """A cold boot means no Ollama and no runner; verified, not assumed."""
    from lecture_copilot.config import OLLAMA_URL
    for _ in range(10):
        subprocess.run(["pkill", "-f", "^ollama serve"], capture_output=True)
        subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
        time.sleep(1)
        try:
            httpx.get(f"{OLLAMA_URL}/api/version", timeout=1)
        except httpx.HTTPError:
            return
    raise RuntimeError("ollama is still answering after pkill — not a cold boot")


def _wait(client: httpx.Client, pred, timeout_s: float, every: float = 0.5):
    """Polls /api/state; the server may still be coming up ("ready" is printed before uvicorn listens)."""
    t0 = time.perf_counter()
    s = {}
    while time.perf_counter() - t0 < timeout_s:
        try:
            s = client.get("/api/state").json()
        except httpx.TransportError:
            time.sleep(every)
            continue
        if pred(s):
            return s, round(time.perf_counter() - t0, 2)
        time.sleep(every)
    return s, None


def cmd_cold_boot(a: argparse.Namespace) -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "db").mkdir(parents=True)
    (WORK / "courses").mkdir()
    steps: list[dict] = []
    _kill_ollama()
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-B", "-m", "lecture_copilot.cli", "copilot", "--no-browser", "--port", str(PORT),
         "--db", str(WORK / "db" / "copilot.sqlite"), "--courses-root", str(WORK / "courses")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    ready_s, ready_line = None, ""
    try:
        for line in proc.stdout:
            if is_ready_line(line):
                ready_s, ready_line = round(time.perf_counter() - t0, 2), line.strip()
                break
            if line.startswith("fix:"):
                print(line.rstrip())
        steps.append({"id": "ready", "status": "ok" if ready_s is not None else "failed", "s": ready_s,
                      "detail": ready_line[:120]})
        if ready_s is None:
            return _finish(steps, proc, None)
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30) as c:
            s, page_s = _wait(c, lambda s: s.get("ready"), 15)
            checks = s.get("checks", {})
            steps.append({"id": "page", "status": "ok" if s.get("ready") and checks.get("ollama") else "failed",
                          "s": page_s, "detail": f"notion {checks.get('notion')} · gemini {checks.get('gemini')}"})
            r = c.post("/api/replay", json={"course": "הדגמה", "title": "הרצאת הדגמה", "file": str(DEMO),
                                            "pace": "realtime", "fact_check": True}).json()
            lecture_id = r.get("lecture_id")
            s, start_s = _wait(c, lambda s: (s.get("current") or {}).get("status") == "recording", 20)
            steps.append({"id": "start", "status": "ok" if start_s is not None else "failed", "s": start_s})
            s, first_s = _wait(c, lambda s: (s.get("current") or {}).get("chunks", 0) >= 1, 120)
            last = (s.get("current") or {}).get("last_chunk") or {}
            steps.append({"id": "first_chunk", "status": "ok" if first_s is not None and last.get("total_s", 99) <= 30
                          else "failed", "s": last.get("total_s"), "wall_s": first_s,
                          "detail": f"processing of chunk 1 (budget 30 s); wall-clock {first_s} s includes waiting "
                                    f"for {last.get('t1')} s of audio at real-time pace"})
            steps.append({"id": "mark", "status": "ok" if c.post("/api/mark").json().get("ok") else "failed"})
            steps.append({"id": "note", "status": "ok" if c.post("/api/note", json={"text": "לבדוק את זה אחר כך"})
                          .json().get("ok") else "failed"})
            rec = c.get("/api/recap", params={"minutes": 5}).json()
            steps.append({"id": "recap", "status": "ok" if rec.get("bullets") else "failed",
                          "detail": f"{len(rec.get('bullets', []))} bullets"})
            s, _ = _wait(c, lambda s: (s.get("current") or {}).get("chunks", 0) >= 3, 240)
            t_stop = time.perf_counter()
            c.post("/api/stop")
            s, _ = _wait(c, lambda s: (s.get("current") or {}).get("status") in ("digested", "failed"), 300)
            cur = s.get("current") or {}
            digest_s = round(time.perf_counter() - t_stop, 2)
            steps.append({"id": "stop", "status": "ok" if cur.get("status") == "digested" and digest_s <= 120 else
                          "failed", "s": digest_s, "detail": f"{cur.get('chunks')} chunks · {cur.get('status')}"})
            folder = Path((cur.get("digest") or {}).get("folder") or "")
            files = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
            course_html = folder.parent / "course.html" if folder.is_dir() else None
            steps.append({"id": "folder", "status": "ok" if {"digest.md", "digest.html", "transcript.txt",
                          "claims.json"} <= set(files) and course_html and course_html.is_file() else "failed",
                          "detail": ", ".join(files)})
            notion_url = (cur.get("digest") or {}).get("notion")
            import sqlite3
            con = sqlite3.connect(WORK / "db" / "copilot.sqlite")
            row = con.execute("select output_json from decisions where node = 'sink' and lecture_id = ? and "
                              "output_json like '%\"NotionSink\"%' order by ts desc limit 1", (lecture_id,)).fetchone()
            out = json.loads(row[0]) if row else {}
            steps.append({"id": "notion", "status": "ok" if notion_url else ("skipped" if out.get("status") == "skipped"
                          else "failed"), "detail": notion_url or out.get("reason") or out.get("error", "no sink row")})
            net = con.execute("select count(*), coalesce(sum(cost_usd), 0), group_concat(distinct json_extract("
                              "output_json, '$.host')) from decisions where node = 'net' and lecture_id = ?",
                              (lecture_id,)).fetchone()
            con.close()
            steps.append({"id": "cost", "status": "ok" if float(net[1]) < 0.20 else "failed",
                          "cost_usd": round(float(net[1]), 4), "calls": net[0], "hosts": net[2]})
    finally:
        _finish(steps, proc, ready_s)


def _finish(steps: list[dict], proc: subprocess.Popen, ready_s) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    data = {"ran_at": time.strftime("%Y-%m-%d %H:%M"), "cold_boot": {"ready_s": ready_s},
            "walk": {"steps": steps, **walk_summary(steps)},
            "manual": [s["id"] for s in STEPS if s["who"] == "dor"]}
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    for s in steps:
        print(f"{s['status']:8} {s['id']:12} {s.get('s', '')} {s.get('detail', '')}".rstrip())
    print(f"cold boot → ready {ready_s} s · walk {data['walk']['ok']}/{data['walk']['n']} ok")


def cmd_report(a: argparse.Namespace) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    page = Path(a.page)
    page.write_text(fill_metrics(page.read_text(encoding="utf-8"), data), encoding="utf-8", newline="")
    print(f"filled {page}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="m7")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cold-boot").set_defaults(fn=cmd_cold_boot)
    r = sub.add_parser("report")
    r.add_argument("--page", required=True)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
