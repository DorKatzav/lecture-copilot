"""M1 measurements. Numbers land in eval/m1.json or come from db/copilot.sqlite; the Hebrew pages read them through
data-metric placeholders (`report`), never typed.

    python scripts/m1.py runner-footprint      # gemma3:12b runner memory: Ollama's default load vs use_mmap
    python scripts/m1.py report --page docs/notes/MEMORY_HE.html
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

from lecture_copilot.config import DB_PATH, LIVE_MODEL, OLLAMA_NUM_CTX, OLLAMA_URL, ROOT
from scripts.stage0 import fill_metrics

RESULTS = ROOT / "eval" / "m1.json"
FIXTURE = ROOT / "eval" / "fixture_10min.m4a"
UNIT_MB = {"B": 1 / 2**20, "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0, "K": 1 / 1024, "M": 1.0, "G": 1024.0}


# ---------- pure helpers (tested) ----------

def parse_footprint(text: str) -> dict:
    """`footprint -p <pid>` → dirty MB per category, and their sum."""
    cats: dict[str, float] = {}
    for m in re.finditer(r"^\s*([\d.]+) (B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+\d+\s+(.+)$",
                         text, re.MULTILINE):
        name = m.group(3).strip()
        name = re.sub(r"^untagged \((.+)\)$", r"\1", name)
        cats[name] = cats.get(name, 0.0) + float(m.group(1)) * UNIT_MB[m.group(2)]
    return {"footprint_mb": round(sum(cats.values()), 1), "categories": {k: round(v, 1) for k, v in cats.items()}}


def parse_vmmap_swapped_mb(text: str, region: str) -> float:
    """`vmmap --summary` row: REGION VIRTUAL RESIDENT DIRTY SWAPPED … → SWAPPED in MB."""
    m = re.search(rf"^{re.escape(region)}\s+([\d.]+[KMG])\s+([\d.]+[KMG])\s+([\d.]+[KMG])\s+([\d.]+)([KMG])",
                  text, re.MULTILINE)
    return round(float(m.group(4)) * UNIT_MB[m.group(5)], 1) if m else 0.0


def load_mode(command: str) -> str:
    m = re.search(r"--load-mode (\S+)", command)
    return m.group(1) if m else "mmap (default)"


def fixture_metrics(db: Path, fixture: Path) -> dict:
    import sqlite3
    con = sqlite3.connect(db)
    lids = [r[0] for r in con.execute("select id from lectures where source = 'file' and audio_path = ?",
                                      (str(fixture.resolve()),))]
    runs = [{"run_id": ref, **json.loads(o)} for lid in lids for ref, o in con.execute(
        "select input_ref, output_json from decisions where node = 'run' and lecture_id = ? order by ts", (lid,))]
    for r in runs:
        calls = [json.loads(o) for (o,) in con.execute(
            "select output_json from decisions where node = 'extractor' and input_ref like ?", (r["run_id"] + "#0%",))]
        r["extract"] = {"calls": len(calls), "first_valid": sum(bool(c.get("first_valid")) for c in calls),
                        "retries": sum(c.get("attempts", 1) - 1 for c in calls)}
    kinds = {k: n for lid in lids for k, n in con.execute(
        "select kind, count(*) from items where lecture_id = ? group by kind", (lid,))}
    return {"n_runs": len(runs), "first": runs[0] if runs else {}, "last": runs[-1] if runs else {},
            "item_kinds": kinds}


# ---------- providers (run for real, not in tests) ----------

def _sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def _runner_pid() -> str:
    """The llama-server that serves the vision-packaged gemma3 (the only runner started with --mmproj)."""
    out = subprocess.run(["pgrep", "-f", "llama-server.*--mmproj"], capture_output=True, text=True).stdout.split()
    if not out:
        raise RuntimeError("no gemma3 runner found")
    return out[0]


def _unload(client: httpx.Client) -> None:
    client.post("/api/generate", json={"model": LIVE_MODEL, "keep_alive": 0})
    for _ in range(30):
        if not any(m["name"].startswith(LIVE_MODEL) for m in client.get("/api/ps").json()["models"]):
            return
        time.sleep(1)


def cmd_runner_footprint(_: argparse.Namespace) -> None:
    res = {}
    with httpx.Client(base_url=OLLAMA_URL, timeout=300) as client:
        for variant, extra in (("default", {}), ("use_mmap", {"use_mmap": True})):
            _unload(client)
            time.sleep(2)
            client.post("/api/generate", json={"model": LIVE_MODEL, "prompt": "Reply with OK.", "stream": False,
                                               "keep_alive": "5m", "options": {"num_ctx": OLLAMA_NUM_CTX, **extra}})
            time.sleep(3)
            pid = _runner_pid()
            fp = parse_footprint(_sh("footprint", "-p", pid))
            res[variant] = {"load_mode": load_mode(_sh("ps", "-o", "command=", "-p", pid)),
                            "footprint_gb": round(fp["footprint_mb"] / 1024, 1),
                            "malloc_large_gb": round(fp["categories"].get("MALLOC_LARGE", 0) / 1024, 1),
                            "vm_allocate_gb": round(fp["categories"].get("VM_ALLOCATE", 0) / 1024, 1),
                            "malloc_large_swapped_gb": round(parse_vmmap_swapped_mb(
                                _sh("vmmap", "--summary", pid), "MALLOC_LARGE") / 1024, 1)}
            print(variant, res[variant])
        _unload(client)  # the next replay loads it with the configured options
    res["ollama_version"] = _sh("ollama", "--version").strip().splitlines()[-1].split()[-1]
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data["runner"] = res
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")


def cmd_report(a: argparse.Namespace) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data["fixture"] = fixture_metrics(DB_PATH, FIXTURE)
    page = Path(a.page)
    page.write_text(fill_metrics(page.read_text(encoding="utf-8"), data), encoding="utf-8", newline="")
    print(f"filled {page}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="m1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runner-footprint").set_defaults(fn=cmd_runner_footprint)
    r = sub.add_parser("report")
    r.add_argument("--page", required=True)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
