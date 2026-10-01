# ruff: noqa: E501  (an HTML page lives inline)
"""M3 measurements and Dor's labelling aid. Numbers land in eval/m3.json.

    python scripts/m3.py candidates             # shared-concept candidates between the two transcript lectures →
                                                # runs/m3/SHARED_CONCEPTS_HE.html (for Dor) + eval/benchmark_candidates.json
    python scripts/m3.py lecture-stats <lecture_id> --key <key>
    python scripts/m3.py report --page docs/reports/M3_HE.html
"""

import argparse
import html as html_lib
import json
import sqlite3
import sys
from pathlib import Path

from lecture_copilot.config import ALREADY_SAID_COSINE, DB_PATH, ROOT
from lecture_copilot.metrics_page import fill_metrics
from lecture_copilot.store.db import Store
from lecture_copilot.store.embed import cosine, unpack

RESULTS = ROOT / "eval" / "m3.json"
WORK = ROOT / "runs" / "m3"
CANDIDATES_JSON = ROOT / "eval" / "benchmark_candidates.json"


# ---------- pure helpers (tested) ----------

def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


def candidate_pairs(store: Store, first: str, again: str) -> list[dict]:
    """Concepts of the later lecture that an earlier one may have explained: same key, same term, or close in
    meaning (vector cosine ≥ ALREADY_SAID_COSINE). `flagged` says what the memory decided during the replay."""
    earlier = store.items(first, kind="concept")
    by_key = {_norm(e["canonical_key"]): e for e in earlier if e["canonical_key"]}
    by_term = {_norm(e["text"]): e for e in earlier}
    vectors = [(e, unpack(e["embedding"])) for e in earlier if e["embedding"]]
    rows, seen = [], set()
    for r in store.items(again, kind="concept"):
        if _norm(r["text"]) in seen:
            continue
        match, how, sim = by_key.get(_norm(r["canonical_key"])), "key", None
        if match is None:
            match, how = by_term.get(_norm(r["text"])), "term"
        if match is None and r["embedding"] and vectors:
            v = unpack(r["embedding"])
            best = max(vectors, key=lambda ev: cosine(v, ev[1]))
            sim = cosine(v, best[1])
            if sim >= ALREADY_SAID_COSINE:
                match, how = best[0], "meaning"
        if match is None:
            continue
        seen.add(_norm(r["text"]))
        rows.append({"term": r["text"], "explanation": r["explanation"], "match": match["text"],
                     "match_explanation": match["explanation"], "how": how,
                     "similarity": round(sim, 3) if sim is not None else None,
                     "flagged": r["first_seen_lecture_id"] == first})
    return rows


def candidates_json(rows: list[dict], first: str, again: str) -> str:
    data = {"shared_concepts": [{"term": r["term"], "first": first, "again": again} for r in rows]}
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def memory_stats(db: Path, lecture_id: str) -> dict:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    run = con.execute("select output_json from decisions where node = 'run' and lecture_id = ? "
                      "order by ts desc limit 1", (lecture_id,)).fetchone()
    r = json.loads(run[0]) if run else {}
    concepts = con.execute("select count(*), sum(first_seen_lecture_id != ?) from items "
                           "where lecture_id = ? and kind = 'concept'", (lecture_id, lecture_id)).fetchone()
    claims = con.execute("select count(*), sum(contradicts_id is not null) from claims where lecture_id = ?",
                         (lecture_id,)).fetchone()
    return {"chunks": r.get("chunks"), "status": r.get("status"), "already_said": r.get("already_said"),
            "contradictions": r.get("contradictions"), "memory_p95_s": (r.get("timing") or {}).get(
                "memory_s", {}).get("p95"), "timing": r.get("timing"), "concepts": concepts[0],
            "returned": concepts[1] or 0, "claims": claims[0], "claims_linked": claims[1] or 0}


# ---------- commands ----------

def _save(key: str, value: object) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")


def cmd_candidates(a: argparse.Namespace) -> None:
    store = Store(Path(a.db))
    try:
        rows = store.con.execute("select id, title, date from lectures where source = 'transcript' "
                                 "order by date, started_at").fetchall()
        if len(rows) < 2:
            sys.exit("need two transcript lectures")
        first, again = rows[-2], rows[-1]
        pairs = candidate_pairs(store, first["id"], again["id"])
    finally:
        store.close()
    WORK.mkdir(parents=True, exist_ok=True)
    CANDIDATES_JSON.write_text(candidates_json(pairs, first["date"], again["date"]), encoding="utf-8", newline="")
    how_he = {"key": "אותו מפתח", "term": "אותו מונח", "meaning": "משמעות קרובה"}
    trs = "\n".join(
        f'<tr><td><bdi>{html_lib.escape(p["term"])}</bdi></td><td>{html_lib.escape(p["explanation"] or "")}</td>'
        f'<td><bdi>{html_lib.escape(p["match"])}</bdi></td><td>{html_lib.escape(p["match_explanation"] or "")}</td>'
        f'<td>{how_he[p["how"]]}</td><td>{"✓" if p["flagged"] else "—"}</td></tr>' for p in pairs)
    page = f'''<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>מושגים משותפים · לתיוג</title>
<style>body{{font-family:Assistant,system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:.95rem}}th,td{{border-bottom:1px solid #ccc;padding:6px 8px;text-align:right;vertical-align:top}}
th{{background:#eee}}bdi{{font-family:monospace}}</style></head><body>
<h1>מושגים משותפים בין {html_lib.escape(first["title"])} ל-{html_lib.escape(again["title"])}</h1>
<p>הטבלה מציעה {len(pairs)} זוגות שהזיכרון מצא. בבקשה עבור על הרשימה: מחק מ-<code>eval/benchmark_candidates.json</code>
כל שורה שאינה באמת אותו מושג, הוסף מושגים משותפים שחסרים (לפי המונח כפי שהוא מופיע ב-{html_lib.escape(again["title"])}),
ושמור את הקובץ בשם <code>eval/benchmark.json</code>. השער של M3 קורא אותו.</p>
<table><thead><tr><th>ב-{html_lib.escape(again["title"])}</th><th>הסבר</th><th>ב-{html_lib.escape(first["title"])}</th><th>הסבר</th><th>איך זוהה</th><th>סומן בהרצה</th></tr></thead>
<tbody>{trs}</tbody></table></body></html>
'''
    (WORK / "SHARED_CONCEPTS_HE.html").write_text(page, encoding="utf-8", newline="")
    flagged = sum(p["flagged"] for p in pairs)
    _save("candidates", {"pairs": len(pairs), "flagged": flagged,
                         "by_how": {h: sum(p["how"] == h for p in pairs) for h in ("key", "term", "meaning")}})
    print(f"{len(pairs)} candidate pairs ({flagged} flagged during the replay) → {WORK / 'SHARED_CONCEPTS_HE.html'}, "
          f"{CANDIDATES_JSON}")


def cmd_lecture_stats(a: argparse.Namespace) -> None:
    out = memory_stats(Path(a.db), a.lecture)
    _save(a.key, out)
    print(json.dumps({k: v for k, v in out.items() if k != "timing"}, indent=2))


def cmd_report(a: argparse.Namespace) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    page = Path(a.page)
    page.write_text(fill_metrics(page.read_text(encoding="utf-8"), data), encoding="utf-8", newline="")
    print(f"filled {page}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="m3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--db", default=str(DB_PATH))
    c.set_defaults(fn=cmd_candidates)
    ls = sub.add_parser("lecture-stats")
    ls.add_argument("lecture")
    ls.add_argument("--key", required=True)
    ls.add_argument("--db", default=str(DB_PATH))
    ls.set_defaults(fn=cmd_lecture_stats)
    r = sub.add_parser("report")
    r.add_argument("--page", required=True)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
