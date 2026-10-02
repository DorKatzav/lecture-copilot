"""M6 measurements: what a Notion sync costs in calls, bytes and seconds — from `decisions`, never typed.

    python scripts/m6.py sync-stats [--lecture ID]       # the latest synced lecture → eval/m6.json
    python scripts/m6.py report --page docs/reports/M6_HE.html
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from lecture_copilot.config import DB_PATH, ROOT
from lecture_copilot.metrics_page import fill_metrics

RESULTS = ROOT / "eval" / "m6.json"


def sync_stats(con: sqlite3.Connection, lecture_id: str) -> dict:
    """Notion calls of one lecture's sync (net rows with ref `notion:<lecture>`), and the sink rows' seconds."""
    calls, by_path = [], {}
    for ms, out in con.execute("select ms, output_json from decisions where node = 'net' and lecture_id = ? "
                               "and input_ref like 'notion:%'", (lecture_id,)):
        o = json.loads(out)
        calls.append((ms, o))
        if o.get("status") == "ok":
            key = f"{o.get('method', '?')} {o.get('path', '?')}"
            by_path[key] = by_path.get(key, 0) + 1

    def sink_s(ref_suffix: str) -> float:
        row = con.execute("select ms from decisions where node = 'sink' and output_json like '%\"NotionSink\"%' "
                          "and input_ref like ? and (lecture_id = ? or lecture_id is null) order by ts desc limit 1",
                          (f"%{ref_suffix}", lecture_id)).fetchone()
        return round(row[0] / 1000, 2) if row else 0.0
    syncs = [{"status": json.loads(o)["status"], "calls": json.loads(o).get("calls", 0), "s": round(ms / 1000, 2)}
             for ms, o in con.execute("select ms, output_json from decisions where node = 'sink' and lecture_id = ? "
                                      "and output_json like '%\"NotionSink\"%' and input_ref like '%:write_lecture' "
                                      "order by ts", (lecture_id,))]
    return {"lecture_id": lecture_id, "calls": len(calls), "ok": sum(o.get("status") == "ok" for _, o in calls),
            "failed": sum(o.get("status") != "ok" for _, o in calls),
            "bytes_out_kb": round(sum(o.get("bytes_out", 0) for _, o in calls) / 1024, 1),
            "bytes_in_kb": round(sum(o.get("bytes_in", 0) for _, o in calls) / 1024, 1),
            "net_s": round(sum(ms for ms, _ in calls) / 1000, 2), "sink_s": sink_s(":write_lecture"),
            "course_s": sink_s(":write_course"), "by_path": by_path,
            "cost_usd": round(sum(o.get("cost_usd") or 0 for _, o in calls), 4), "syncs": syncs}


def _save(key: str, value: object) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")


def cmd_sync_stats(a: argparse.Namespace) -> None:
    con = sqlite3.connect(a.db)
    lid = a.lecture or con.execute(
        "select lecture_id from decisions where node = 'sink' and output_json like '%\"NotionSink\"%' and "
        "output_json like '%\"ok\"%' and input_ref like '%:write_lecture' order by ts desc limit 1").fetchone()[0]
    stats = sync_stats(con, lid)
    rows = {k: con.execute("select count(*) from items where lecture_id = ? and kind = ?", (lid, k)).fetchone()[0]
            for k in ("concept", "action", "decision")}
    stats["rows"] = {"concepts": rows["concept"], "tasks": rows["action"] + rows["decision"],
                     "claims": con.execute("select count(*) from claims where lecture_id = ? and importance >= 70",
                                           (lid,)).fetchone()[0]}
    _save("sync", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_report(a: argparse.Namespace) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    page = Path(a.page)
    page.write_text(fill_metrics(page.read_text(encoding="utf-8"), data), encoding="utf-8", newline="")
    print(f"filled {page}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="m6")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync-stats")
    s.add_argument("--lecture")
    s.add_argument("--db", type=Path, default=DB_PATH)
    s.set_defaults(fn=cmd_sync_stats)
    r = sub.add_parser("report")
    r.add_argument("--page", required=True)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
