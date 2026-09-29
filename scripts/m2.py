"""M2 measurements. Numbers land in eval/m2.json; model outputs stay in runs/m2/ (gitignored: course material).

    python scripts/m2.py extract-compare runs/<lecture_id> --prompts extract_v0 extract_v1
        the same transcripts (mw JSON next to each chunk) through both prompts: counts by kind
    python scripts/m2.py digest-compare <lecture_id> --map digest_sections_v0 digest_sections_v1 \\
        --exec digest_exec_v0 digest_exec_v1                    the same rows through both Digest prompt pairs
    python scripts/m2.py sample-digest                         a made-up lecture, every section filled (screenshots)
    python scripts/m2.py report --page docs/reports/M2_HE.html
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from lecture_copilot.agents.extractor import ExtractError, extract
from lecture_copilot.agents.schemas import Continuation, ExtractResult
from lecture_copilot.asr.macwhisper import parse_output
from lecture_copilot.audio.sources import AudioChunk
from lecture_copilot.config import DB_PATH, OLLAMA_URL, ROOT, VERIFY_MIN_IMPORTANCE, Profile
from lecture_copilot.metrics_page import fill_metrics
from lecture_copilot.output.digest import ClaimRow, ConceptRow, DigestDoc, TaskRow, digest, render_markdown
from lecture_copilot.output.sinks import FolderSink
from lecture_copilot.pipeline import Ctx
from lecture_copilot.stats import percentile
from lecture_copilot.store.db import Store, new_id

RESULTS = ROOT / "eval" / "m2.json"
WORK = ROOT / "runs" / "m2"
LECTURER = ("המרצה", "המרצים")


# ---------- pure helpers (tested) ----------

def count_extraction(results: list[ExtractResult | None]) -> dict:
    ok = [r for r in results if r is not None]
    kinds = [it.kind for r in ok for it in r.items]
    return {"chunks": len(results), "failed": len(results) - len(ok),
            "concepts": sum(len(r.concepts) for r in ok), "claims": sum(len(r.claims) for r in ok),
            "claims_flagged": sum(c.importance >= VERIFY_MIN_IMPORTANCE for r in ok for c in r.claims),
            "highlights": kinds.count("highlight"), "questions": kinds.count("question"),
            "actions": kinds.count("action"), "chunks_with_a_claim": sum(bool(r.claims) for r in ok)}


def lecturer_mentions(texts: list[str]) -> int:
    return sum(any(w in t for w in LECTURER) for t in texts)


# ---------- providers (run for real, not in tests) ----------

def _save(key: str, value: object) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")


async def _extract_all(prompt: str, chunk_dir: Path, client: httpx.AsyncClient) -> tuple[list, list[float]]:
    """A scratch store per prompt, so `previous_chunk_summary` comes from the same prompt's own output."""
    WORK.mkdir(parents=True, exist_ok=True)
    db = WORK / f"{prompt}.sqlite"
    db.unlink(missing_ok=True)
    store = Store(db)
    course = store.upsert_course("AI Developers — Python", language="he")
    lid = store.upsert_lecture(course, audio_path=str(chunk_dir), source="file", title="Tirgul fixture (10 min)",
                               date="2026-06-19", fact_check=False)
    ctx = Ctx(lecture_id=lid, course_id=course, course_name="AI Developers — Python",
              lecture_title="Tirgul fixture (10 min)", profile=Profile(fact_check=False, language="he"),
              store=store, asr=None, ollama=client, run_id=new_id())
    results, secs, t0 = [], [], 0.0
    for idx, out in enumerate(sorted(chunk_dir.glob("chunk_*.json")), 1):
        segments = parse_output(out)
        t1 = t0 + (segments[-1].t1 if segments else 0)
        chunk = AudioChunk(lid, idx, out, t0, t1)
        t = time.perf_counter()
        try:
            r = await extract(chunk, segments, ctx, f"{ctx.run_id}#{idx:04d}", prompt=prompt)
        except ExtractError:
            r = None
        secs.append(round(time.perf_counter() - t, 2))
        store.write_chunk(lid, idx, [s.model_copy(update={"t0": s.t0 + t0, "t1": s.t1 + t0}) for s in segments],
                          asr="mw", result=r)
        results.append(r)
        t0 = t1
        print(f"{prompt} chunk {idx:04d}: {secs[-1]} s")
    store.close()
    with (WORK / f"{prompt}.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r.model_dump() if r else None, ensure_ascii=False) + "\n" for r in results)
    return results, secs


def cmd_extract_compare(a: argparse.Namespace) -> None:
    async def go() -> dict:
        out = {}
        async with httpx.AsyncClient(base_url=OLLAMA_URL) as client:
            for prompt in a.prompts:
                results, secs = await _extract_all(prompt, Path(a.dir), client)
                ok = [r for r in results if r]
                out[prompt] = {**count_extraction(results), "p50_s": percentile(secs, 50),
                               "p95_s": percentile(secs, 95),
                               "summaries_naming_the_lecturer": lecturer_mentions([r.chunk_summary for r in ok])}
        return out
    res = asyncio.run(go())
    _save("extract_compare", res)
    print(json.dumps(res, indent=2))


def cmd_digest_compare(a: argparse.Namespace) -> None:
    async def go() -> dict:
        out = {}
        store = Store(Path(a.db))
        async with httpx.AsyncClient(base_url=OLLAMA_URL) as client:
            for mp, ep in zip(a.map, a.exec, strict=True):
                t = time.perf_counter()
                doc = await digest(a.lecture, store=store, client=client, map_prompt=mp, exec_prompt=ep, save=False)
                WORK.mkdir(parents=True, exist_ok=True)
                (WORK / f"digest_{mp}.md").write_text(render_markdown(doc), encoding="utf-8", newline="")
                texts = doc.full_summary + doc.exec_summary
                out[mp] = {"exec_prompt": ep, "seconds": round(time.perf_counter() - t, 1), "degraded": doc.degraded,
                           "full_summary_words": sum(len(p.split()) for p in doc.full_summary),
                           "sentences_naming_the_lecturer": sum(
                               lecturer_mentions([s]) for t_ in texts for s in t_.split(". "))}
        store.close()
        return out
    res = asyncio.run(go())
    _save(a.key, res)
    print(json.dumps(res, indent=2))


def sample_doc() -> DigestDoc:
    """A made-up lecture (the spec's example course) with every section filled: what the report's screenshots
    show. Real Digests are course material and never go into the repository."""
    return DigestDoc(
        lecture_id="SAMPLE", course_name="יזמות וחדשנות", title="מודלים עסקיים ב'", date="2026-11-04", week=5,
        minutes=88, language="he",
        exec_summary=[
            "מודל עסקי בריא מחזיר את עלות רכישת הלקוח (CAC) בתוך 12 חודשים לכל היותר.",
            "היחס LTV/CAC צריך להיות לפחות 3; מתחת ל-1 העסק מפסיד על כל לקוח חדש.",
            "Churn חודשי של 5% מוחק כמעט חצי מהלקוחות בשנה, ולכן שימור קודם לגיוס.",
            "במודל Freemium רק 2%–5% מהמשתמשים משלמים, והמחיר צריך לכסות את כל השאר.",
            "לפני שמגדילים תקציב שיווק בודקים את ה-Unit Economics של לקוח אחד.",
        ],
        full_summary=[
            "עלות רכישת לקוח (CAC) היא סך הוצאות השיווק והמכירות חלקי מספר הלקוחות החדשים באותה תקופה. "
            "ערך חיי הלקוח (LTV) הוא ההכנסה הממוצעת מלקוח כפול משך הזמן שהוא נשאר. "
            "היחס בין השניים קובע אם צמיחה יוצרת ערך או שורפת כסף.",
            "במודל מנויים, Churn הוא שיעור הלקוחות שעוזבים בחודש. Churn של 5% נשמע קטן, אבל בחישוב שנתי "
            "נשארים רק כ-54% מהלקוחות. לכן שיפור של נקודת אחוז אחת בשימור שווה יותר מקמפיין גיוס.",
            "במודל Freemium רוב המשתמשים לא משלמים לעולם. הדוגמה מהשיעור: 100,000 משתמשים, 3% משלמים "
            "39 ש\"ח בחודש, כלומר הכנסה חודשית של 117,000 ש\"ח שצריכה לממן את כולם.",
        ],
        highlights=["ההגדרה של CAC ושל LTV, כולל הנוסחאות — זה במבחן.",
                    "לזכור: Churn חודשי ו-Churn שנתי אינם אותו מספר."],
        prev_title="W04 · מודלים עסקיים א'",
        continuation=Continuation(new=["LTV", "Churn", "Freemium"], repeated=["CAC", "Unit Economics"],
                                  contradicts=["ב-W04 נאמר שתקופת ההחזר המקובלת היא 18 חודשים; היום 12"]),
        concepts=[ConceptRow("CAC", "עלות רכישת לקוח: הוצאות שיווק ומכירות חלקי מספר הלקוחות החדשים.", "cac"),
                  ConceptRow("LTV", "ערך חיי לקוח: כמה הכנסה מביא לקוח ממוצע לאורך כל התקופה שלו.", "ltv"),
                  ConceptRow("Churn", "שיעור הלקוחות שעוזבים בתקופה נתונה.", "churn"),
                  ConceptRow("Freemium", "מוצר בסיסי חינם, ותשלום על יכולות מתקדמות.", "freemium"),
                  ConceptRow("תקופת החזר", "כמה חודשים עוברים עד שלקוח מחזיר את עלות הרכישה שלו.", "payback")],
        claims=[ClaimRow("Dropbox הגיעה ל-4% משלמים במודל Freemium", 90, "pending", "עדיין לא נבדק"),
                ClaimRow("Netflix איבדה מיליון מנויים ברבעון אחד ב-2022", 85, "unchecked", "לא נבדק — אין רשת"),
                ClaimRow("יחס LTV/CAC של 3 הוא הסטנדרט בתעשיית ה-SaaS", 75, "pending", "עדיין לא נבדק")],
        all_claims=[], questions=["איך מחשבים LTV כשעדיין אין נתוני Churn של שנה שלמה?",
                                  "האם CAC כולל גם את המשכורות של צוות המכירות?"],
        tasks=[TaskRow("לחשב CAC ו-LTV למיזם שלי לפי הגיליון מהמודל", None, "2026-11-11"),
               TaskRow("לקרוא את פרק 4 ב-Business Model Generation", None, None)],
        notes=["לשאול את המרצה על Churn שלילי.", "הדוגמה של Freemium מתאימה לפרויקט הגמר."],
        segments=[{"t0": 0.0, "t1": 8.0, "text": "ערב טוב, היום נמשיך במודלים עסקיים.", "speaker": None}])


def cmd_sample_digest(_: argparse.Namespace) -> None:
    folder = FolderSink(WORK / "sample").write_lecture(sample_doc())
    print(f"sample digest written ({len(list(folder.iterdir()))} files)")


def cmd_report(a: argparse.Namespace) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    page = Path(a.page)
    page.write_text(fill_metrics(page.read_text(encoding="utf-8"), data), encoding="utf-8", newline="")
    print(f"filled {page}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="m2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract-compare")
    e.add_argument("dir")
    e.add_argument("--prompts", nargs="+", required=True)
    e.set_defaults(fn=cmd_extract_compare)
    d = sub.add_parser("digest-compare")
    d.add_argument("lecture")
    d.add_argument("--map", nargs="+", required=True)
    d.add_argument("--exec", nargs="+", required=True)
    d.add_argument("--db", default=str(DB_PATH))
    d.add_argument("--key", default="digest_compare")
    d.set_defaults(fn=cmd_digest_compare)
    sub.add_parser("sample-digest").set_defaults(fn=cmd_sample_digest)
    r = sub.add_parser("report")
    r.add_argument("--page", required=True)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
