"""The minimal eval (DESIGN_HE §eval, PLAN M4): does it work well enough to trust in class?

    python -m lecture_copilot.cli eval          → eval/results.json

Verdict accuracy on the benchmark claims (real ones + injected errors), Precision@5 of "material" on the ranked
claims of each labelled lecture, one cache-hit probe, and the cost. The verifier runs on a scratch copy of the
database, so the eval never touches real rows; the benchmark's own course memory is what the claims are checked
against. `labels_by` says whose labels the numbers rest on.
"""

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from lecture_copilot.agents.ranker import rank
from lecture_copilot.agents.schemas import Claim, ExtractResult
from lecture_copilot.agents.verifier import Verifier, VerifierBackend
from lecture_copilot.asr.base import Segment
from lecture_copilot.config import ROOT, VERIFIER_MODEL
from lecture_copilot.store.db import Store
from lecture_copilot.store.net import Net

BENCHMARK = ROOT / "eval" / "benchmark.json"
RESULTS = ROOT / "eval" / "results.json"
VERDICTS = ("correct", "incorrect", "imprecise", "unverifiable")


def score_verdicts(labels: list[dict], predictions: list[str]) -> dict:
    confusion: dict[str, dict[str, int]] = {v: {} for v in VERDICTS}
    hits = 0
    injected = caught = 0
    for lab, pred in zip(labels, predictions, strict=True):
        confusion[lab["verdict"]][pred] = confusion[lab["verdict"]].get(pred, 0) + 1
        hits += lab["verdict"] == pred
        if lab.get("injected"):
            injected += 1
            caught += pred in ("incorrect", "imprecise")
    return {"n": len(labels), "accuracy": round(hits / len(labels), 3) if labels else 0.0,
            "injected": injected, "injected_caught": caught,
            "confusion": {k: v for k, v in confusion.items() if v}}


def precision_at_k(ranked_texts: list[str], material: dict[str, bool], k: int = 5) -> float:
    """Share of the top k that is labelled material (over fewer than k when the list is shorter)."""
    top = ranked_texts[:k]
    return round(sum(bool(material.get(t)) for t in top) / len(top), 3) if top else 0.0


async def run_eval(db: Path, benchmark: Path, backend: VerifierBackend, *, results_path: Path = RESULTS,
                   work: Path = ROOT / "runs" / "eval") -> dict:
    bench = json.loads(benchmark.read_text(encoding="utf-8"))
    work.mkdir(parents=True, exist_ok=True)
    scratch = work / "eval.sqlite"
    shutil.copy(db, scratch)
    store = Store(scratch)
    t0 = time.perf_counter()
    try:
        course = store.upsert_course("AI Developers — Python", language="he")
        # the benchmark claims live in one scratch lecture dated after the real ones, so course memory applies
        lid = store.upsert_lecture(course, audio_path="eval:benchmark", source="transcript", title="eval",
                                   date="2026-06-10", fact_check=True)
        res = ExtractResult(chunk_summary="eval", items=[], concepts=[], claims=[
            Claim(text=c["text"], normalized=c["normalized"], importance=90) for c in bench["claims"]])
        store.write_chunk(lid, 1, [Segment(t0=0, t1=1, text="eval")], asr="transcript", result=res)
        verifier = Verifier(store, Net(store), backend)
        by_text = {c["text"]: c["id"] for c in store.claims(lid)}
        predictions = []
        for c in bench["claims"]:
            v = await verifier.verify(by_text[c["text"]])
            predictions.append(v.verdict if v else "unchecked")
        # cache: the first claim again must not reach the network
        calls_before = store.con.execute("select count(*) from decisions where node = 'net'").fetchone()[0]
        await verifier.verify(by_text[bench["claims"][0]["text"]])
        calls_after = store.con.execute("select count(*) from decisions where node = 'net'").fetchone()[0]
        p5 = {}
        for date, labels in bench.get("material", {}).items():
            row = store.con.execute("select id from lectures where date = ? and id != ? order by started_at desc "
                                    "limit 1", (date, lid)).fetchone()
            if row:
                p5[date] = precision_at_k([r.text for r in rank(row[0], store)], labels)
        cost = store.con.execute("select coalesce(sum(cost_usd), 0) from decisions where node = 'net' "
                                 "and lecture_id = ?", (lid,)).fetchone()[0]
        out = {"when": datetime.now(UTC).isoformat(timespec="seconds"), "labels_by": bench.get("labels_by", "?"),
               "model": VERIFIER_MODEL, "verdicts": score_verdicts(bench["claims"], predictions),
               "unchecked": predictions.count("unchecked"), "precision_at_5": p5,
               "cache_hit": calls_after == calls_before, "cost_usd": round(float(cost), 4),
               "seconds": round(time.perf_counter() - t0, 1),
               "per_claim": [{"text": c["text"][:60], "label": c["verdict"], "predicted": p, "injected": c["injected"]}
                             for c, p in zip(bench["claims"], predictions, strict=True)]}
    finally:
        store.close()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    return out
