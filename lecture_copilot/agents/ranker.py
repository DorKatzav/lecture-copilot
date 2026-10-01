"""Ranker (PLAN.md §3.5): importance orders the dashboard and the Digest. It never alerts (DESIGN_HE: silence).

D-M4-3: the extractor's importance alone ranked worked examples and class logistics beside real rules (eval:
Precision@5 40% / 20% on my labels). Two demotions: a claim the verifier found unverifiable (opinion, forecast,
class logistics), and a claim phrased in the first or second person — about the class, not the world.
"""

import re
from dataclasses import dataclass, field

from lecture_copilot.config import MATERIAL_MIN_IMPORTANCE
from lecture_copilot.store.db import Store

LABELS = {"correct": "נכון", "incorrect": "לא נכון", "imprecise": "לא מדויק", "unverifiable": "לא ניתן לאימות",
          "pending": "עדיין לא נבדק", "unchecked": "לא נבדק — אין רשת", "skipped": "לא נבדק"}


UNVERIFIABLE_PENALTY, PERSONAL_PENALTY = 30, 25
_PERSONAL = re.compile(r"(^|\s)(אני|שלי|לי|אתם|לכם|שלכם|אתכם|אנחנו|שלנו|המטרה שלי|אני רוצה|תעשו)(\s|$|[,.:])")


def score(importance: int, verdict: str | None, text: str) -> int:
    s = importance
    if verdict == "unverifiable":
        s -= UNVERIFIABLE_PENALTY
    if _PERSONAL.search(text):
        s -= PERSONAL_PENALTY
    return s


@dataclass(frozen=True)
class RankedClaim:
    id: str
    text: str
    importance: int
    status: str
    label: str                       # material | minor
    verdict: str | None = None
    verdict_he: str = ""
    confidence: float | None = None
    explanation: str | None = None
    sources: list[str] = field(default_factory=list)
    contradicts_id: str | None = None
    score: int = 0


def rank(lecture_id: str, store: Store) -> list[RankedClaim]:
    import json
    out = []
    for c in store.claims(lecture_id):
        verdict = c["verdict"] if c["status"] == "verified" else None
        out.append(RankedClaim(
            id=c["id"], text=c["text"], importance=c["importance"] or 0, status=c["status"],
            label="material" if (c["importance"] or 0) >= MATERIAL_MIN_IMPORTANCE else "minor",
            verdict=verdict, verdict_he=LABELS.get(verdict or c["status"], c["status"]),
            confidence=c["confidence"], explanation=c["explanation"],
            sources=json.loads(c["sources_json"]) if c["sources_json"] else [], contradicts_id=c["contradicts_id"],
            score=score(c["importance"] or 0, verdict, c["text"])))
    return sorted(out, key=lambda r: (-r.score, -r.importance))
