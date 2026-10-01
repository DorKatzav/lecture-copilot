"""The course page (DESIGN_HE §Digest: the Study Pack): a cumulative glossary (term · explanation · first seen),
every ★, every flagged claim with its verdict, open questions and tasks — templates over SQLite, no LLM. The
search box on the page filters what is on it; the live search over the same memory is the web page's."""

import json
from dataclasses import dataclass, field

from lecture_copilot.agents.ranker import LABELS, rank
from lecture_copilot.config import VERIFY_MIN_IMPORTANCE
from lecture_copilot.output.digest import lecture_label
from lecture_copilot.output.sinks import _html  # the same environment and bidi filters as digest.html
from lecture_copilot.store.db import Store


@dataclass(frozen=True)
class GlossaryRow:
    term: str
    explanation: str
    key: str
    first_seen: str
    lecture_id: str


@dataclass(frozen=True)
class Starred:
    text: str
    lecture: str
    kind: str          # highlight (the lecturer) | user (★ pressed in class)


@dataclass(frozen=True)
class CourseClaim:
    text: str
    lecture: str
    importance: int
    verdict: str | None
    verdict_he: str
    explanation: str | None
    sources: list[str]


@dataclass(frozen=True)
class CourseItem:
    text: str
    lecture: str
    due: str | None = None
    owner: str | None = None


@dataclass
class LectureRef:
    id: str
    label: str
    date: str
    status: str
    folder: str = ""     # set by the sink from the lecture folders it finds


@dataclass
class CoursePage:
    course_id: str
    course_name: str
    lectures: list[LectureRef] = field(default_factory=list)
    glossary: list[GlossaryRow] = field(default_factory=list)
    highlights: list[Starred] = field(default_factory=list)
    claims: list[CourseClaim] = field(default_factory=list)
    questions: list[CourseItem] = field(default_factory=list)
    tasks: list[CourseItem] = field(default_factory=list)
    notes: list[CourseItem] = field(default_factory=list)


def _week(lec: dict) -> str:
    return f"W{lec['week']:02d}" if lec["week"] is not None else lec["title"]


def course_page(course_id: str, store: Store) -> CoursePage:
    course = store.course(course_id)
    page = CoursePage(course_id=course_id, course_name=course["name"])
    lecture_ids = store.course_lectures(course_id)
    lectures = {lid: store.lecture(lid) for lid in lecture_ids}
    page.lectures = [LectureRef(lid, lecture_label(lectures[lid]), lectures[lid]["date"], lectures[lid]["status"])
                     for lid in reversed(lecture_ids)]
    seen: dict[str, GlossaryRow] = {}
    for lid in lecture_ids:                      # chronological: the first explanation wins
        week = _week(lectures[lid])
        for it in store.items(lid):
            if it["kind"] == "concept":
                key = (it["canonical_key"] or it["text"]).strip().lower()
                if key not in seen:
                    first = it["first_seen_lecture_id"] or lid
                    seen[key] = GlossaryRow(it["text"], it["explanation"] or "", key,
                                            _week(lectures.get(first) or lectures[lid]), first)
            elif it["kind"] == "highlight":
                page.highlights.append(Starred(it["text"], week, "user" if it["owner"] == "user" else "highlight"))
            elif it["kind"] == "question":
                page.questions.append(CourseItem(it["text"], week))
            elif it["kind"] in ("action", "decision"):
                page.tasks.append(CourseItem(it["text"], week, it["due"], it["owner"]))
            elif it["kind"] == "note":
                page.notes.append(CourseItem(it["text"], week))
        for c in rank(lid, store):
            if c.importance >= VERIFY_MIN_IMPORTANCE:
                page.claims.append(CourseClaim(c.text, week, c.importance, c.verdict, c.verdict_he, c.explanation,
                                               c.sources))
    page.glossary = sorted(seen.values(), key=lambda g: g.term.lower())
    return page


def render_course_html(page: CoursePage) -> str:
    data = {"glossary": [g.__dict__ for g in page.glossary], "highlights": [h.__dict__ for h in page.highlights],
            "claims": [c.__dict__ for c in page.claims], "questions": [q.__dict__ for q in page.questions],
            "tasks": [t.__dict__ for t in page.tasks], "notes": [n.__dict__ for n in page.notes]}
    return _html.get_template("course.html.j2").render(page=page, data_json=json.dumps(data, ensure_ascii=False),
                                                       labels=LABELS)
