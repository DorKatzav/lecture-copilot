"""Sinks (PLAN.md §3.7): where a finished Digest lands. FolderSink writes plain files into the course folder that
Google Drive for desktop syncs — no OAuth, no API. Another destination (Notion, M6) is another Sink.

    COURSES_ROOT/<course>/W05_2026-11-04_<title>/{digest.md, digest.html, transcript.txt, claims.json}
    COURSES_ROOT/<course>/index.md
"""

import json
import re
from pathlib import Path
from typing import Protocol

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from lecture_copilot.config import COURSES_ROOT
from lecture_copilot.output.digest import DigestDoc, render_markdown

_UNSAFE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
_html = Environment(loader=PackageLoader("lecture_copilot.output", "templates"), undefined=StrictUndefined,
                    autoescape=select_autoescape(default=True, default_for_string=True))
CLAIM_FIELDS = ("text", "normalized", "importance", "status", "verdict", "confidence")


class Sink(Protocol):
    def write_lecture(self, doc: DigestDoc) -> Path: ...


def safe_name(name: str) -> str:
    """A file or folder name that survives Drive, macOS and a phone: no separators, no leading dots."""
    clean = " ".join(_UNSAFE.sub("-", name).split()).strip(" .-")
    return clean or "untitled"


def folder_name(doc: DigestDoc) -> str:
    week = f"W{doc.week:02d}_" if doc.week is not None else ""
    return f"{week}{doc.date}_{safe_name(doc.title).replace(' ', '-')}"


def short_date(doc: DigestDoc) -> str:
    return doc.heading.rsplit(" · ", 1)[-1]


def render_html(doc: DigestDoc) -> str:
    return _html.get_template("digest.html.j2").render(doc=doc, date=short_date(doc))


def render_transcript(doc: DigestDoc) -> str:
    lines = []
    for s in doc.segments:
        t = int(s["t0"])
        who = f"{s['speaker']}: " if s.get("speaker") else ""
        lines.append(f"[{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}] {who}{s['text']}")
    return "\n".join(lines) + ("\n" if lines else "")


def render_claims(doc: DigestDoc) -> str:
    rows = [{**{k: c.get(k) for k in CLAIM_FIELDS}, "sources": json.loads(c.get("sources_json") or "[]")}
            for c in doc.all_claims]
    return json.dumps(rows, ensure_ascii=False, indent=2) + "\n"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


class FolderSink:
    def __init__(self, root: Path = COURSES_ROOT):
        self.root = Path(root)

    def write_lecture(self, doc: DigestDoc) -> Path:
        if not self.root.parent.is_dir():
            raise OSError(f"{self.root.parent} does not exist — is Google Drive for desktop installed? "
                          "(COURSES_ROOT sets the course folder)")
        course = self.root / safe_name(doc.course_name)
        folder = course / folder_name(doc)
        folder.mkdir(parents=True, exist_ok=True)
        _write(folder / "digest.md", render_markdown(doc))
        _write(folder / "digest.html", render_html(doc))
        _write(folder / "transcript.txt", render_transcript(doc))
        _write(folder / "claims.json", render_claims(doc))
        self._write_index(course, doc.course_name)
        return folder

    def _write_index(self, course: Path, name: str) -> None:
        """Rebuilt from the folders that exist, newest first; the link text is each Digest's own heading."""
        lines = []
        for digest in sorted(course.glob("*/digest.md"), key=lambda p: p.parent.name.split("_", 1)[-1]
                             if p.parent.name.startswith("W") else p.parent.name, reverse=True):
            heading = digest.read_text(encoding="utf-8").splitlines()[0].removeprefix("# ")
            lines.append(f"- [{heading.removeprefix(name + ' — ')}](<{digest.parent.name}/digest.md>)")
        _write(course / "index.md", "\n".join([f"# {name}", "", *lines]) + "\n")
