"""NotionSink (PLAN.md §3.8, DESIGN_HE §Notion): the same Digest that lands in the course folder lands in Notion.

    "🎓 לימודים" (Dor's page, shared with the integration)
    ├── 🎓 קורסים   ├── 📚 הרצאות   ├── 📖 מילון   ├── 🔍 טענות   └── 📌 משימות

`notion_init` creates the five databases once (ids go to `.env`). `NotionSink.write_lecture` upserts one Courses
row, one Lectures row (body = the Digest as Notion-flavored markdown), and one row per concept / flagged claim /
task. Every row carries a hidden `key` (lecture_id · course:canonical_key · lecture:claim_id), so re-syncing the
same lecture updates in place. All calls go through `store/net.py` (host api.notion.com, cost 0).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from jinja2 import Environment, PackageLoader, StrictUndefined

from lecture_copilot.output.digest import DigestDoc
from lecture_copilot.store.net import Net, NetError, Offline

NOTION_HOST = "api.notion.com"
NOTION_VERSION = "2026-03-11"           # the first version with the markdown page endpoints (D-M6-1)
RICH_TEXT_MAX = 2000                    # Notion's limit per rich-text item
ENV_KEYS = {"courses": "NOTION_DS_COURSES", "lectures": "NOTION_DS_LECTURES", "glossary": "NOTION_DS_GLOSSARY",
            "claims": "NOTION_DS_CLAIMS", "tasks": "NOTION_DS_TASKS"}
STATUS_HE = {"recording": "מוקלט", "interrupted": "נקטע", "digesting": "מסכם", "digested": "סוכם"}
VERDICT_HE = {"correct": "נכון", "incorrect": "לא נכון", "imprecise": "לא מדויק", "unverifiable": "לא ניתן לאימות"}
UNCHECKED_HE = "לא נבדק"
_HEX32 = re.compile(r"[0-9a-f]{32}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_env = Environment(loader=PackageLoader("lecture_copilot.output", "templates"), undefined=StrictUndefined,
                   trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, autoescape=False)


class NotionError(RuntimeError):
    pass


# ---------- ids, .env ----------

def page_id_from(text: str) -> str | None:
    """A Notion page id from a pasted URL, a dashed uuid or a bare 32-hex id."""
    text = text.strip().lower()
    m = _UUID.search(text)
    if m:
        return m.group()
    found = _HEX32.findall(text.split("?")[0])
    if not found:
        return None
    h = found[-1]                                      # the last one in a URL is the page's
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def save_env_keys(path: Path, values: dict[str, str]) -> None:
    """Set keys in `.env`, in place when present, appended when not; other lines are untouched."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    todo = dict(values)
    for i, line in enumerate(lines):
        name = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and name in todo:
            lines[i] = f"{name}={todo.pop(name)}"
    lines += [f"{k}={v}" for k, v in todo.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


@dataclass(frozen=True)
class NotionIds:
    """Data-source ids of the five databases."""
    courses: str
    lectures: str
    glossary: str
    claims: str
    tasks: str

    @classmethod
    def from_env(cls, env) -> "NotionIds | None":
        vals = {k: env.get(v, "") for k, v in ENV_KEYS.items()}
        return cls(**vals) if all(vals.values()) else None

    def as_env(self) -> dict[str, str]:
        return {v: getattr(self, k) for k, v in ENV_KEYS.items()}


# ---------- the API ----------

class NotionAPI:
    """One method over httpx. The token lives only in the header; nothing of it reaches the log."""

    def __init__(self, token: str, *, client: httpx.AsyncClient | None = None, timeout_s: float = 30.0):
        self.client = client or httpx.AsyncClient(base_url=f"https://{NOTION_HOST}", timeout=timeout_s)
        self.client.headers.update({"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION})

    async def request(self, method: str, path: str, body: dict | None = None) -> tuple[dict, dict]:
        r = await self.client.request(method, path, json=body)
        usage = {"bytes_out": len(r.request.content or b""), "bytes_in": len(r.content), "cost_usd": 0.0,
                 "method": method, "path": _UUID.sub("…", _HEX32.sub("…", path))}
        if r.status_code >= 400:
            msg = r.json().get("message", r.text[:200]) if r.headers.get("content-type", "").startswith(
                "application/json") else r.text[:200]
            raise NotionError(f"{r.status_code} {method} {path}: {msg}")
        return r.json(), usage


# ---------- the five databases ----------

def _title(s: str) -> dict:
    return {"title": [{"text": {"content": s[:RICH_TEXT_MAX]}}]}


def _text(s: str | None) -> dict:
    return {"rich_text": [{"text": {"content": s[:RICH_TEXT_MAX]}}] if s else []}


def _select(name: str | None) -> dict:
    return {"select": {"name": name} if name else None}


def _relation(*ids: str | None) -> dict:
    return {"relation": [{"id": i} for i in ids if i]}


def _date(iso: str | None) -> dict:
    return {"date": {"start": iso} if iso else None}


def database_schemas(courses: str | None, lectures: str | None) -> dict[str, tuple[str, str, dict]]:
    """name → (title, icon, properties). Relations need the target data source, so Courses comes first,
    then Lectures, then the three that relate to both."""
    rel_course = {"relation": {"data_source_id": courses, "type": "single_property", "single_property": {}}}
    rel_lecture = {"relation": {"data_source_id": lectures, "type": "single_property", "single_property": {}}}
    key = {"rich_text": {}}
    return {
        "courses": ("🎓 קורסים", "🎓", {"שם": {"title": {}}, "שפה": {"select": {"options": [
            {"name": "he", "color": "blue"}, {"name": "en", "color": "purple"}]}},
            "תיקייה": {"rich_text": {}}, "key": key}),
        "lectures": ("📚 הרצאות", "📚", {"כותרת": {"title": {}}, "קורס": rel_course, "שבוע": {"number": {}},
                                        "תאריך": {"date": {}}, "סטטוס": {"select": {"options": [
                                            {"name": v, "color": "green" if k == "digested" else "yellow"}
                                            for k, v in STATUS_HE.items()]}},
                                        "סיכום מנהלים": {"rich_text": {}}, "טענות מהותיות": {"number": {}},
                                        "דקות": {"number": {}}, "key": key}),
        "glossary": ("📖 מילון", "📖", {"מושג": {"title": {}}, "הסבר": {"rich_text": {}}, "קורס": rel_course,
                                       "נראה לראשונה": rel_lecture, "canonical_key": {"rich_text": {}},
                                       "★": {"checkbox": {}}, "key": key}),
        "claims": ("🔍 טענות", "🔍", {"המרצה אמר": {"title": {}}, "בפועל": {"rich_text": {}},
                                     "פסיקה": {"select": {"options": [
                                         {"name": VERDICT_HE["correct"], "color": "green"},
                                         {"name": VERDICT_HE["incorrect"], "color": "red"},
                                         {"name": VERDICT_HE["imprecise"], "color": "orange"},
                                         {"name": VERDICT_HE["unverifiable"], "color": "gray"},
                                         {"name": UNCHECKED_HE, "color": "default"}]}},
                                     "מקור": {"url": {}}, "חשיבות": {"number": {}}, "קורס": rel_course,
                                     "הרצאה": rel_lecture, "key": key}),
        "tasks": ("📌 משימות", "📌", {"משימה": {"title": {}}, "תאריך יעד": {"date": {}}, "בוצע": {"checkbox": {}},
                                     "בעלים": {"rich_text": {}}, "קורס": rel_course, "הרצאה": rel_lecture,
                                     "key": key}),
    }


async def notion_init(api: NotionAPI, net: Net, root_page_id: str, existing: NotionIds | None = None) -> NotionIds:
    """Create the five databases under the root page. With `existing`, each data source that still answers is
    kept and only the missing ones are created (never duplicates a database)."""
    ids: dict[str, str] = {}
    if existing:
        for name in ENV_KEYS:
            ds = getattr(existing, name)
            try:
                await net.call(NOTION_HOST, lambda ds=ds: api.request("GET", f"/v1/data_sources/{ds}"),
                               lecture_id=None, ref=f"notion-init:{name}")
                ids[name] = ds
            except Offline:
                raise
            except NetError:
                pass                                  # deleted in Notion: created again below
    for name in ENV_KEYS:                                 # ordered: courses, lectures, then the rest
        if name in ids:
            continue
        title, icon, props = database_schemas(ids.get("courses"), ids.get("lectures"))[name]
        body = {"parent": {"type": "page_id", "page_id": root_page_id}, "icon": {"type": "emoji", "emoji": icon},
                "title": [{"text": {"content": title}}], "initial_data_source": {"properties": props}}
        res = await net.call(NOTION_HOST, lambda body=body: api.request("POST", "/v1/databases", body),
                             lecture_id=None, ref=f"notion-init:{name}")
        ids[name] = res["data_sources"][0]["id"]
    return NotionIds(**ids)


# ---------- the lecture page ----------

def render_notion_markdown(doc: DigestDoc) -> str:
    """The Digest in Notion-flavored markdown: a callout for the executive summary, a toggle for the full one,
    H2 per section, checkboxes for tasks. Sections are separated by `@@` in the template like digest.md.j2."""
    parts = _env.get_template("notion.md.j2").render(doc=doc).split("@@")
    return "\n\n".join(p.strip() for p in parts) + "\n"


def _plain(page: dict, prop: str) -> str:
    v = page.get("properties", {}).get(prop, {})
    return "".join(t.get("plain_text", "") for t in v.get("rich_text", v.get("title", [])))


class NotionSink:
    def __init__(self, api: NotionAPI, net: Net | None, ids: NotionIds):
        self.api, self.net, self.ids = api, net, ids
        self.lecture_id: str | None = None

    def bind(self, store) -> None:
        """Without a store at construction (`cli`), the sink logs through the store the Digest runs on."""
        if self.net is None:
            self.net = Net(store)

    # -- the one door --
    async def _call(self, method: str, path: str, body: dict | None, ref: str) -> dict:
        return await self.net.call(NOTION_HOST, lambda: self.api.request(method, path, body),
                                   lecture_id=self.lecture_id, ref=ref)

    async def _index(self, ds: str, prefix: str, ref: str) -> dict[str, dict]:
        """key → page, for every row of a data source whose key starts with `prefix` (one lecture, one course)."""
        out, cursor = {}, None
        while True:
            body = {"filter": {"property": "key", "rich_text": {"starts_with": prefix}}, "page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            res = await self._call("POST", f"/v1/data_sources/{ds}/query", body, ref)
            for page in res["results"]:
                out[_plain(page, "key")] = page
            if not res.get("has_more"):
                return out
            cursor = res["next_cursor"]

    async def _upsert(self, ds: str, index: dict[str, dict], key: str, props: dict, ref: str, *,
                      create_only: dict | None = None, markdown: str | None = None, icon: str | None = None) -> dict:
        """`create_only` properties are set when the row is born and never touched again (first_seen, done)."""
        props = {**props, "key": _text(key)}
        page = index.get(key)
        if page is None:
            body = {"parent": {"data_source_id": ds}, "properties": {**props, **(create_only or {})}}
            if markdown is not None:
                body["markdown"] = markdown
            if icon:
                body["icon"] = {"type": "emoji", "emoji": icon}
            page = await self._call("POST", "/v1/pages", body, ref)
            index[key] = page
            return page
        await self._call("PATCH", f"/v1/pages/{page['id']}", {"properties": props}, ref)
        if markdown is not None:
            await self._call("PATCH", f"/v1/pages/{page['id']}/markdown",
                             {"type": "replace_content", "replace_content": {"new_str": markdown,
                                                                             "allow_deleting_content": True}}, ref)
        return page

    # -- Sink --
    async def write_lecture(self, doc: DigestDoc) -> str:
        """Returns the lecture page's URL."""
        self.lecture_id = doc.lecture_id
        ref = f"notion:{doc.lecture_id}"
        courses = await self._index(self.ids.courses, doc.course_id, ref)
        course = await self._upsert(self.ids.courses, courses, doc.course_id, {
            "שם": _title(doc.course_name), "שפה": _select(doc.language)}, ref, icon="🎓")
        lectures = await self._index(self.ids.lectures, doc.lecture_id, ref)
        week = f"W{doc.week:02d} · " if doc.week is not None else ""
        lecture = await self._upsert(self.ids.lectures, lectures, doc.lecture_id, {
            "כותרת": _title(f"{week}{doc.title}"), "קורס": _relation(course["id"]), "שבוע": {"number": doc.week},
            "תאריך": _date(doc.date if _ISO_DATE.match(doc.date) else None), "סטטוס": _select(STATUS_HE["digested"]),
            "סיכום מנהלים": _text("\n".join(f"• {b}" for b in doc.exec_summary)),
            "טענות מהותיות": {"number": len(doc.claims)}, "דקות": {"number": doc.minutes}},
            ref, markdown=render_notion_markdown(doc), icon="📚")
        both = (course["id"], lecture["id"])

        glossary = await self._index(self.ids.glossary, f"{doc.course_id}:", ref)
        for c in doc.concepts:
            await self._upsert(self.ids.glossary, glossary, f"{doc.course_id}:{c.canonical_key}", {
                "מושג": _title(c.term), "הסבר": _text(c.explanation), "קורס": _relation(course["id"]),
                "canonical_key": _text(c.canonical_key),
                "★": {"checkbox": any(c.term in h for h in doc.highlights)}},
                ref, create_only={"נראה לראשונה": _relation(lecture["id"])})
        claims = await self._index(self.ids.claims, f"{doc.lecture_id}:", ref)
        for c in doc.claims:
            await self._upsert(self.ids.claims, claims, f"{doc.lecture_id}:{c.id}", {
                "המרצה אמר": _title(c.text), "בפועל": _text(c.explanation),
                "פסיקה": _select(VERDICT_HE.get(c.verdict or "", UNCHECKED_HE)),
                "מקור": {"url": c.sources[0] if c.sources else None}, "חשיבות": {"number": c.importance},
                "קורס": _relation(both[0]), "הרצאה": _relation(both[1])}, ref)
        tasks = await self._index(self.ids.tasks, f"{doc.lecture_id}:", ref)
        for t in doc.tasks:
            iso = t.due if t.due and _ISO_DATE.match(t.due) else None
            title = t.text if iso or not t.due else f"{t.text} (עד {t.due})"
            await self._upsert(self.ids.tasks, tasks, f"{doc.lecture_id}:{t.id}", {
                "משימה": _title(title), "תאריך יעד": _date(iso), "בעלים": _text(t.owner),
                "קורס": _relation(both[0]), "הרצאה": _relation(both[1])},
                ref, create_only={"בוצע": {"checkbox": False}})
        return lecture["url"]

    async def write_course(self, page) -> str:
        """The course row's body: the numbers line and links to the lecture pages (the rest are views)."""
        self.lecture_id = None
        ref = f"notion:course:{page.course_id}"
        courses = await self._index(self.ids.courses, page.course_id, ref)
        lectures = await self._index(self.ids.lectures, "", ref)
        links = [f"- <mention-page url=\"{p['url']}\">{_plain(p, 'כותרת')}</mention-page>"
                 for lec in page.lectures if (p := lectures.get(lec.id))]
        open_tasks = sum(1 for _ in page.tasks)
        md = "\n".join([f"**{len(page.lectures)} הרצאות · {len(page.glossary)} מושגים · {len(page.highlights)} ★ · "
                        f"{open_tasks} משימות**", "", "## הרצאות", *(links or ["אין עדיין."]), ""])
        row = await self._upsert(self.ids.courses, courses, page.course_id, {"שם": _title(page.course_name)},
                                 ref, markdown=md, icon="🎓")
        return row["url"]
