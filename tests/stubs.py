"""Fakes for every provider. Tests never call MacWhisper, Ollama, Gemini or Notion."""

import json

import httpx

from lecture_copilot.config import OLLAMA_URL

EMBED_DIMS = 1024   # like bge-m3, so the real vector tables accept them


def fake_embedding(text: str, dims: int = EMBED_DIMS) -> list[float]:
    """A bag-of-words vector: texts that share words are similar, so tests can reason about nearest neighbours."""
    import hashlib
    import math
    v = [0.0] * dims
    for w in text.lower().split():
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dims] += 1.0 if (h >> 8) % 2 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeOllama:
    """Scripted Ollama server. Each reply is a str (assistant content), an int (HTTP status), an exception, or a
    dict/list sent verbatim as the response body (malformed shapes)."""

    def __init__(self, replies=(), models=("qwen3:8b", "gemma3:12b", "bge-m3:latest"), fail_loads=False):
        self.replies = list(replies)
        self.models = list(models)
        self.fail_loads = fail_loads
        self.requests: list[dict] = []
        self.loads: list[tuple[str, dict]] = []
        self.embed_calls: list[list[str]] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if request.url.path in ("/api/generate", "/api/embed"):  # model load / unload: never consumes a reply
            body = json.loads(request.content)
            self.loads.append((request.url.path, body))
            if self.fail_loads:
                return httpx.Response(500, json={"error": "scripted load failure"})
            if request.url.path == "/api/embed" and "input" in body:
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                self.embed_calls.append(inputs)
                return httpx.Response(200, json={"model": body["model"],
                                                 "embeddings": [fake_embedding(t) for t in inputs],
                                                 "prompt_eval_count": sum(len(t.split()) for t in inputs)})
            return httpx.Response(200, json={"done": True, "embeddings": [[0.0]]})
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "scripted failure"})
        if isinstance(reply, dict | list):
            return httpx.Response(200, json=reply)
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": reply},
            "prompt_eval_count": 100, "eval_count": 20, "load_duration": 5_000_000, "total_duration": 50_000_000,
        })

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle), base_url=OLLAMA_URL)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle), base_url=OLLAMA_URL)


FAKE_MW = '''#!{python}
import json, os, sys, time
mode = {mode!r}
args = sys.argv[1:]
open({log!r}, "a").write(json.dumps(args) + "\\n")
out = args[args.index("-o") + 1]
if mode == "hang":
    open({log!r} + ".pid", "w").write(str(os.getpid()))
    time.sleep(30)
if mode == "exit1":
    sys.stderr.write("Error: could not decode audio\\n")
    sys.exit(1)
if mode == "garbage":
    open(out, "w").write("not json")
if mode in ("ok", "silent"):
    segs = [] if mode == "silent" else [
        {{"id": 0, "start": 0, "end": 4200, "text": " שלום לכולם ", "words": []}},
        {{"id": 1, "start": 4200, "end": 9000, "text": "היום נדבר על CAC", "words": []}},
        {{"id": 2, "start": 9000, "end": 9500, "text": "  ", "words": []}}]
    json.dump({{"text": " ".join(s["text"] for s in segs), "segments": segs}}, open(out, "w"), ensure_ascii=False)
'''


def fake_mw(directory, mode="ok"):
    """A stand-in `mw` executable. mode: ok | silent | exit1 | hang | no_output | garbage.
    Every call's argv is appended to <directory>/mw_calls.jsonl."""
    import sys
    from pathlib import Path

    path = Path(directory) / "mw"
    log = str(Path(directory) / "mw_calls.jsonl")
    path.write_text(FAKE_MW.format(python=sys.executable, mode=mode, log=log), encoding="utf-8")
    path.chmod(0o755)
    return path


class FakeASR:
    """Scripted ASR: `script[idx]` is a list of Segments (chunk-relative) or an exception to raise."""

    name = "fake"

    def __init__(self, script=None, default=None):
        from lecture_copilot.asr.base import Segment
        self.script = dict(script or {})
        self.default = default if default is not None else [Segment(t0=0.0, t1=5.0, text="שלום, היום נדבר על CAC")]
        self.calls: list[tuple[str, str]] = []

    async def transcribe(self, wav, language):
        self.calls.append((str(wav), language))
        idx = int(str(wav).rsplit("_", 1)[-1].split(".")[0]) if "chunk_" in str(wav) else 0
        out = self.script.get(idx, self.default)
        if isinstance(out, Exception):
            raise out
        return out


class ListSource:
    """A ChunkSource over prepared chunks; `fail_after=n` raises after n chunks (a source that breaks)."""

    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after = list(chunks), fail_after

    async def __aiter__(self):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("ffmpeg could not decode lecture.m4a")
            yield c


class FakeGemini:
    """Scripted verifier backend. Each reply is a Verdict-shaped dict, or an exception to raise."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate(self, system: str, user: str) -> tuple[str, dict]:
        import asyncio
        import json
        self.calls.append((system, user))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            reply = self.replies.pop(0) if self.replies else {"verdict": "unverifiable", "confidence": 0.0,
                                                               "explanation": "אין מקור", "sources": []}
            if isinstance(reply, Exception):
                raise reply
            text = json.dumps(reply, ensure_ascii=False)
            return text, {"tokens_in": 170, "tokens_out": 150, "bytes_out": len(user.encode()),
                          "bytes_in": len(text.encode()), "grounding": reply.get("grounding", [])}
        finally:
            self.in_flight -= 1


class FakeNotion:
    """An in-memory Notion behind httpx: databases with one data source each, pages with properties and a
    markdown body, and the data-source query filters the sink uses (rich_text equals / starts_with).
    `fail_next` makes the next N requests return 500; `offline` raises a transport error on every request."""

    def __init__(self, root_page="root-page-id", offline=False):
        self.root_page = root_page
        self.offline = offline
        self.fail_next = 0
        self.databases: dict[str, dict] = {}       # data_source_id → {"title", "properties", "parent"}
        self.pages: dict[str, dict] = {}           # page_id → {"parent", "properties", "markdown", "icon"}
        self.requests: list[tuple[str, str, dict | None]] = []
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"{self._n:032x}"

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        if self.offline:
            raise httpx.ConnectError("offline", request=request)
        if self.fail_next:
            self.fail_next -= 1
            return httpx.Response(500, json={"object": "error", "message": "scripted failure"})
        path, parts = request.url.path, request.url.path.strip("/").split("/")
        if path == "/v1/databases" and request.method == "POST":
            ds = self._id()
            self.databases[ds] = {"title": body["title"][0]["text"]["content"], "parent": body["parent"],
                                  "properties": body["initial_data_source"]["properties"]}
            return httpx.Response(200, json={"object": "database", "id": self._id(),
                                             "data_sources": [{"id": ds, "name": self.databases[ds]["title"]}]})
        if parts[:2] == ["v1", "data_sources"] and len(parts) == 3 and request.method == "GET":
            if parts[2] not in self.databases:
                return httpx.Response(404, json={"object": "error", "message": "Could not find data source"})
            return httpx.Response(200, json={"object": "data_source", "id": parts[2]})
        if parts[:2] == ["v1", "data_sources"] and len(parts) == 4 and parts[3] == "query":
            rows = [self._page_json(pid) for pid, p in self.pages.items() if p["parent"] == parts[2]]
            f = (body or {}).get("filter")
            if f:
                cond, prop = f["rich_text"], f["property"]
                def hit(row):
                    text = "".join(t["plain_text"] for t in row["properties"].get(prop, {}).get("rich_text", []))
                    return text == cond["equals"] if "equals" in cond else text.startswith(cond["starts_with"])
                rows = [r for r in rows if hit(r)]
            return httpx.Response(200, json={"object": "list", "results": rows, "has_more": False,
                                             "next_cursor": None})
        if path == "/v1/pages" and request.method == "POST":
            parent = body["parent"].get("data_source_id") or body["parent"].get("page_id")
            if parent not in self.databases and parent != self.root_page:
                return httpx.Response(404, json={"object": "error", "message": "Could not find parent"})
            pid = self._id()
            self.pages[pid] = {"parent": parent, "properties": self._store_props(body.get("properties", {})),
                               "markdown": body.get("markdown", ""), "icon": body.get("icon")}
            return httpx.Response(200, json=self._page_json(pid))
        if parts[:2] == ["v1", "pages"] and len(parts) == 3:
            if parts[2] not in self.pages:
                return httpx.Response(404, json={"object": "error", "message": "Could not find page"})
            if request.method == "PATCH":
                self.pages[parts[2]]["properties"].update(self._store_props(body.get("properties", {})))
                if "icon" in body:
                    self.pages[parts[2]]["icon"] = body["icon"]
            return httpx.Response(200, json=self._page_json(parts[2]))
        if parts[:2] == ["v1", "pages"] and len(parts) == 4 and parts[3] == "markdown":
            if parts[2] not in self.pages:
                return httpx.Response(404, json={"object": "error", "message": "Could not find page"})
            if request.method == "PATCH":
                self.pages[parts[2]]["markdown"] = body["replace_content"]["new_str"]
            return httpx.Response(200, json={"object": "page_markdown", "id": parts[2],
                                             "markdown": self.pages[parts[2]]["markdown"], "truncated": False})
        return httpx.Response(404, json={"object": "error", "message": f"no route {request.method} {path}"})

    @staticmethod
    def _store_props(props: dict) -> dict:
        """Rich text and titles are kept as Notion returns them: lists of {plain_text}."""
        out = {}
        for name, v in props.items():
            v = dict(v)
            for kind in ("title", "rich_text"):
                if kind in v:
                    v[kind] = [{"plain_text": t["text"]["content"], "text": t["text"]} for t in v[kind]]
            out[name] = v
        return out

    def _page_json(self, pid: str) -> dict:
        p = self.pages[pid]
        return {"object": "page", "id": pid, "url": f"https://www.notion.so/{pid}", "properties": p["properties"],
                "parent": {"data_source_id": p["parent"]}, "icon": p["icon"]}

    def rows(self, data_source_id: str) -> list[dict]:
        return [p for p in self.pages.values() if p["parent"] == data_source_id]

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle), base_url="https://api.notion.com")
