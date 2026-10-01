"""FastAPI + one page over a WebSocket (PLAN.md §3.9). The page hydrates from /api/state (SQLite through the
session) and refetches when the socket says "changed"; it never keeps truth of its own."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from lecture_copilot.web.session import Session

INDEX = Path(__file__).with_name("index.html")


class RecordBody(BaseModel):
    course: str
    title: str
    fact_check: bool = True


class ReplayBody(BaseModel):
    course: str
    title: str
    file: str
    pace: str = "fast"
    fact_check: bool = True


class NoteBody(BaseModel):
    text: str


class ResumeBody(BaseModel):
    lecture_id: str


def create_app(session: Session) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        session.close()

    app = FastAPI(title="Lecture Copilot", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX.read_text(encoding="utf-8")

    @app.get("/api/state")
    async def state() -> dict:
        return session.state()

    @app.post("/api/record")
    async def record(body: RecordBody):
        try:
            lecture_id = await session.start(body.course, body.title, body.fact_check, source="mic")
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return {"lecture_id": lecture_id}

    @app.post("/api/replay")
    async def replay(body: ReplayBody):
        file = Path(body.file).expanduser()
        if not file.is_file():
            return JSONResponse({"error": "file not found"}, status_code=400)
        kind = "transcript" if file.suffix.lower() in (".vtt", ".json") else "file"
        try:
            lecture_id = await session.start(body.course, body.title, body.fact_check, source=kind, file=file,
                                             pace=body.pace)
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return {"lecture_id": lecture_id}

    @app.post("/api/stop")
    async def stop():
        try:
            await session.stop()
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return {"ok": True}

    @app.post("/api/resume")
    async def resume(body: ResumeBody):
        try:
            await session.resume(body.lecture_id)
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        return {"ok": True}

    @app.post("/api/mark")
    async def mark() -> dict:
        return {"ok": session.mark()}

    @app.post("/api/note")
    async def note(body: NoteBody) -> dict:
        return {"ok": session.note(body.text)}

    @app.get("/api/recap")
    async def recap(minutes: int = 5) -> dict:
        return await session.recap(minutes)

    @app.get("/api/search")
    async def search(q: str = "") -> list[dict]:
        return await session.search(q)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        session.listeners.add(queue)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    cur = session.current
                    event = {"event": "tick", "level": getattr(cur.live, "level", None) if cur else None,
                             "elapsed_s": round(__import__("time").time() - cur.started) if cur else None}
                await socket.send_text(json.dumps(event))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            session.listeners.discard(queue)

    return app
