"""Fakes for every provider. Tests never call MacWhisper, Ollama, Gemini or Notion."""

import json

import httpx

from lecture_copilot.config import OLLAMA_URL


class FakeOllama:
    """Scripted Ollama server. Each reply is a str (assistant content), an int (HTTP status), an exception, or a
    dict/list sent verbatim as the response body (malformed shapes)."""

    def __init__(self, replies=(), models=("qwen3:8b", "gemma3:12b", "bge-m3:latest")):
        self.replies = list(replies)
        self.models = list(models)
        self.requests: list[dict] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
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
