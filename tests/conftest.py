"""Every test runs sealed off from Dor's real `.env` (2.10 incident: a `cli.main` test loaded the real file into
`os.environ`, and the web tests then wrote 42 rows into his real Notion). Two guards:

- the Notion / Gemini secrets are removed from the environment and `cli.ENV_FILE` points at a file that does not exist;
- `NotionAPI` without an injected client (= a real network client) raises. Tests always pass `FakeNotion`.
"""

import pytest

from lecture_copilot import cli
from lecture_copilot.output import notion

SECRETS = ("NOTION_TOKEN", "NOTION_ROOT_PAGE", "GEMINI_API_KEY", "COURSES_ROOT", *notion.ENV_KEYS.values())


@pytest.fixture(autouse=True)
def sealed_env(tmp_path, monkeypatch):
    for name in SECRETS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path / "no-such.env")
    real_init = notion.NotionAPI.__init__

    def guarded(self, token, *, client=None, timeout_s=30.0):
        if client is None:
            raise RuntimeError("a test tried to build a real Notion client — tests use tests.stubs.FakeNotion")
        real_init(self, token, client=client, timeout_s=timeout_s)
    monkeypatch.setattr(notion.NotionAPI, "__init__", guarded)
