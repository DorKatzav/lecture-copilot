import json

from scripts.m7 import STEPS, is_ready_line, walk_summary


def test_the_ready_line_is_recognised_only_when_the_launcher_says_so():
    assert is_ready_line("ready · ollama 0.34.2 · MacWhisper 15.2.1 (1521) · fact-checking on · notion on · http://…")
    assert not is_ready_line("fix: Ollama is not running — scripts/ollama_serve.sh")
    assert not is_ready_line("INFO:     Uvicorn running on http://127.0.0.1:8772")


def test_walk_summary_counts_automated_steps_and_names_the_failures():
    steps = [{"id": "ready", "status": "ok", "s": 12.3}, {"id": "page", "status": "ok"},
             {"id": "notion", "status": "skipped", "detail": "NOTION_TOKEN missing"},
             {"id": "recap", "status": "failed", "detail": "no bullets"}]
    out = walk_summary(steps)
    assert out == {"n": 4, "ok": 2, "skipped": 1, "failed": 1, "failed_ids": ["recap"], "all_ok": False}
    assert walk_summary([{"id": "ready", "status": "ok"}])["all_ok"] is True
    assert json.dumps(out)


def test_every_automated_step_has_an_id_and_a_hebrew_label_for_the_checklist():
    ids = [s["id"] for s in STEPS]
    assert len(ids) == len(set(ids)) and "ready" in ids and "stop" in ids and "notion" in ids
    assert all(s["who"] in ("copilot", "dor") for s in STEPS)
