import pytest

from lecture_copilot import prompts

EXTRACT_VARS = {
    "language", "course_name", "lecture_title", "idx", "t0", "t1", "known_terms", "previous_chunk_summary", "text",
}


def test_every_prompt_file_has_system_and_user():
    for name in ("extract_v0", "digest_v0", "verifier_v0", "recap_v0"):
        p = prompts.load(name)
        assert p.system.strip() and p.user.strip(), name


def test_extract_v0_placeholders_are_the_contract():
    assert prompts.load("extract_v0").variables == EXTRACT_VARS


def test_render_substitutes_and_keeps_json_braces():
    p = prompts.load("extract_v0")
    system, user = p.render(**{v: f"<{v}>" for v in EXTRACT_VARS})
    assert "<language>" in system
    assert "<text>" in user and "{{" not in user
    assert '"chunk_summary":' in user  # the literal JSON example survives


def test_render_missing_variable_raises():
    p = prompts.load("extract_v0")
    with pytest.raises(KeyError, match="known_terms"):
        p.render(**{v: "x" for v in EXTRACT_VARS - {"known_terms"}})


def test_render_unknown_variable_raises():
    p = prompts.load("extract_v0")
    with pytest.raises(KeyError, match="lectrue_title"):
        p.render(**{v: "x" for v in EXTRACT_VARS}, lectrue_title="typo")


def test_missing_prompt_file_raises():
    with pytest.raises(FileNotFoundError):
        prompts.load("extract_v99")
