from lecture_copilot import config


def test_root_is_the_repo():
    assert (config.ROOT / "PLAN.md").is_file()
    assert config.DB_PATH == config.ROOT / "db" / "copilot.sqlite"


def test_one_local_llm_for_live_and_digest_d_m0_10():
    assert (config.LIVE_MODEL, config.DIGEST_MODEL, config.EMBED_MODEL) == ("gemma3:12b", "gemma3:12b", "bge-m3")


def test_budget_matches_spec():
    assert config.BUDGET_S == {"asr": 8, "extract": 15, "embed": 2}


def test_profile_defaults():
    p = config.Profile()
    assert p.fact_check is True and p.language == "he"


def test_asr_model_follows_course_language_d_m0_8():
    assert config.MW_MODELS == {
        "he": "whisper-cpp:ivrit-ai-largev3",
        "en": "whisperkit:openai_whisper-large-v3-v20240930",
    }


def test_extraction_limits_follow_the_budget():
    assert config.CHUNK_BUDGET_S == 30 and config.EXTRACT_TIMEOUT_S == 2 * config.BUDGET_S["extract"]
    assert config.EXTRACT_OPTIONS["temperature"] == 0 and "seed" in config.EXTRACT_OPTIONS  # D-M1-2


def test_the_prompts_in_use_exist_and_render_their_own_variables():
    from lecture_copilot import prompts
    for name in (config.EXTRACT_PROMPT, config.DIGEST_MAP_PROMPT, config.DIGEST_EXEC_PROMPT):
        p = prompts.load(name)
        p.render(**{v: "x" for v in p.variables})
    assert {"text", "previous_chunk_summary", "known_terms"} <= prompts.load(config.EXTRACT_PROMPT).variables
    assert {"chunk_summaries", "concepts", "words"} <= prompts.load(config.DIGEST_MAP_PROMPT).variables
    assert {"full_summary", "prev_bullets"} <= prompts.load(config.DIGEST_EXEC_PROMPT).variables
