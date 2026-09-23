from lecture_copilot import config


def test_root_is_the_repo():
    assert (config.ROOT / "PLAN.md").is_file()
    assert config.DB_PATH == config.ROOT / "db" / "copilot.sqlite"


def test_models_are_explicit():
    assert (config.LIVE_MODEL, config.DIGEST_MODEL, config.EMBED_MODEL) == ("qwen3:8b", "gemma3:12b", "bge-m3")


def test_budget_matches_spec():
    assert config.BUDGET_S == {"asr": 8, "extract": 15, "embed": 2}


def test_profile_defaults():
    p = config.Profile()
    assert p.fact_check is True and p.language == "he"


def test_asr_model_is_explicit_not_the_app_default():
    assert config.MW_MODEL == "whisperkit:openai_whisper-large-v3-v20240930"
