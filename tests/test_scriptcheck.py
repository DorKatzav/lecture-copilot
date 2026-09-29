import pytest

from lecture_copilot.scriptcheck import forbidden_scripts


@pytest.mark.parametrize("text", [
    "מושג CAC הוא עלות רכישת לקוח",
    "git push origin main",
    "café, naïve, Müller, Ångström",          # accented Latin is allowed (CLAUDE.md)
    "‏— “quotes” … 3.5% ₪",
    "",
])
def test_hebrew_english_and_accented_latin_pass(text):
    assert forbidden_scripts(text) == set()


@pytest.mark.parametrize("text, script", [
    ("מודל бизнес", "Cyrillic"),
    ("המודל هو", "Arabic"),
    ("לקוח 客户", "CJK"),
    ("לקוח クライアント", "CJK"),
    ("לקוח 고객", "CJK"),
    ("Việt Nam", "Latin Extended Additional"),   # seen in qwen3:8b output in stage 0
])
def test_each_listed_script_is_caught(text, script):
    assert forbidden_scripts(text) == {script}


def test_several_scripts_are_all_named():
    assert forbidden_scripts("бизнес هو 客户") == {"Cyrillic", "Arabic", "CJK"}
