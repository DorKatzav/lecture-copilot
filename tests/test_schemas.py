import pytest
from pydantic import ValidationError

from lecture_copilot.agents.schemas import ExtractResult

VALID = {
    "chunk_summary": "המרצה הגדיר CAC והסביר איך מחשבים אותו.",
    "concepts": [
        {"term": "CAC", "explanation": "כמה עולה להשיג לקוח אחד.", "canonical_key": "customer_acquisition_cost"},
    ],
    "claims": [{"text": "Uber was founded in 2009", "normalized": "Uber was founded in 2009.", "importance": 72}],
    "items": [{"kind": "highlight", "text": "זה יהיה במבחן", "owner": None, "due": None}],
}


def test_valid_payload_parses():
    res = ExtractResult.model_validate(VALID)
    assert res.concepts[0].canonical_key == "customer_acquisition_cost"
    assert res.claims[0].importance == 72
    assert res.items[0].kind == "highlight"


def test_empty_lists_are_valid():
    res = ExtractResult.model_validate({"chunk_summary": "שקט.", "concepts": [], "claims": [], "items": []})
    assert res.claims == []


@pytest.mark.parametrize("importance", [-1, 101])
def test_importance_out_of_range_is_rejected(importance):
    bad = {**VALID, "claims": [{**VALID["claims"][0], "importance": importance}]}
    with pytest.raises(ValidationError):
        ExtractResult.model_validate(bad)


def test_unknown_item_kind_is_rejected():
    bad = {**VALID, "items": [{"kind": "rumor", "text": "x"}]}
    with pytest.raises(ValidationError):
        ExtractResult.model_validate(bad)


def test_missing_chunk_summary_is_rejected():
    bad = {k: v for k, v in VALID.items() if k != "chunk_summary"}
    with pytest.raises(ValidationError):
        ExtractResult.model_validate(bad)
