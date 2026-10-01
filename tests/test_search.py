import pytest

from lecture_copilot.agents.schemas import Claim, Concept, ExtractResult
from lecture_copilot.asr.base import Segment
from lecture_copilot.store.db import SCHEMA_VERSION, Store
from lecture_copilot.store.search import fts_query, rrf
from tests.stubs import EMBED_DIMS, fake_embedding


@pytest.fixture(params=["sqlite-vec", "numpy"])
def store(tmp_path, request):
    s = Store(tmp_path / "c.sqlite", vec_backend=request.param, dims=EMBED_DIMS)
    assert s.vec_backend == request.param
    yield s
    s.close()


def concept(term, explanation, key):
    return Concept(term=term, explanation=explanation, canonical_key=key)


def write(store, lid, idx, concepts=(), claims=(), t0=0.0):
    res = ExtractResult(chunk_summary="s", concepts=list(concepts), items=[],
                        claims=[Claim(text=c, normalized=c, importance=70) for c in claims])
    store.write_chunk(lid, idx, [Segment(t0=t0, t1=t0 + 40, text="x")], asr="mw", result=res)
    for row in store.items(lid) + store.claims(lid):
        if row["embedding"] is None:
            store.set_embedding("items" if "kind" in row else "claims", row["id"],
                                fake_embedding(f"{row['text']} {row.get('explanation') or ''}"))


def two_lectures(store):
    course = store.upsert_course("יזמות", language="he")
    w4 = store.upsert_lecture(course, audio_path="/w4", source="file", title="W4", date="2026-10-28", fact_check=True)
    w5 = store.upsert_lecture(course, audio_path="/w5", source="file", title="W5", date="2026-11-04", fact_check=True)
    write(store, w4, 1, [concept("CAC", "עלות רכישת לקוח: הוצאות שיווק חלקי לקוחות חדשים", "cac"),
                        concept("Churn", "שיעור הלקוחות שעוזבים", "churn")], ["CAC של Dropbox היה 30 דולר"])
    write(store, w5, 1, [concept("LTV", "ערך חיי לקוח", "ltv")], ["LTV צריך להיות פי 3 מה-CAC"])
    return course, w4, w5


# ---------- pure helpers ----------

def test_fts_query_quotes_every_token_and_drops_syntax():
    assert fts_query('CAC "עלות" OR (לקוח) NOT x*') == '"CAC" OR "עלות" OR "OR" OR "לקוח" OR "NOT" OR "x"'
    assert fts_query("   ") == ""


def test_rrf_rewards_items_found_by_both_lists():
    fused = rrf([["a", "b", "c"], ["c", "a"]], k=60)
    assert [i for i, _ in fused] == ["a", "c", "b"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_breaks_ties_by_first_appearance():
    assert [i for i, _ in rrf([["concept"], ["claim"]])] == ["concept", "claim"]


# ---------- the store ----------

def test_schema_has_fts_tables_and_the_current_version(store):
    names = {r[0] for r in store.con.execute("select name from sqlite_master")}
    assert {"items_fts", "claims_fts"} <= names and SCHEMA_VERSION == 4


def test_text_search_finds_a_concept_by_term_and_by_explanation(store):
    course, w4, w5 = two_lectures(store)
    assert store.search("CAC", course)[0].canonical_key == "cac"
    hit = store.search("הוצאות שיווק", course)[0]
    assert hit.kind == "concept" and hit.text == "CAC" and hit.lecture_id == w4


def test_text_search_finds_claims_too(store):
    course, w4, w5 = two_lectures(store)
    hits = store.search("Dropbox", course)
    assert hits[0].kind == "claim" and "Dropbox" in hits[0].text


def test_vector_search_finds_the_nearest_meaning(store):
    course, w4, w5 = two_lectures(store)
    hits = store.search("", course, query_vec=fake_embedding("שיעור הלקוחות שעוזבים"))
    assert hits[0].text == "Churn"


def test_text_and_vector_are_fused_by_rrf(store):
    course, w4, w5 = two_lectures(store)
    hits = store.search("CAC", course, query_vec=fake_embedding("CAC עלות רכישת לקוח: הוצאות שיווק חלקי לקוחות חדשים"))
    assert hits[0].text == "CAC" and hits[0].score > hits[1].score


def test_search_is_scoped_to_the_course_and_can_exclude_a_lecture(store):
    course, w4, w5 = two_lectures(store)
    other = store.upsert_course("אחר", language="he")
    assert store.search("CAC", other) == []
    assert all(h.lecture_id == w4 for h in store.search("CAC", course, exclude_lecture_id=w5))
    assert all(h.lecture_id != w5 for h in store.search("LTV", course, exclude_lecture_id=w5))


def test_hits_carry_the_lecture_week_and_title(store):
    course, w4, w5 = two_lectures(store)
    store.con.execute("update lectures set week = 4 where id = ?", (w4,))
    store.con.commit()
    hit = store.search("Churn", course)[0]
    assert (hit.week, hit.title) == (4, "W4")


def test_k_limits_the_result(store):
    course, w4, w5 = two_lectures(store)
    assert len(store.search("לקוח", course, k=1)) == 1


def test_rewriting_a_chunk_leaves_no_stale_index_rows(store):
    course, w4, w5 = two_lectures(store)
    write(store, w4, 1, [concept("MRR", "הכנסה חודשית חוזרת", "mrr")])
    assert store.search("CAC", course, exclude_lecture_id=w5) == []
    assert store.search("MRR", course)[0].text == "MRR"
    assert store.con.execute("select count(*) from items_fts").fetchone()[0] == store.con.execute(
        "select count(*) from items").fetchone()[0]


def test_embeddings_are_stored_as_float32_blobs(store):
    course, w4, w5 = two_lectures(store)
    blob = store.con.execute("select embedding from items where text = 'CAC'").fetchone()[0]
    assert isinstance(blob, bytes) and len(blob) == EMBED_DIMS * 4


def test_a_v2_database_gets_its_text_index_backfilled(tmp_path):
    s = Store(tmp_path / "old.sqlite", vec_backend="numpy", dims=EMBED_DIMS)
    course, w4, w5 = two_lectures(s)
    s.con.executescript("drop table items_fts; drop table claims_fts; pragma user_version = 2;")
    s.close()
    s = Store(tmp_path / "old.sqlite", vec_backend="numpy", dims=EMBED_DIMS)
    assert s.search("CAC", course)[0].text == "CAC"
    s.close()


def test_sqlite_vec_tables_hold_no_orphans_after_a_rewrite(tmp_path):
    s = Store(tmp_path / "v.sqlite", vec_backend="sqlite-vec", dims=EMBED_DIMS)
    course, w4, w5 = two_lectures(s)
    write(s, w4, 1, [concept("MRR", "הכנסה חודשית חוזרת", "mrr")])
    assert s.con.execute("select count(*) from vec_items").fetchone()[0] == s.con.execute(
        "select count(*) from items where embedding is not null").fetchone()[0]
    s.close()
