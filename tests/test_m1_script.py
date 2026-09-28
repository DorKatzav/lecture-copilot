import json

import pytest

from lecture_copilot.store.db import Store
from scripts import m1

FOOTPRINT = """======================================================================
llama-server [61332]: 64-bit    Footprint: 18 GB (16384 bytes per page)
======================================================================

  Dirty      Clean  Reclaimable    Regions    Category
    ---        ---          ---        ---    ---
9310 MB        0 B          0 B        101    untagged (VM_ALLOCATE)
8645 MB        0 B          0 B        106    MALLOC_LARGE
  33 MB        0 B          0 B         29    MALLOC_SMALL
9678 KB        0 B          0 B          1    page table
 800 KB        0 B          0 B        174    IOAccelerator (graphics)
    0 B    4256 KB          0 B        520    __TEXT
"""

VMMAP = """Physical footprint:         10.1G
MALLOC_LARGE                       8.7G    9280K    9280K     8.4G       0K       0K       0K      104
MALLOC_LARGE (empty)              32.0M      32K      32K    24.0M       0K       0K       0K        4
"""


def test_footprint_categories_in_mb():
    f = m1.parse_footprint(FOOTPRINT)
    assert f["categories"]["MALLOC_LARGE"] == 8645 and f["categories"]["VM_ALLOCATE"] == 9310
    assert f["footprint_mb"] == pytest.approx(9310 + 8645 + 33 + 9678 / 1024 + 800 / 1024, abs=0.5)


def test_vmmap_swapped_size_of_a_region():
    assert m1.parse_vmmap_swapped_mb(VMMAP, "MALLOC_LARGE") == pytest.approx(8.4 * 1024)


def test_load_mode_flag_from_the_runner_command():
    assert m1.load_mode("llama-server --model x --load-mode none --flash-attn auto") == "none"
    assert m1.load_mode("llama-server --model x --flash-attn auto") == "mmap (default)"


def test_fixture_metrics_come_from_the_run_rows(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    course = s.upsert_course("c", language="he")
    lid = s.upsert_lecture(course, audio_path="/f/fixture.m4a", source="file", title="t", date="d", fact_check=True)
    for run_id, pressure in (("R1", 2), ("R2", 1)):
        s.log("extractor", lecture_id=lid, input_ref=f"{run_id}#0001", output={"status": "ok", "attempts": 1,
                                                                               "first_valid": True})
        s.log("run", lecture_id=lid, input_ref=run_id, output={"chunks": 1, "counts": {"segments": 3},
                                                              "memory": {"max_pressure": pressure}})
    s.close()
    from pathlib import Path
    out = m1.fixture_metrics(tmp_path / "c.sqlite", Path("/f/fixture.m4a"))
    assert out["n_runs"] == 2 and out["first"]["memory"]["max_pressure"] == 2 and out["last"]["counts"]["segments"] == 3
    assert out["last"]["extract"] == {"calls": 1, "first_valid": 1, "retries": 0}
    json.dumps(out)


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_transcripts_compare_text_and_segmentation_separately():
    a = {"segments": [seg(0, 1000, " שלום "), seg(1000, 2000, "לכולם")]}
    same_text_other_cuts = {"segments": [seg(0, 2000, "שלום לכולם")]}
    other_text = {"segments": [seg(0, 1000, "שלום"), seg(1000, 2000, "לכולן")]}
    assert m1.compare_transcripts(a, a) == {"same_text": True, "same_segments": True}
    assert m1.compare_transcripts(a, same_text_other_cuts) == {"same_text": True, "same_segments": False}
    assert m1.compare_transcripts(a, other_text) == {"same_text": False, "same_segments": True}


def test_repeat_summary_counts_chunks():
    rows = [{"same_text": True, "same_segments": True}, {"same_text": True, "same_segments": False},
            {"same_text": False, "same_segments": False}]
    assert m1.summarize_repeat(rows) == {"n": 3, "same_text": 2, "same_segments": 1}


def test_growth_is_the_mean_step_between_prompts():
    assert m1.cache_growth([4650, 5590, 6530, 7470]) == {"first_mb": 4650, "last_mb": 7470, "per_prompt_mb": 940}
    assert m1.cache_growth([4650, 4650, 4650])["per_prompt_mb"] == 0


def test_fixture_runs_are_addressable_by_number(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    course = s.upsert_course("c", language="he")
    lid = s.upsert_lecture(course, audio_path="/f/fixture.m4a", source="file", title="t", date="d", fact_check=True)
    for run_id in ("R1", "R2", "R3"):
        s.log("run", lecture_id=lid, input_ref=run_id, output={"chunks": 1, "counts": {"segments": 3}})
    s.close()
    from pathlib import Path
    out = m1.fixture_metrics(tmp_path / "c.sqlite", Path("/f/fixture.m4a"))
    assert list(out["by_run"]) == ["r1", "r2", "r3"] and out["by_run"]["r3"]["run_id"] == "R3"
