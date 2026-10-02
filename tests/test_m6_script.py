import json

from scripts.m6 import sync_stats


def test_sync_stats_counts_calls_bytes_and_seconds_per_lecture(tmp_path):
    from lecture_copilot.store.db import Store
    s = Store(tmp_path / "c.sqlite")
    for i in range(3):
        s.log("net", lecture_id="L1", input_ref="notion:L1", ms=200.0 + i, output={
            "host": "api.notion.com", "status": "ok", "bytes_out": 1000, "bytes_in": 500, "method": "POST",
            "path": "/v1/pages"})
    s.log("net", lecture_id="L1", input_ref="notion:L1", ms=50.0, output={"host": "api.notion.com",
                                                                            "status": "failed", "error": "500"})
    s.log("net", lecture_id="L1", input_ref="gemini:C1", ms=50.0, output={"host": "generativelanguage.googleapis.com",
                                                                            "status": "ok", "bytes_out": 10})
    s.log("sink", lecture_id="L1", input_ref="L1:write_lecture", ms=4100.0,
          output={"sink": "NotionSink", "status": "ok", "calls": 4})
    s.log("sink", lecture_id="L1", input_ref="L1:write_lecture", ms=900.0,
          output={"sink": "FolderSink", "status": "ok", "files": 5})
    s.log("sink", lecture_id="L1", input_ref=":write_course", ms=1200.0,
          output={"sink": "NotionSink", "status": "ok", "calls": 3})
    out = sync_stats(s.con, "L1")
    s.close()
    assert out == {"lecture_id": "L1", "calls": 4, "ok": 3, "failed": 1, "bytes_out_kb": 2.9, "bytes_in_kb": 1.5,
                   "net_s": 0.65, "sink_s": 4.1, "course_s": 1.2, "by_path": {"POST /v1/pages": 3},
                   "cost_usd": 0.0, "syncs": [{"status": "ok", "calls": 4, "s": 4.1}]}
    assert json.dumps(out)
