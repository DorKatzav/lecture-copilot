import pytest

from lecture_copilot.memprobe import MemoryProbe, parse_mem, parse_swap_mb, parse_top, parse_vm_stat_gb, watched_pids

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    17528.
Pages active:                                 604365.
Pages wired down:                             201715.
Pages purgeable:                               30437.
Anonymous pages:                              776516.
Pages occupied by compressor:                  85963.
"""

TOP = """Processes: 612 total
PhysMem: 23G used (3120M wired, 1351M compressor), 263M unused.

PID    COMMAND    MEM
39900  ollama     19M
41200  ollama     8191M+
39183  MacWhisper 2.9G
"""


def test_used_memory_is_app_plus_wired_plus_compressed():
    # (anonymous - purgeable + wired + compressor) * page size
    assert parse_vm_stat_gb(VM_STAT) == pytest.approx((776516 - 30437 + 201715 + 85963) * 16384 / 2**30, abs=0.01)


def test_swap_used():
    assert parse_swap_mb("total = 2048.00M  used = 1079.00M  free = 969.00M  (encrypted)") == 1079.0


@pytest.mark.parametrize("text, mb", [("19M", 19), ("8191M+", 8191), ("2.9G", 2.9 * 1024), ("512K", 0.5),
                                      ("1024B", 1024 / 2**20)])
def test_top_mem_units(text, mb):
    assert parse_mem(text) == pytest.approx(mb)


def test_top_footprints_are_summed_per_command():
    assert parse_top(TOP) == pytest.approx({"ollama": 19 + 8191, "MacWhisper": 2.9 * 1024})


def sample(used, pressure=1, swap=1000.0, procs=None, models=None):
    return {"used_gb": used, "pressure": pressure, "swap_mb": swap, "procs_mb": procs or {}, "models_gb": models or {}}


def test_summary_reports_baseline_peak_and_what_was_resident_at_the_peak():
    samples = [sample(15.0), sample(21.5, models={"gemma3:12b": 9.1, "bge-m3": 1.2}), sample(19.0, swap=1200.0)]
    probe = MemoryProbe(sampler=lambda: samples.pop(0), total_gb=24.0)
    for _ in range(3):
        probe.take()
    s = probe.summary()
    assert (s["baseline_used_gb"], s["peak_used_gb"], s["delta_gb"], s["total_gb"]) == (15.0, 21.5, 6.5, 24.0)
    assert s["models_at_peak_gb"] == {"gemma3:12b": 9.1, "bge-m3": 1.2}
    assert s["swap_growth_mb"] == 200.0 and s["samples"] == 3


def test_worst_pressure_and_per_process_peaks_are_kept_across_samples():
    samples = [sample(15.0, procs={"mw": 3000}), sample(16.0, pressure=2, procs={"ollama": 9000}), sample(15.5)]
    probe = MemoryProbe(sampler=lambda: samples.pop(0), total_gb=24.0)
    for _ in range(3):
        probe.take()
    s = probe.summary()
    assert s["max_pressure"] == 2 and s["procs_peak_mb"] == {"mw": 3000, "ollama": 9000}


def test_a_failing_sampler_is_counted_not_raised():
    def broken():
        raise FileNotFoundError("vm_stat")

    probe = MemoryProbe(sampler=broken, total_gb=24.0)
    probe.take()
    assert probe.summary() == {"samples": 0, "errors": 1}


def test_background_thread_samples_until_stopped():
    probe = MemoryProbe(sampler=lambda: sample(15.0), total_gb=24.0, interval_s=0.01)
    probe.start()
    import time
    time.sleep(0.1)
    probe.stop()
    n = probe.summary()["samples"]
    time.sleep(0.05)
    assert n >= 2 and probe.summary()["samples"] == n


def test_watched_processes_include_ollamas_runner():
    ps = """  409 /System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer
39183 /Applications/MacWhisper.app/Contents/MacOS/MacWhisper
39900 ollama
61332 /opt/homebrew/Cellar/ollama/0.34.2/libexec/lib/ollama/llama-server
70000 /usr/local/bin/mw
"""
    assert watched_pids(ps) == ["39183", "39900", "61332", "70000"]
