"""Peak memory during a run (CLAUDE.md: MacWhisper + Gemma + bge-m3 must fit 24 GB — measured in M1).

A background thread samples every second: system memory in use (app + wired + compressed, as Activity Monitor
counts it), swap, the kernel's memory-pressure level (1 normal, 2 warning, 4 critical), the physical footprint of
MacWhisper / mw / Ollama processes (top's MEM), and the models Ollama holds (/api/ps). macOS only; on other systems
the sampler fails and the summary says so.
"""

import re
import subprocess
import threading

import httpx

from lecture_copilot.config import OLLAMA_URL

WATCH = ("MacWhisper", "mw", "ollama")
UNITS = {"B": 1 / 2**20, "K": 1 / 1024, "M": 1.0, "G": 1024.0}


def parse_vm_stat_gb(text: str) -> float:
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    v = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(.+?):\s+(\d+)\.$", text, re.MULTILINE)}
    pages = v["Anonymous pages"] - v["Pages purgeable"] + v["Pages wired down"] + v["Pages occupied by compressor"]
    return round(pages * page / 2**30, 2)


def parse_swap_mb(text: str) -> float:
    return float(re.search(r"used = ([\d.]+)M", text).group(1))


def parse_mem(text: str) -> float:
    m = re.fullmatch(r"([\d.]+)([BKMG])[+-]?", text.strip())
    return float(m.group(1)) * UNITS[m.group(2)]


def parse_top(text: str) -> dict[str, float]:
    """`top -l 1 -stats pid,command,mem` → MB per command name (processes of one name summed)."""
    out: dict[str, float] = {}
    rows = text.split("PID")[-1].splitlines()[1:]
    for line in rows:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            name = " ".join(parts[1:-1])
            out[name] = out.get(name, 0.0) + parse_mem(parts[-1])
    return out


def _sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10).stdout


def macos_sample(ollama_url: str = OLLAMA_URL) -> dict:
    procs_mb: dict[str, float] = {}
    pids = [line.split()[0] for line in _sh("ps", "-A", "-o", "pid=,comm=").splitlines()
            if line.split() and line.split()[-1].rsplit("/", 1)[-1] in WATCH]
    if pids:
        pid_args = [a for p in pids for a in ("-pid", p)]
        procs_mb = parse_top(_sh("top", "-l", "1", "-stats", "pid,command,mem", *pid_args))
    try:
        ps = httpx.get(f"{ollama_url}/api/ps", timeout=2).json()["models"]
        models_gb = {m["name"].removesuffix(":latest"): round(m["size"] / 2**30, 2) for m in ps}
    except (httpx.HTTPError, KeyError, ValueError):
        models_gb = {}
    return {"used_gb": parse_vm_stat_gb(_sh("vm_stat")), "swap_mb": parse_swap_mb(_sh("sysctl", "-n", "vm.swapusage")),
            "pressure": int(_sh("sysctl", "-n", "kern.memorystatus_vm_pressure_level")),
            "procs_mb": {k: round(v) for k, v in procs_mb.items()}, "models_gb": models_gb}


def total_gb() -> float | None:
    try:
        return round(int(_sh("sysctl", "-n", "hw.memsize")) / 2**30, 1)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class MemoryProbe:
    def __init__(self, sampler=macos_sample, total_gb: float | None = None, interval_s: float = 1.0):
        self.sampler, self.total, self.interval_s = sampler, total_gb, interval_s
        self.samples: list[dict] = []
        self.errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def take(self) -> None:
        try:
            self.samples.append(self.sampler())
        except Exception:
            self.errors += 1

    def _loop(self) -> None:
        self.take()
        while not self._stop.wait(self.interval_s):
            self.take()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="memprobe", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)

    def summary(self) -> dict:
        s = list(self.samples)
        if not s:
            return {"samples": 0, "errors": self.errors}
        peak = max(s, key=lambda x: x["used_gb"])
        procs: dict[str, float] = {}
        for x in s:
            for name, mb in x["procs_mb"].items():
                procs[name] = max(procs.get(name, 0), mb)
        out = {"baseline_used_gb": s[0]["used_gb"], "peak_used_gb": peak["used_gb"],
               "delta_gb": round(peak["used_gb"] - s[0]["used_gb"], 2), "total_gb": self.total,
               "max_pressure": max(x["pressure"] for x in s),
               "swap_growth_mb": round(s[-1]["swap_mb"] - s[0]["swap_mb"], 1),
               "models_at_peak_gb": peak["models_gb"], "procs_peak_mb": procs, "samples": len(s)}
        if self.errors:
            out["errors"] = self.errors
        return out
