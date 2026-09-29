"""Small numeric helpers shared by the pipeline, the gate and the stage-0 script."""

import math


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile (p95 of up to 20 values is their max)."""
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]
