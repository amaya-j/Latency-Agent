from __future__ import annotations

import math
from typing import List


def percentile(xs: List[float], p: float) -> float:
    """
    Nearest-rank percentile.
    p in [0, 100]. Returns NaN for empty input.
    """
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = int(math.ceil((p / 100.0) * len(ys))) - 1
    k = max(0, min(k, len(ys) - 1))
    return ys[k]
