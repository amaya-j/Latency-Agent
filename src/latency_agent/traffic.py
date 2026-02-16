from __future__ import annotations

import math
import random
from typing import List


def _poisson_knuth(lam: float) -> int:
    """Sample Poisson(lam) using Knuth's method (fine for small lam)."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def generate_traffic(
    duration_s: float,
    base_rate: float,   # orders/sec
    burst_rate: float,  # orders/sec during burst
    burst_prob: float,  # probability a slice is bursty
    dt: float = 0.005,  # 5ms slices
) -> List[float]:
    """
    Returns sorted arrival times in SECONDS relative to t=0.
    Bursty Poisson arrivals: each slice chooses base or burst rate.
    """
    t = 0.0
    arrivals: List[float] = []
    while t < duration_s:
        rate = burst_rate if random.random() < burst_prob else base_rate
        lam = rate * dt
        n = _poisson_knuth(lam)
        for _ in range(n):
            arrivals.append(t + random.random() * dt)
        t += dt
    arrivals.sort()
    return arrivals
