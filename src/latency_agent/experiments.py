from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List

from .agent import BatchTuningAgent
from .engine import MatchingEngineSim, Order
from .metrics import percentile
from .traffic import generate_traffic


def run_baseline(
    batch_size: int,
    duration_s: float = 3.0,
    base_rate: float = 500.0,
    burst_rate: float = 2500.0,
    burst_prob: float = 0.20,
) -> Dict[str, float]:
    engine = MatchingEngineSim()
    rel_arrivals = generate_traffic(duration_s, base_rate, burst_rate, burst_prob)

    t0 = time.perf_counter()
    abs_arrivals = [t0 + ra for ra in rel_arrivals]

    latencies: List[float] = []
    idx = 0

    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break

        while idx < len(abs_arrivals) and abs_arrivals[idx] <= now:
            engine.enqueue(Order(t_arrival=abs_arrivals[idx]))
            idx += 1

        latencies.extend(engine.step(batch_size=batch_size))

        if len(engine.q) == 0 and idx < len(abs_arrivals):
            sleep_for = max(0.0, min(0.002, abs_arrivals[idx] - time.perf_counter()))
            if sleep_for > 0:
                time.sleep(sleep_for)

    return _summarize(latencies, duration_s, final_q=len(engine.q))


def run_episode(
    agent: BatchTuningAgent,
    engine: MatchingEngineSim,
    duration_s: float = 3.0,
    base_rate=800,
    burst_rate=4000,
    burst_prob=0.30,
    decision_interval_s: float = 0.050,
    trace: bool = False,   # NEW
):
    rel_arrivals = generate_traffic(duration_s, base_rate, burst_rate, burst_prob)

    t0 = time.perf_counter()
    abs_arrivals = [t0 + ra for ra in rel_arrivals]

    latencies: List[float] = []
    recent: Deque[float] = deque(maxlen=500)

    # NEW: trace buffers (seconds since start)
    trace_t: List[float] = []
    trace_batch: List[int] = []
    trace_q: List[int] = []
    trace_burst: List[int] = []

    idx = 0
    next_decision = t0
    burst_flag = 0

    s, batch_size = agent.act(q_len=0, burst=0)

    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break

        while idx < len(abs_arrivals) and abs_arrivals[idx] <= now:
            engine.enqueue(Order(t_arrival=abs_arrivals[idx]))
            idx += 1

        observed_rate = idx / max(1e-9, (now - t0))
        burst_flag = 1 if observed_rate > (base_rate * 1.5) else 0

        # Decision point
        if now >= next_decision:
            s, batch_size = agent.act(q_len=len(engine.q), burst=burst_flag)

            if trace:
                trace_t.append(now - t0)
                trace_batch.append(int(batch_size))
                trace_q.append(int(len(engine.q)))
                trace_burst.append(int(burst_flag))

            next_decision += decision_interval_s

        lats = engine.step(batch_size=batch_size)
        if lats:
            latencies.extend(lats)
            recent.extend(lats)

        if len(recent) >= 50:
            p999 = percentile(list(recent), 99.9)
            backlog_penalty = 1e-6 * len(engine.q)
            r = -p999 - backlog_penalty
            s2 = agent.state(len(engine.q), burst_flag)
            agent.learn(s, batch_size, r, s2)

        if len(engine.q) == 0 and idx < len(abs_arrivals):
            sleep_for = max(0.0, min(0.002, abs_arrivals[idx] - time.perf_counter()))
            if sleep_for > 0:
                time.sleep(sleep_for)

    metrics = _summarize(latencies, duration_s, final_q=len(engine.q))

    if trace:
        metrics["trace"] = {
            "t_s": trace_t,
            "batch": trace_batch,
            "q_len": trace_q,
            "burst": trace_burst,
        }

    return metrics



def _summarize(latencies: List[float], duration_s: float, final_q: int) -> Dict[str, float]:
    return {
        "orders_processed": float(len(latencies)),
        "throughput_ops": (len(latencies) / duration_s),
        "p50_ms": percentile(latencies, 50) * 1e3,
        "p99_ms": percentile(latencies, 99) * 1e3,
        "p99.9_ms": percentile(latencies, 99.9) * 1e3,
        "final_queue": float(final_q),
    }
