from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from latency_agent.agent import BatchTuningAgent
from latency_agent.engine import MatchingEngineSim, Order
from latency_agent.experiments import run_baseline, run_episode
from latency_agent.traffic import generate_traffic

OUTDIR = Path("plots")

# ---- Traffic regime (match plot_agent_trace.py) ----
BASE_RATE = 800.0
BURST_RATE = 4000.0
BURST_PROB = 0.30

# ---- Actions ----
BATCHES = [1, 4, 8, 16, 32, 64]


def ecdf(xs: List[float]) -> Tuple[List[float], List[float]]:
    """Return x(sorted) and y=cumulative probabilities for an empirical CDF."""
    if not xs:
        return [], []
    x = sorted(xs)
    n = len(x)
    y = [(i + 1) / n for i in range(n)]
    return x, y


def plot_baselines() -> List[Dict[str, float]]:
    random.seed(7)

    rows: List[Dict[str, float]] = []
    for b in BATCHES:
        m = run_baseline(
            batch_size=b,
            duration_s=3.0,
            base_rate=BASE_RATE,
            burst_rate=BURST_RATE,
            burst_prob=BURST_PROB,
        )
        m["batch"] = float(b)
        rows.append(m)

    OUTDIR.mkdir(exist_ok=True)

    x = [int(r["batch"]) for r in rows]
    p50 = [r["p50_ms"] for r in rows]
    p99 = [r["p99_ms"] for r in rows]
    p999 = [r["p99.9_ms"] for r in rows]

    plt.figure()
    plt.bar(x, p50)
    plt.xlabel("Batch size")
    plt.ylabel("p50 latency (ms)")
    plt.title("Baseline: batch size vs p50 latency")
    plt.tight_layout()
    plt.savefig(OUTDIR / "baseline_p50.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(x, p99)
    plt.xlabel("Batch size")
    plt.ylabel("p99 latency (ms)")
    plt.title("Baseline: batch size vs p99 latency")
    plt.tight_layout()
    plt.savefig(OUTDIR / "baseline_p99.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(x, p999)
    plt.xlabel("Batch size")
    plt.ylabel("p99.9 latency (ms)")
    plt.title("Baseline: batch size vs p99.9 latency")
    plt.tight_layout()
    plt.savefig(OUTDIR / "baseline_p999.png", dpi=200)
    plt.close()

    print(f"Saved baseline plots to: {OUTDIR.resolve()}")
    return rows


def collect_latencies_for_fixed_batch(
    batch_size: int,
    duration_s: float = 5.0,
) -> List[float]:
    """
    Run a baseline-like simulation but return raw latency samples (ms) for ECDF.
    Uses the same traffic regime constants above.
    """
    engine = MatchingEngineSim()

    rel_arrivals = generate_traffic(
        duration_s=duration_s,
        base_rate=BASE_RATE,
        burst_rate=BURST_RATE,
        burst_prob=BURST_PROB,
    )

    t0 = time.perf_counter()
    abs_arrivals = [t0 + ra for ra in rel_arrivals]

    latencies_s: List[float] = []
    idx = 0

    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break

        while idx < len(abs_arrivals) and abs_arrivals[idx] <= now:
            engine.enqueue(Order(t_arrival=abs_arrivals[idx]))
            idx += 1

        latencies_s.extend(engine.step(batch_size=batch_size))

        if len(engine.q) == 0 and idx < len(abs_arrivals):
            sleep_for = max(0.0, min(0.002, abs_arrivals[idx] - time.perf_counter()))
            if sleep_for > 0:
                time.sleep(sleep_for)

    return [v * 1e3 for v in latencies_s]


def collect_latencies_for_agent(
    duration_s: float = 5.0,
    train_episodes: int = 120,
) -> List[float]:
    """
    Train quickly under the same traffic regime, then run greedy once and return raw latencies (ms).
    """
    actions = BATCHES[:]  # same set
    agent = BatchTuningAgent(actions=actions, eps=0.25)

    base_seed = 7
    for ep in range(train_episodes):
        random.seed(base_seed + ep)
        engine = MatchingEngineSim()
        run_episode(
            agent,
            engine,
            duration_s=3.0,
            base_rate=BASE_RATE,
            burst_rate=BURST_RATE,
            burst_prob=BURST_PROB,
        )

    agent.eps = 0.0
    random.seed(base_seed + 10_000)

    engine = MatchingEngineSim()
    rel_arrivals = generate_traffic(
        duration_s=duration_s,
        base_rate=BASE_RATE,
        burst_rate=BURST_RATE,
        burst_prob=BURST_PROB,
    )
    t0 = time.perf_counter()
    abs_arrivals = [t0 + ra for ra in rel_arrivals]

    latencies_s: List[float] = []
    idx = 0
    next_decision = t0
    decision_interval_s = 0.20

    _, batch_size = agent.act(q_len=0, burst=0)

    from collections import deque

    recent = deque(maxlen=500)

    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break

        while idx < len(abs_arrivals) and abs_arrivals[idx] <= now:
            engine.enqueue(Order(t_arrival=abs_arrivals[idx]))
            idx += 1

        observed_rate = idx / max(1e-9, (now - t0))
        burst_flag = 1 if observed_rate > (BASE_RATE * 1.5) else 0

        if now >= next_decision:
            _, batch_size = agent.act(q_len=len(engine.q), burst=burst_flag)
            next_decision += decision_interval_s

        lats = engine.step(batch_size=batch_size)
        if lats:
            latencies_s.extend(lats)
            recent.extend(lats)

        if len(engine.q) == 0 and idx < len(abs_arrivals):
            sleep_for = max(0.0, min(0.002, abs_arrivals[idx] - time.perf_counter()))
            if sleep_for > 0:
                time.sleep(sleep_for)

    return [v * 1e3 for v in latencies_s]


def plot_ecdf_compare() -> None:
    OUTDIR.mkdir(exist_ok=True)

    # Compare a couple of baselines + agent
    l4 = collect_latencies_for_fixed_batch(4, duration_s=5.0)
    l16 = collect_latencies_for_fixed_batch(16, duration_s=5.0)
    l32 = collect_latencies_for_fixed_batch(32, duration_s=5.0)
    lagent = collect_latencies_for_agent(duration_s=5.0, train_episodes=120)

    plt.figure()

    for label, data in [
        ("baseline batch=4", l4),
        ("baseline batch=16", l16),
        ("baseline batch=32", l32),
        ("trained agent (greedy)", lagent),
    ]:
        x, y = ecdf(data)
        plt.step(x, y, where="post", label=label)

    plt.xlabel("Latency (ms)")
    plt.ylabel("ECDF")
    plt.title("Latency ECDF comparison (tail zoom)")
    plt.legend()
    plt.xlim(0, 5)
    plt.ylim(0.95, 1.0)
    plt.tight_layout()
    plt.savefig(OUTDIR / "latency_ecdf_tail.png", dpi=200)
    plt.close()

    print(f"Saved ECDF plot to: {(OUTDIR / 'latency_ecdf_tail.png').resolve()}")


def main() -> None:
    plot_baselines()
    plot_ecdf_compare()


if __name__ == "__main__":
    main()
