from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt

from latency_agent.agent import BatchTuningAgent
from latency_agent.engine import MatchingEngineSim
from latency_agent.experiments import run_episode

OUTDIR = Path("plots")

# ---- Traffic regime (stress it so adaptation is visible) ----
BASE_RATE = 800.0
BURST_RATE = 4000.0
BURST_PROB = 0.30

# ---- Agent config ----
ACTIONS = [1, 4, 8, 16, 32, 64]
BASE_SEED = 7


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)

    agent = BatchTuningAgent(actions=ACTIONS, eps=0.25)

    # Train enough to get a non-random greedy trace, but keep it reasonable
    TRAIN_EPISODES = 120
    TRAIN_DURATION_S = 3.0

    for ep in range(TRAIN_EPISODES):
        random.seed(BASE_SEED + ep)
        engine = MatchingEngineSim()
        run_episode(
            agent,
            engine,
            duration_s=TRAIN_DURATION_S,
            base_rate=BASE_RATE,
            burst_rate=BURST_RATE,
            burst_prob=BURST_PROB,
        )

    # Greedy trace run
    agent.eps = 0.05
    random.seed(BASE_SEED + 10_000)

    engine = MatchingEngineSim()
    m = run_episode(
        agent,
        engine,
        duration_s=5.0,
        base_rate=BASE_RATE,
        burst_rate=BURST_RATE,
        burst_prob=BURST_PROB,
        decision_interval_s=0.20,
        trace=True,
    )

    tr = m["trace"]
    t = tr["t_s"]
    batch = tr["batch"]
    qlen = tr["q_len"]

    fig, ax1 = plt.subplots()

    # Agent batch size (red)
    ax1.step(
        t,
        batch,
        where="post",
        label="agent batch size",
        color="#d62728",   # red
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Agent batch size")
    ax1.set_title("Agent behavior: batch choice vs queue length over time")

    # Queue length (blue)
    ax2 = ax1.twinx()
    ax2.plot(
        t,
        qlen,
        label="queue length",
        color="#1f77b4",   # blue
    )
    ax2.set_ylabel("Queue length")

    # Combine legends from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    out = OUTDIR / "agent_trace.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"Saved: {out.resolve()}")
    print(
        "Greedy run summary:",
        {k: m[k] for k in ["throughput_ops", "p50_ms", "p99_ms", "p99.9_ms", "final_queue"]},
    )


if __name__ == "__main__":
    main()
