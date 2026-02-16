from __future__ import annotations

import random
import statistics as stats

from latency_agent.agent import BatchTuningAgent
from latency_agent.engine import MatchingEngineSim
from latency_agent.experiments import run_episode


def summarize(metrics_list: list[dict]) -> dict:
    keys = ["throughput_ops", "p50_ms", "p99_ms", "p99.9_ms"]
    out = {}
    for k in keys:
        xs = [m[k] for m in metrics_list]
        out[k] = {
            "mean": stats.mean(xs),
            "std": stats.pstdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }
    return out


def main() -> None:
    # Global base seed for reproducibility
    BASE_SEED = 7

    actions = [1, 4, 8, 16, 32, 64]
    agent = BatchTuningAgent(actions=actions, eps=0.25, alpha=0.20, gamma=0.90)

    # ---- TRAINING ----
    TRAIN_EPISODES = 200
    EP_DURATION_S = 5.0

    eps_start = 0.25
    eps_end = 0.02

    for ep in range(TRAIN_EPISODES):
        # New seed each episode (but deterministic overall)
        random.seed(BASE_SEED + ep)

        # Linear epsilon decay
        frac = ep / max(1, TRAIN_EPISODES - 1)
        agent.eps = eps_start + frac * (eps_end - eps_start)

        engine = MatchingEngineSim()
        m = run_episode(agent, engine, duration_s=EP_DURATION_S)

        if ep % 20 == 0 or ep == TRAIN_EPISODES - 1:
            print(f"train ep {ep:03d} eps={agent.eps:.3f} -> {m}")

    # ---- EVALUATION (greedy, multi-run) ----
    agent.eps = 0.0
    EVAL_RUNS = 30
    eval_metrics = []

    for i in range(EVAL_RUNS):
        random.seed(BASE_SEED + 10_000 + i)  # separate eval seed range
        engine = MatchingEngineSim()
        m = run_episode(agent, engine, duration_s=EP_DURATION_S)
        eval_metrics.append(m)

    summary = summarize(eval_metrics)

    print("\nEVAL (greedy) over", EVAL_RUNS, "runs")
    for k, v in summary.items():
        print(f"{k:>10}: mean={v['mean']:.4f} std={v['std']:.4f}  min={v['min']:.4f} max={v['max']:.4f}")


if __name__ == "__main__":
    main()
