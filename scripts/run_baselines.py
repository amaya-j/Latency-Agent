from __future__ import annotations

import random

from latency_agent.experiments import run_baseline


def main() -> None:
    random.seed(7)

    for b in [1, 4, 8, 16, 32, 64]:
        m = run_baseline(batch_size=b)
        print(f"baseline batch={b:>2} -> {m}")


if __name__ == "__main__":
    main()
