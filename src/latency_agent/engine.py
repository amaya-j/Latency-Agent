from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class Order:
    t_arrival: float
    qty: int = 1
    side: int = 1  # +1 buy, -1 sell (placeholder)


class MatchingEngineSim:
    """
    Toy event-driven engine:
    - orders arrive into a queue
    - engine processes them in batches
    - processing cost depends on batch size (+ occasional stress spike)
    """

    def __init__(
        self,
        stress_prob: float = 0.08,
        stress_extra_cost_s: float = 200e-6,  # 200 microseconds
        base_cost_s: float = 30e-6,           # 30 microseconds
        per_order_cost_s: float = 10e-6,      # 10 microseconds/order
    ):
        self.q: Deque[Order] = deque()
        self.stress_prob = stress_prob
        self.stress_extra_cost_s = stress_extra_cost_s
        self.base_cost_s = base_cost_s
        self.per_order_cost_s = per_order_cost_s

    def enqueue(self, order: Order) -> None:
        self.q.append(order)

    def _simulate_processing_cost(self, batch_n: int) -> None:
        cost = self.base_cost_s + self.per_order_cost_s * batch_n
        if random.random() < self.stress_prob:
            cost += self.stress_extra_cost_s
        time.sleep(cost)

    def step(self, batch_size: int) -> List[float]:
        """
        Process up to batch_size orders.
        Returns per-order latency samples (seconds).
        """
        batch_n = min(batch_size, len(self.q))
        if batch_n <= 0:
            return []

        self._simulate_processing_cost(batch_n)

        t_done = time.perf_counter()
        lats: List[float] = []
        for _ in range(batch_n):
            o = self.q.popleft()
            lats.append(t_done - o.t_arrival)
        return lats
