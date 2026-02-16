from __future__ import annotations

import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple


class BatchTuningAgent:
    """
    Tiny epsilon-greedy Q-learner.

    State: (queue_bucket, burst_flag)
    Action: pick a batch size from a small discrete set
    """

    def __init__(self, actions: List[int], eps: float = 0.15, alpha: float = 0.20, gamma: float = 0.90):
        self.actions = actions
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.Q: DefaultDict[Tuple[int, int], Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def _bucket_q(self, q_len: int) -> int:
        if q_len < 10:
            return 0
        if q_len < 50:
            return 1
        if q_len < 200:
            return 2
        return 3

    def state(self, q_len: int, burst: int) -> Tuple[int, int]:
        return (self._bucket_q(q_len), burst)

    def act(self, q_len: int, burst: int) -> Tuple[Tuple[int, int], int]:
        s = self.state(q_len, burst)
        if random.random() < self.eps:
            return s, random.choice(self.actions)
        qs = self.Q[s]
        best_a = max(self.actions, key=lambda a: qs[a])
        return s, best_a

    def learn(self, s: Tuple[int, int], a: int, r: float, s2: Tuple[int, int]) -> None:
        qsa = self.Q[s][a]
        best_next = max(self.actions, key=lambda a2: self.Q[s2][a2])
        target = r + self.gamma * self.Q[s2][best_next]
        self.Q[s][a] = qsa + self.alpha * (target - qsa)
