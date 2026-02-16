# latency-agent

A reproducible systems experiment that simulates a simplified low-latency matching engine and trains a lightweight reinforcement learning agent to reduce **tail latency (p99 / p99.9)** under bursty traffic.

This project explores how adaptive batching trades throughput vs. latency in event-driven systems.

---

## Motivation

In latency-sensitive systems (e.g., trading infrastructure, exchange gateways, HFT engines), **tail latency matters more than average latency**.

Under bursty traffic, static batching policies can increase queueing delay and inflate p99 latency.

This project investigates:

- How batching affects the latency distribution  
- Whether a simple RL agent can dynamically adapt batch size  
- How performance compares to static baselines  

---

## Architecture

```
Traffic Generator → Engine Simulator → Latency Metrics → RL Agent
```

### Components

**Event-driven engine simulator**
- Processes incoming events in batches  
- Simulates service time and queueing delay  

**Bursty Poisson traffic generator**
- Models clustered arrival bursts  
- Configurable intensity and burst duration  

**Latency metrics**
- p50  
- p99  
- p99.9  

**RL Agent**
- Epsilon-greedy Q-learning  
- Discrete action space (batch sizes)  
- Reward based on weighted tail latency  

---

## Installation

### 1. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install in editable mode

```bash
pip install -e ".[dev]"
```

---

## Running Baselines

Run static batching policies:

```bash
python scripts/run_baseline.py 
```

Outputs:
- Latency percentiles  
- Throughput  
- Summary statistics  

---

## Training the Agent

```bash
python scripts/train_agent.py 
```

Example output:

```
Episode 150:
Batch size chosen: 4
p99 latency: 842 µs
p99.9 latency: 1210 µs
Reward: -0.84
```

The agent learns to adapt batch size based on traffic regime.

---

## Reproducibility

- Deterministic random seeds supported  
- Hyperparameters configurable via CLI  
- Results saved as CSV  
- Plotting scripts included (CDFs, percentile comparisons)  

---

## Experiments to Try

- Static vs adaptive batching comparison  
- Increase burst amplitude  
- Increase service-time variance  
- Simulate traffic regime shifts  

---

## Future Extensions

- Replace tabular Q-learning with DQN  
- Introduce deeper order book simulation  
- Model NIC interrupt coalescing  
- Multi-agent congestion control  
- Kernel-level timestamp integration  

---

## Why This Project Matters

This project connects:

- Systems performance engineering  
- Queueing theory  
- Tail latency analysis  
- Reinforcement learning  
- Low-latency infrastructure concepts  

It demonstrates how adaptive control policies can reduce extreme tail events without sacrificing throughput.

