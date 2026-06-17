# ♟️ Chess AI — Reinforcement Learning with MCTS

[![CI](https://github.com/Franion03/chess_ai_reinforcementlearning/actions/workflows/python-package.yml/badge.svg)](https://github.com/Franion03/chess_ai_reinforcementlearning/actions/workflows/python-package.yml)
![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-GPL--3.0-green)
![RL](https://img.shields.io/badge/Method-Reinforcement%20Learning-purple)

A chess engine trained via **self-play reinforcement learning**, combining **Temporal Difference (TD) learning** with **Monte Carlo Tree Search (MCTS)** and CNN-based board evaluation. Inspired by AlphaZero's approach but implemented from scratch with prioritized experience replay.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                         │
│                                                         │
│  ┌───────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │  Self-Play │───▶│   MCTS   │───▶│  CNN Value Net   │  │
│  │  (White)   │    │ Thompson │    │  (8×8×8 input)   │  │
│  └───────────┘    │ Sampling  │    │  Conv + Dense    │  │
│        │          └──────────┘    │  → V(s) scalar   │  │
│        ▼                          └──────────────────┘  │
│  ┌───────────┐                           │              │
│  │  Greedy   │    ┌──────────────┐       │              │
│  │  (Black)  │    │  Prioritized │◀──────┘              │
│  └───────────┘    │  Experience  │                      │
│        │          │   Replay     │                      │
│        ▼          └──────────────┘                      │
│  ┌───────────┐           │                              │
│  │ TD Error  │◀──────────┘                              │
│  │  Update   │    V_target = r + γ·V(s')                │
│  └───────────┘                                          │
└─────────────────────────────────────────────────────────┘
```

---

## AI Approaches

| Agent | Method | Description |
|-------|--------|-------------|
| **RL Agent** | TD Learning + MCTS | Self-play with CNN value network, bootstrapped MCTS, prioritized experience replay |
| **Greedy Agent** | Material Evaluation | Evaluates positions by material count (P=1, N/B=3, R=5, Q=9) |
| **Random Agent** | Uniform Random | Selects uniformly from legal moves (baseline) |

### RL Agent Details
- **State Representation**: 8×8×8 tensor (6 piece layers + move counter + turn indicator)
- **Value Network**: Multi-scale CNN with file/rank/diagonal convolutions → Dense layers → scalar V(s)
- **Search**: Monte Carlo Tree Search with Thompson Sampling for node selection
- **Learning**: TD(0) updates with prioritized experience replay (priority = |TD error|)
- **Self-Play**: White uses MCTS + learned value; Black uses greedy opponent for curriculum

---

## Quick Start

### Docker

```bash
docker build -t chess-rl .
docker run chess-rl python train.py --iterations 100 --search-time 1
```

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py --iterations 50 --search-time 1 --network big
```

---

## Training

```bash
# Quick training run (50 games, fast search)
python train.py --iterations 50 --search-time 0.5

# Full training (500 games, deeper search, big network)
python train.py --iterations 500 --search-time 2 --network big

# Lightweight network for faster iteration
python train.py --iterations 200 --search-time 1 --network simple
```

Checkpoints are saved to `checkpoints/` after training.

---

## Evaluation

```bash
# Evaluate against Random and Greedy baselines (default 20 games each)
python evaluate.py --checkpoint checkpoints/model.h5 --games 20

# Evaluate with specific network architecture
python evaluate.py --checkpoint checkpoints/model.h5 --games 50 --network big
```

---

## Benchmark Results

| Opponent | Games | Win | Draw | Loss | Win Rate |
|----------|-------|-----|------|------|----------|
| Random   | 50    | —   | —    | —    | —        |
| Greedy   | 50    | —   | —    | —    | —        |

*Run `python evaluate.py` to populate these results.*

---

## Tech Stack

- **Python 3.12+**
- **TensorFlow / Keras** — CNN value network (multi-scale convolutions)
- **Reinforcement Learning** — TD(0) with bootstrapped returns
- **Monte Carlo Tree Search** — Thompson Sampling node selection
- **Prioritized Experience Replay** — Sampling proportional to |TD error|
- **python-chess** — Legal move generation, board state management

---

## Project Structure

```
chess_ai_reinforcementlearning/
├── train.py                 # CLI training entry point
├── evaluate.py              # Evaluation against baselines
├── Dockerfile               # Reproducible training environment
├── Makefile                 # Build/train/evaluate shortcuts
├── requirements.txt         # Pinned dependencies
├── real_chess/              # Full chess RL (TD + MCTS)
│   ├── agent.py             #   Agent, GreedyAgent, RandomAgent, CNN networks
│   ├── environment.py       #   Board environment (8×8×8 state)
│   ├── learn.py             #   TD_search: MCTS + TD learning loop
│   └── tree.py              #   MCTS Node with Thompson Sampling
├── capture_chess/           # Simplified: capture-focused RL
│   ├── agent.py
│   ├── environment.py
│   └── learn.py
├── move_chess/              # Simplified: move-focused RL
│   ├── agent.py
│   ├── environment.py
│   └── learn.py
└── checkpoints/             # Saved model weights
```

---

## License

GPL-3.0 — see [LICENSE](LICENSE).
