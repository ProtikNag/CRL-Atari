# CRL-Atari: Continual Reinforcement Learning on Atari with HTCL Consolidation

Continual RL experiments comparing HTCL (Hierarchical Taylor Series-based Continual Learning) consolidation against standard baselines on Atari games.

## Overview

Three DQN agents are trained on Pong, Breakout, and SpaceInvaders independently (local agents). A consolidated global agent is then produced using HTCL's second-order Taylor expansion, and compared against:

- **Individual**: Per-game DQN agents (upper bound, no consolidation needed)
- **Sequential**: Naive fine-tuning through games in order (demonstrates catastrophic forgetting)
- **EWC**: Elastic Weight Consolidation (Fisher-based regularization)
- **HTCL**: Taylor-series consolidation from the HTCL paper (Eq. 6)

All methods share a unified 6-action discrete action space with per-game action masking.

## Project Structure

```
crl_atari/
├── main.py                 # Entry point (orchestrates training, eval, plots)
├── utils.py                # Seed, device, config loading, ResultsManager
├── config.yaml             # All hyperparameters (single source of truth)
├── requirements.txt        # Python dependencies
├── run.sh                  # SLURM submission script
├── agents/
│   ├── networks.py         # Nature DQN architecture
│   └── dqn.py              # DQN agent, replay buffer, training loop
├── envs/
│   └── wrappers.py         # Atari preprocessing, unified action wrapper
├── consolidation/
│   ├── taylor.py           # HTCL Taylor update adapted for DQN
│   ├── ewc.py              # EWC baseline
│   └── sequential.py       # Naive sequential fine-tuning
├── evaluation/
│   └── metrics.py          # Reward evaluation, forgetting, transfer
└── visualization/
    └── plots.py            # Publication-ready plots (PNG + SVG)
```

## Setup

```bash
pip install -r requirements.txt
```

Atari ROMs are auto-installed via `gymnasium[atari,accept-rom-license]`.

## Usage

### Quick test (50k steps per game)

```bash
python main.py --config config.yaml --quick
```

### Full experiment

```bash
python main.py --config config.yaml --seed 42 --train-steps 500000
```

### Run specific methods

```bash
python main.py --config config.yaml --methods htcl ewc
```

### SLURM cluster

```bash
sbatch run.sh
```

## Configuration

All hyperparameters are in `config.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `dqn.train_steps` | 500000 | Environment steps per game |
| `dqn.lr` | 0.00025 | Adam learning rate |
| `dqn.buffer_size` | 50000 | Replay buffer capacity |
| `consolidation.htcl.eta` | 0.9 | Taylor update step size |
| `consolidation.htcl.lambda_reg` | 1000.0 | Hessian regularization strength |
| `consolidation.htcl.catchup_iterations` | 5 | Catch-up phase iterations |
| `consolidation.ewc.lambda_ewc` | 5000.0 | EWC penalty weight |

## Output

Results are saved to `./results/` (configurable via `--output-dir`). The directory and all subdirectories are created automatically on first run.

```
results/
├── csv/                    # Tabular results
├── json/                   # Config + summary
├── checkpoints/            # Model weights (.pt)
└── plots/
    ├── png/                # 300 DPI raster plots
    └── svg/                # Vector plots
```

### Generated Plots

- `per_game_rewards`: Grouped bar chart of mean reward per game per method
- `forgetting`: Catastrophic forgetting per game
- `forgetting_trajectory`: Reward evolution across training stages
- `performance_heatmap`: Method × game reward matrix
- `aggregate_summary`: Overall mean reward comparison
- `radar_comparison`: Spider chart across games
- `training_curves`: Local agent learning curves

## Evaluation Metrics

- **Mean Reward ± Std**: Average episode reward across evaluation episodes
- **Forgetting**: Peak reward on game $i$ minus final reward after all games trained
- **Forward Transfer**: Performance on unseen game before training on it
- **Aggregate**: Mean reward across all 3 games (single-number summary)

## Adapting the HTCL Update for RL

The original HTCL uses cross-entropy loss for gradient/Hessian computation. In the DQN setting:

- **Loss**: Smooth L1 (Huber) TD loss replaces cross-entropy
- **Data**: Transitions $(s, a, r, s', d)$ from a consolidation buffer replace classification samples
- **Hessian**: Diagonal Fisher approximation (squared gradients) of the TD loss
- **Update rule**: Identical to Eq. (6): $w_g \leftarrow w_g + (H + \lambda I)^{-1}[\lambda(w_l - w_g) - g]$

## Reproducibility

- Seeds are set for Python, NumPy, and PyTorch (including CUDA)
- Full config is saved to `results/json/config.json`
- All model checkpoints are saved
- `cudnn.deterministic = True` is enforced
