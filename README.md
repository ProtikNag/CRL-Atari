# CRL-Atari: Continual Reinforcement Learning on Atari Games

Continual Reinforcement Learning (CRL) framework that trains DQN expert agents on a sequence of Atari games and consolidates them into a single model using three knowledge consolidation methods: **EWC**, **Knowledge Distillation**, and **HTCL**.

## Overview

### Problem

Different Atari games are presented **sequentially** as distinct tasks. The goal is to produce a **single consolidated agent** that retains high performance across all previously learned tasks without catastrophic forgetting.

### Key Design Decisions

| Challenge | Solution |
|---|---|
| Different action spaces across games | **Unified 18-action DQN** head covering all Atari joystick actions. Invalid actions are masked per-game. |
| Q-value scale imbalance | **Running-stats normalization** brings Q-values to comparable scales before consolidation. |
| Expert drift from global model | Each expert is **initialized from the current global state** (HTCL-style) to limit parameter drift. |
| Catastrophic forgetting | Three consolidation methods compared: EWC, Distillation, HTCL. |

### Task Sequence (Default)

1. **Pong** (6 actions)
2. **Breakout** (4 actions)
3. **SpaceInvaders** (6 actions)

## Project Structure

```
CRL-Atari/
├── configs/
│   └── base.yaml              # All hyperparameters (training, consolidation, debug)
├── src/
│   ├── models/
│   │   └── dqn.py             # DQN with unified 18-action head
│   ├── agents/
│   │   └── dqn_agent.py       # Agent: epsilon-greedy, action masking, training step
│   ├── data/
│   │   ├── replay_buffer.py   # Circular replay buffer (uint8 storage)
│   │   └── atari_wrappers.py  # DeepMind-style Atari preprocessing
│   ├── consolidation/
│   │   ├── ewc.py             # Elastic Weight Consolidation
│   │   ├── distillation.py    # Knowledge Distillation (temperature-scaled)
│   │   └── htcl.py            # Hierarchical Taylor-based Continual Learning
│   ├── trainers/
│   │   └── expert_trainer.py  # Full training loop for one expert
│   └── utils/
│       ├── config.py          # YAML config loading + debug mode
│       ├── seed.py            # Reproducibility (all RNG seeds)
│       ├── logger.py          # TensorBoard + CSV + console logging
│       └── normalization.py   # Q-value running-stats / PopArt normalizer
├── scripts/
│   ├── train_experts.py       # Train expert DQN per task (sequential)
│   ├── consolidate.py         # Merge experts via EWC / Distillation / HTCL
│   ├── evaluate.py            # Evaluate any model on any task
│   └── compare.py             # Side-by-side comparison + plots
├── main.py                    # Debug-friendly entry point (runs locally)
├── run_all.sh                 # Full experiment pipeline in one command
├── configs/base.yaml          # All hyperparameters
├── requirements.txt
├── results/
│   ├── logs/                  # TensorBoard + CSV logs
│   ├── checkpoints/           # Model checkpoints
│   └── figures/               # Comparison plots (PNG + SVG)
└── notebooks/                 # Analysis only
```

## Quickstart

### 1. Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Accept Atari ROM license (required once)
pip install gymnasium[accept-rom-license]
```

### 2. Run Full Pipeline

```bash
# Full experiment (trains experts, consolidates, evaluates, plots)
./run_all.sh

# Debug mode (fast, ~5000 steps per expert)
./run_all.sh --debug

# Custom tag and device
./run_all.sh --tag experiment_v1 --device cuda
```

### 3. Debug Locally (No Shell Script)

```bash
# Run everything in debug mode from Python directly
python main.py

# Only train experts
python main.py --step train

# Only consolidate (after training)
python main.py --step consolidate --method htcl

# Only compare (after all checkpoints exist)
python main.py --step compare

# Full training (non-debug)
python main.py --no-debug --tag full_run
```

### 4. Run Individual Scripts

```bash
# Train experts
python scripts/train_experts.py --debug --tag myexp

# Consolidate with a specific method
python scripts/consolidate.py --method htcl --debug --tag myexp
python scripts/consolidate.py --method ewc --debug --tag myexp
python scripts/consolidate.py --method distillation --debug --tag myexp

# Evaluate a checkpoint on all tasks
python scripts/evaluate.py --model-path results/checkpoints/myexp/consolidated_htcl.pt --all-tasks --debug

# Generate comparison plots
python scripts/compare.py --debug --tag myexp
```

## Consolidation Methods

### 1. EWC (Elastic Weight Consolidation)

Computes a diagonal Fisher Information Matrix per task to identify important parameters. During consolidation, deviations from each expert's optimal parameters are penalized proportionally to Fisher importance.

**Key hyperparameters** (in `configs/base.yaml`):
- `lambda_ewc`: Penalty strength (default: 5000)
- `fisher_samples`: Samples for Fisher computation (default: 2000)
- `online`: Whether to use online EWC with decaying Fisher (default: true)

### 2. Knowledge Distillation

Trains a student (global) model to match the soft Q-value distributions of all teachers (experts) using temperature-scaled softmax. Q-values are normalized per-task before distillation.

**Key hyperparameters**:
- `temperature`: Softmax temperature (default: 2.0)
- `alpha`: Distillation vs task loss weight (default: 0.5)
- `distill_epochs`: Training epochs (default: 50)

### 3. HTCL (Hierarchical Taylor-based Continual Learning)

From [Nag, Raghavan, Narayanan 2026]. Uses a second-order Taylor expansion around the global model to find an optimal parameter update that balances:
- **Stability**: staying in low-curvature regions of the past-task loss landscape
- **Plasticity**: moving toward the new expert's parameters

$$\mathbf{w}^{(t)} = \mathbf{w}^{(t-1)} + (\mathbf{H} + \lambda \mathbf{I})^{-1} [\lambda \Delta\mathbf{d} - \mathbf{g}]$$

**Lambda constraint**: $\lambda > -\mu_{\min}(\mathbf{H})$ ensures the surrogate objective is strictly convex. With `lambda_auto: true`, this is enforced automatically.

**Key hyperparameters**:
- `lambda_htcl`: Base lambda value (default: 1.0)
- `lambda_auto`: Auto-adjust to satisfy constraint (default: true)
- `catch_up_iterations`: Refinement iterations after each merge (default: 2)
- `diagonal_fisher`: Use diagonal Fisher for Hessian approximation (default: true)

## Configuration

All hyperparameters live in [`configs/base.yaml`](configs/base.yaml). The config system supports:

- **Debug mode**: Set `--debug` to use reduced training (~5000 steps). Ideal for testing pipeline correctness.
- **Override configs**: Pass `--override-config path/to/override.yaml` to selectively override base settings.
- **Per-run saving**: Each run saves its effective config to the log directory.

### Key Config Sections

| Section | Purpose |
|---|---|
| `task_sequence` | Ordered list of Atari environments |
| `model` | CNN architecture: channels, kernels, FC size |
| `training` | Expert training: LR, buffer, exploration, etc. |
| `normalization` | Q-value normalization settings |
| `consolidation` | Global-to-local initialization flag |
| `ewc`, `distillation`, `htcl` | Method-specific hyperparameters |
| `debug` | Reduced values for fast local testing |

## DQN Architecture

```
Input: (batch, 4, 84, 84)  [4 stacked grayscale frames]
  └─ Conv2d(4→64, 8x8, stride 4) + ReLU
  └─ Conv2d(64→128, 4x4, stride 2) + ReLU
  └─ Conv2d(128→128, 3x3, stride 1) + ReLU
  └─ Flatten
  └─ Linear(→1024) + ReLU
  └─ Linear(1024→18)  [unified action head]
```

**Parameter count**: ~1.7M (sufficient for 3+ games).

The unified 18-action output covers all possible Atari joystick positions. Per-game action masking sets invalid actions' Q-values to $-\infty$ during action selection and training.

## Outputs

After a full run, you'll find:

| Path | Content |
|---|---|
| `results/checkpoints/<tag>/expert_*_best.pt` | Best expert per game |
| `results/checkpoints/<tag>/consolidated_ewc.pt` | EWC-merged model |
| `results/checkpoints/<tag>/consolidated_distillation.pt` | Distillation-merged model |
| `results/checkpoints/<tag>/consolidated_htcl.pt` | HTCL-merged model |
| `results/figures/comparison_bar.{png,svg}` | Bar chart: expert vs. consolidated |
| `results/figures/performance_heatmap.{png,svg}` | Heatmap: method x game |
| `results/figures/comparison_results_<tag>.json` | Raw numerical results |
| `results/logs/` | TensorBoard events + CSV metrics |

## Shell Script Options

```
./run_all.sh [OPTIONS]

Options:
  --debug              Debug mode (fast training, ~5K steps)
  --tag TAG            Experiment tag (default: 'default')
  --device DEVICE      Force device: cpu, cuda, mps
  --config PATH        Config file (default: configs/base.yaml)
  --skip-train         Skip expert training (use existing checkpoints)
  --skip-consolidate   Skip consolidation (only evaluate)
  --eval-episodes N    Override evaluation episode count
```

## References

- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017 (EWC)
- Hinton et al., "Distilling the knowledge in a neural network", 2015 (Knowledge Distillation)
- Nag, Raghavan, Narayanan, "Mitigating Task-Order Sensitivity and Forgetting via Hierarchical Second-Order Consolidation", 2026 (HTCL)
- Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015 (DQN)
