# CRL-Atari: Continual Reinforcement Learning on Atari Games

Continual Reinforcement Learning (CRL) framework that trains DQN expert agents on a sequence of Atari games and consolidates them into a single model using three knowledge consolidation methods: **EWC**, **Knowledge Distillation**, and **HTCL**.

## Overview

### Problem

Different Atari games are presented **sequentially** as distinct tasks. The goal is to produce a **single consolidated agent** that retains high performance across all previously learned tasks without catastrophic forgetting.

### Key Design Decisions

| Challenge | Solution |
|---|---|
| Different action spaces across games | **Union action space** (6 actions for current task set). Only valid actions per game are used during selection and training. |
| Q-value scale imbalance | **PopArt normalization** rescales output layer weights when target statistics change, preserving network predictions. |
| Catastrophic forgetting | Three consolidation methods compared: EWC, Distillation, HTCL. |
| Expert diversity | Each expert starts from **random initialization** and trains independently. |

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
│   │   └── dqn.py             # DQN with union action head
│   ├── agents/
│   │   └── dqn_agent.py       # Agent: epsilon-greedy, action masking, Double DQN
│   ├── data/
│   │   ├── replay_buffer.py   # Circular replay buffer (uint8 storage)
│   │   └── atari_wrappers.py  # DeepMind-style Atari preprocessing
│   ├── consolidation/
│   │   ├── ewc.py             # Elastic Weight Consolidation
│   │   ├── distillation.py    # Knowledge Distillation (temperature-scaled)
│   │   └── htcl.py            # Hierarchical Taylor-based Continual Learning
│   ├── trainers/
│   │   └── expert_trainer.py  # Training loop + inline reward curve generation
│   └── utils/
│       ├── config.py          # YAML config loading + debug mode
│       ├── seed.py            # Reproducibility (all RNG seeds)
│       ├── logger.py          # TensorBoard + CSV + console logging
│       └── normalization.py   # PopArt Q-value normalizer
├── scripts/
│   ├── train_experts.py       # Train expert DQN per task (sequential)
│   ├── consolidate.py         # Merge experts via EWC / Distillation / HTCL
│   ├── evaluate.py            # Evaluate any model on any task
│   ├── compare.py             # Side-by-side comparison + plots
│   ├── visualize.py           # Additional visualization utilities
│   ├── generate_report.py     # Generate HTML technical report
│   └── play.py                # Watch a trained agent play (pygame GUI)
├── main.py                    # Debug-friendly entry point (runs locally)
├── run_all.sh                 # Full experiment pipeline (SLURM-compatible)
├── requirements.txt
├── results/
│   ├── logs/                  # TensorBoard + CSV logs
│   ├── checkpoints/           # Model checkpoints (best per expert)
│   └── figures/               # Reward curves and comparison plots (PNG + SVG)
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

# Debug mode (fast, ~10K steps per expert)
./run_all.sh --debug

# Custom tag and device
./run_all.sh --tag experiment_v1 --device cuda
```

The pipeline trains each expert for **3M environment steps** (12M frames with frame-skip=4, configurable in `configs/base.yaml`). Reward curves are automatically generated during training and saved to `results/figures/`.

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

### 5. Watch an Agent Play

```bash
# Watch the best Pong expert play
python scripts/play.py --game Pong

# Watch with faster playback
python scripts/play.py --game Breakout --speed 2

# Watch a consolidated model play
python scripts/play.py --game SpaceInvaders --model-path results/checkpoints/default/consolidated_htcl.pt

# List available games and checkpoints
python scripts/play.py --list-games
```

**Controls**: `Q`/`ESC` quit, `R` restart, `P`/`Space` pause, `+`/`-` speed.

## Training & Visualization

Expert training automatically generates reward curves at the end of each expert's training run. Both per-game and combined multi-panel figures are saved in `results/figures/{png,svg}/`.

Generated figures:
- `expert_{Game}_reward_curve.{png,svg}` — per-game curve with raw + smoothed rewards and best-point marker
- `expert_all_reward_curves.{png,svg}` — combined side-by-side panel for all games

Training also resumes from the best checkpoint if one exists, so re-running `run_all.sh` continues from where you left off.

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
- `lambda_htcl`: Base lambda value (default: 100000)
- `lambda_auto`: Auto-adjust to satisfy constraint (default: true)
- `catch_up_iterations`: Refinement iterations after each merge (default: 10)
- `diagonal_fisher`: Use diagonal Fisher for Hessian approximation (default: true)

## Configuration

All hyperparameters live in [`configs/base.yaml`](configs/base.yaml). The config system supports:

- **Debug mode**: Set `--debug` to use reduced training (~10K steps). Ideal for testing pipeline correctness.
- **Override configs**: Pass `--override-config path/to/override.yaml` to selectively override base settings.
- **Per-run saving**: Each run saves its effective config to the log directory.

### Key Config Sections

| Section | Purpose |
|---|---|
| `task_sequence` | Ordered list of Atari environments |
| `model` | CNN architecture: channels, kernels, FC size |
| `training` | Expert training: LR, buffer, exploration, etc. |
| `normalization` | PopArt Q-value normalization settings |
| `consolidation` | Shared consolidation settings |
| `ewc`, `distillation`, `htcl` | Method-specific hyperparameters |
| `evaluation` | Eval episodes, deterministic flag |
| `logging` | Log/checkpoint/figure directories, TensorBoard |
| `debug` | Reduced values for fast local testing |

## DQN Architecture

```
Input: (batch, 4, 84, 84)  [4 stacked grayscale frames]
  └─ Conv2d(4→32, 8×8, stride 4) + ReLU
  └─ Conv2d(32→64, 4×4, stride 2) + ReLU
  └─ Conv2d(64→64, 3×3, stride 1) + ReLU
  └─ Flatten
  └─ Linear(→512) + ReLU
  └─ Linear(512→6)  [union action head: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE]
```

**Parameter count**: ~1.7M.

The union action space is computed at runtime from the minimal action sets of all games in the task sequence. For {Pong, Breakout, SpaceInvaders}, the union is [0, 1, 3, 4, 11, 12] = [NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE] = 6 actions. Per-game action masking sets invalid actions' Q-values to $-\infty$ during action selection and training.

**Double DQN** is enabled by default — the policy network selects actions while the target network evaluates them, reducing overestimation bias.

## Outputs

After a full run, you'll find:

| Path | Content |
|---|---|
| `results/checkpoints/<tag>/expert_*_best.pt` | Best expert per game |
| `results/checkpoints/<tag>/consolidated_ewc.pt` | EWC-merged model |
| `results/checkpoints/<tag>/consolidated_distillation.pt` | Distillation-merged model |
| `results/checkpoints/<tag>/consolidated_htcl.pt` | HTCL-merged model |
| `results/figures/png/expert_*_reward_curve.png` | Per-game training reward curves |
| `results/figures/png/expert_all_reward_curves.png` | Combined multi-panel reward curves |
| `results/figures/png/comparison_bar.png` | Bar chart: expert vs. consolidated |
| `results/figures/png/performance_heatmap.png` | Heatmap: method × game |
| `results/figures/comparison_results_<tag>.json` | Raw numerical results |
| `results/logs/` | TensorBoard events + CSV metrics |

## Shell Script Options

```
./run_all.sh [OPTIONS]

Options:
  --debug              Debug mode (fast training, ~10K steps)
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
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2016 (Double DQN)
- van Hasselt et al., "Learning values across many orders of magnitude", NeurIPS 2016 (PopArt)
