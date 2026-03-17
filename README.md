# CRL-Atari: Continual Reinforcement Learning on Atari Games

Continual Reinforcement Learning (CRL) framework that trains DQN expert agents on a sequence of Atari games and consolidates them into a single model. Implements five consolidation methods (Knowledge Distillation, One-Shot, Iterative, Hybrid, WHC) alongside two baselines (EWC, Multi-task joint training), providing a comprehensive empirical testbed for the theory developed in [Nag, Raghavan, Narayanan 2026].

## Overview

### Problem

Different Atari games are presented **sequentially** as distinct tasks. The goal is to produce a **single consolidated agent** that retains high performance across all previously learned tasks without catastrophic forgetting.

### Key Design Decisions

| Challenge | Solution |
|---|---|
| Different action spaces across games | **Union action space** (6 actions for current task set). Only valid actions per game are used during selection and training. |
| Q-value scale imbalance | **PopArt normalization** rescales output layer weights when target statistics change, preserving network predictions. |
| Catastrophic forgetting | Five consolidation methods + two baselines compared across tasks. |
| Expert diversity | Each expert starts from **random initialization** and trains independently. |
| Action-space mismatch in Taylor updates | **Action-masked drift**: head-layer rows for unused actions are zeroed before applying any Taylor correction. |

### Task Sequence (Default)

1. **Breakout** (4 actions)
2. **SpaceInvaders** (6 actions)
3. **Pong** (6 actions)

## Project Structure

```
CRL-Atari/
├── configs/
│   └── base.yaml                    # All hyperparameters (training, consolidation, debug)
├── src/
│   ├── models/
│   │   └── dqn.py                   # DQN with union action head (standard + dueling)
│   ├── agents/
│   │   └── dqn_agent.py             # Agent: epsilon-greedy, action masking, Double DQN
│   ├── data/
│   │   ├── replay_buffer.py         # Circular replay buffer (uint8 storage)
│   │   └── atari_wrappers.py        # DeepMind-style Atari preprocessing
│   ├── consolidation/
│   │   ├── distillation.py          # Knowledge Distillation (temperature-scaled KL)
│   │   ├── oneshot.py               # One-Shot Joint Consolidation (Section 3, Thm 3.4)
│   │   ├── iterative.py             # Multi-Round Iterative Consolidation (Section 4, Alg 1)
│   │   ├── hybrid.py                # Hybrid: Iterative Taylor + KD refinement (Section 5, Alg 2)
│   │   ├── whc.py                   # Weighted Hessian Consolidation (Eq. 11–13)
│   │   └── htcl.py                  # HTCL: sequential multi-pass Taylor consolidation
│   ├── baselines/
│   │   ├── ewc.py                   # Elastic Weight Consolidation (Kirkpatrick et al. 2017)
│   │   └── multitask.py             # Multi-task joint training (upper-bound oracle)
│   ├── trainers/
│   │   └── expert_trainer.py        # Training loop + reward curve generation
│   └── utils/
│       ├── config.py                # YAML config loading + debug mode
│       ├── seed.py                  # Reproducibility (all RNG seeds)
│       ├── logger.py                # TensorBoard + CSV + console logging
│       └── normalization.py         # PopArt Q-value normalizer
├── scripts/
│   ├── train_experts.py             # Train expert DQN per task (sequential)
│   ├── train_ewc.py                 # Sequential EWC training
│   ├── train_multitask.py           # Multi-task joint training
│   ├── consolidate.py               # Merge experts (distillation / oneshot / iterative / hybrid / whc)
│   ├── evaluate.py                  # Evaluate any checkpoint on any task
│   ├── compare.py                   # Comprehensive comparison + plots
│   ├── visualize.py                 # Core visualization suite (4 focused figures)
│   ├── visualize_loss_landscape.py  # 2D/3D loss landscape in PCA-projected weight space
│   ├── visualize_hybrid.py          # Hybrid method deep-dive analysis
│   ├── visualize_qvalues.py         # Q-value distribution analysis (KDE, violins, heatmaps)
│   ├── visualize_umap.py            # UMAP feature space visualization
│   ├── generate_report.py           # Generate HTML technical report
│   └── play.py                      # Watch a trained agent play (pygame GUI)
├── docs/
│   └── report.html                  # Technical experiment report
├── main.py                          # Debug-friendly entry point (runs locally)
├── run_train_experts.sh             # Expert training pipeline (SLURM-compatible)
├── run_consolidate.sh               # Consolidation + evaluation pipeline
├── run_epoch_sweep.sh               # Distillation epoch-count sweep
├── requirements.txt
├── results/
│   ├── logs/                        # TensorBoard + CSV logs
│   ├── checkpoints/                 # Model checkpoints
│   └── figures/                     # All plots (PNG + SVG) + eval JSON files
└── notebooks/                       # Analysis only
```

## Quickstart

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gymnasium[accept-rom-license]   # Accept Atari ROM license (once)
```

### 2. Train Experts

```bash
# Train on Hyperion (SLURM)
sbatch run_train_experts.sh --tag full_run_v1

# Or locally in debug mode
python scripts/train_experts.py --debug --tag debug
```

Training runs for **3M environment steps** per expert (configurable in `configs/base.yaml`).

### 3. Consolidate and Evaluate

```bash
# Full consolidation + evaluation + comparison plots
./run_consolidate.sh --tag full_run_v1

# Debug mode (fast)
./run_consolidate.sh --debug --tag debug

# Skip consolidation, only re-evaluate and re-plot
./run_consolidate.sh --skip-consolidate --tag full_run_v1
```

### 4. Debug Locally

```bash
python main.py                                        # Full pipeline (debug)
python main.py --step train                           # Only experts
python main.py --step consolidate --method hybrid     # Only consolidate
python main.py --step compare                         # Only compare
python main.py --no-debug --tag full_run              # Full training
```

### 5. Run Individual Scripts

```bash
# Consolidate with any method
python scripts/consolidate.py --method distillation --debug --tag myexp
python scripts/consolidate.py --method oneshot       --debug --tag myexp
python scripts/consolidate.py --method iterative     --debug --tag myexp
python scripts/consolidate.py --method hybrid        --debug --tag myexp
python scripts/consolidate.py --method whc           --debug --tag myexp

# EWC and multi-task (separate training scripts)
python scripts/train_ewc.py       --debug --tag myexp
python scripts/train_multitask.py --debug --tag myexp

# Evaluate a checkpoint on all tasks
python scripts/evaluate.py --model-path results/checkpoints/myexp/consolidated_hybrid.pt --all-tasks --debug

# Generate comparison plots
python scripts/compare.py --debug --tag myexp

# Distillation epoch-count sweep
./run_epoch_sweep.sh --tag myexp
```

### 6. Watch an Agent Play

```bash
python scripts/play.py --game Pong
python scripts/play.py --game Breakout --speed 2
python scripts/play.py --game SpaceInvaders --model-path results/checkpoints/default/consolidated_hybrid.pt
python scripts/play.py --list-games
```

**Controls**: `Q`/`ESC` quit, `R` restart, `P`/`Space` pause, `+`/`-` speed.

## Consolidation Methods

All methods share a common interface: load expert checkpoints, collect high-confidence replay data, apply the consolidation update, and save the merged model.

### 1. Knowledge Distillation

Trains a student (global) model to match the soft Q-value distributions of all expert teachers using temperature-scaled KL divergence. Q-values are normalized per-task (zero-mean, unit-variance) before applying softmax to handle reward scale differences.

$$\mathcal{L}_\text{KD} = \frac{1}{N} \sum_{i} T^2 \, D_\text{KL}\!\left(\sigma(Q_\text{expert}^{(i)}/T) \;\|\; \sigma(Q_\text{student}/T)\right)$$

**Key hyperparameters**:
- `temperature`: Softmax temperature (default: 0.01)
- `epochs`: Training epochs (default: 10,000)
- `lr`: Learning rate (default: 5e-5)

### 2. One-Shot Joint Consolidation

Single closed-form Taylor step at the ensemble-mean anchor. Fisher and gradient are averaged over all experts; the drift centroid $\mathbf{d}^*$ vanishes by symmetry (Remark 3.6), reducing the update to a Newton step:

$$\mathbf{u}^* = -(\bar{\mathbf{F}} + \lambda \mathbf{I})^{-1}\bar{\mathbf{g}}, \quad \mathbf{w}_g = \bar{\mathbf{w}} + \mathbf{u}^*$$

No step size or iteration is used. Fisher is computed on 20K high-confidence states per task (ranked by Q-value gap).

### 3. Iterative Consolidation

Extends One-Shot to $K$ rounds. Each round re-expands around the current global model and applies a decaying Taylor correction:

$$\mathbf{w}_g^{(k+1)} = \mathbf{w}_g^{(k)} + \eta_k \left(\bar{\mathbf{F}}_k + \lambda \mathbf{I}\right)^{-1}\!\left[\lambda \mathbf{d}_k^* - \bar{\mathbf{g}}_k\right]$$

**Key hyperparameters**:
- `num_rounds`: $K$ (default: 10)
- `eta`: Initial step size $\eta_0$ (default: 0.9)
- `gamma`: Step-size decay $\gamma$ (default: 0.9)
- `recompute_fisher`: Recompute Fisher each round or cache (default: false)

### 4. Hybrid (Iterative Taylor + KD)

Two-phase consolidation. Phase 1 provides a Taylor warm start (identical to Iterative above). Phase 2 refines via knowledge distillation starting from the warm-start solution rather than the raw ensemble mean. Theorem 5.2 bounds the reduction in KD convergence gap as a function of the Taylor approximation quality.

- **Phase 1**: $K=10$ rounds, $\eta_0=0.9$, $\gamma=0.9$, cached Fisher
- **Phase 2**: KD for 10,000 epochs, $T=0.01$, lr $= 5\times 10^{-5}$

Epoch snapshots are saved at 10, 100, 500, 5K, and 10K epochs for convergence analysis.

### 5. Weighted Hessian Consolidation (WHC)

Closed-form solution derived from a surrogate loss formed by Hessian-weighted quadratic approximations expanded **at each expert's own optimum** (not a shared anchor). This placement minimizes the Taylor remainder and eliminates the gradient term entirely.

$$\hat{\mathbf{w}}_\lambda = \left(\sum_i \alpha_i \mathbf{H}_i + \lambda \mathbf{I}\right)^{-1} \sum_i \alpha_i \mathbf{H}_i \mathbf{w}_i^*$$

**Key hyperparameters**:
- `lambda_reg`: Tikhonov regularization (default: 1.0)
- `fisher_samples`: States for Fisher estimation (default: 20,000)

## Baselines

### EWC (Elastic Weight Consolidation)

Sequential continual learning with a Fisher-weighted L2 penalty that anchors parameters toward previous task solutions during fine-tuning:

$$\mathcal{L}_\text{EWC}(\theta) = \mathcal{L}_\text{task}(\theta) + \frac{\lambda}{2} \sum_{i,j} F_i^j \left(\theta_j - \theta_{i,j}^*\right)^2$$

Implements both standard EWC (per-task Fisher storage) and Online EWC (Schwarz et al. 2018, exponential Fisher averaging). Trained via `scripts/train_ewc.py` sequentially on the task order.

**Key hyperparameters**: `lambda`: 5000.0, `fisher_samples`: 5000, `gamma_ewc`: 0.95

### Multi-Task Joint Training

A single DQN trained simultaneously on all tasks via round-robin environment stepping and random task sampling. Serves as an oracle upper bound since it is never subject to catastrophic forgetting. Trained via `scripts/train_multitask.py` for 6M total steps (~2M per task).

## Visualizations

### Expert Training

Reward curves are generated automatically at the end of each expert's run:
- `01_expert_training_curves.{png,svg}` — combined multi-panel reward curves
- `02_retention_heatmap.{png,svg}` — per-method expert retention heatmap
- `03_sample_efficiency.{png,svg}` — reward vs environment steps
- `04_reward_distributions.{png,svg}` — per-episode reward box plots

### Loss Landscape (`scripts/visualize_loss_landscape.py`)

PCA-projected 2D/3D loss landscapes showing expert optima and consolidated model positions:
- `loss_landscape_combined_default.{png,svg}` — all experts + all methods (2D)
- `loss_landscape_per_game_default.{png,svg}` — per-game landscapes
- `loss_landscape_3d_default.{png,svg}` — 3D surface

### Q-Value Analysis (`scripts/visualize_qvalues.py`)

- `qvalue_maxq_kde_default.{png,svg}` — max-Q KDE per game
- `qvalue_action_violins_default.{png,svg}` — per-action Q distributions
- `qvalue_action_heatmap_default.{png,svg}` — action-preference heatmap
- `qvalue_umap_default.{png,svg}` — UMAP of Q-value vectors

### Feature Space (`scripts/visualize_umap.py`)

- `umap_expert_default.{png,svg}` — UMAP of penultimate-layer activations across tasks

### Distillation Epoch Sweep (`run_epoch_sweep.sh`)

Sweeps distillation epochs across {10, 100, 500, 5K, 10K} for both Distillation and Hybrid, saving intermediate checkpoints and evaluation JSON files for convergence analysis.

## Configuration

All hyperparameters live in [`configs/base.yaml`](configs/base.yaml).

- **Debug mode**: `--debug` reduces training to ~10K steps. Ideal for pipeline testing.
- **Override configs**: `--override-config path/to/override.yaml` for selective overrides.
- **Per-run saving**: Each run saves its effective config to the log directory.

### Key Config Sections

| Section | Purpose |
|---|---|
| `task_sequence` | Ordered list of Atari environments |
| `model` | CNN architecture: channels, kernels, FC size |
| `training` | Expert training: LR, buffer, exploration, etc. |
| `normalization` | PopArt Q-value normalization settings |
| `consolidation` | Shared consolidation settings (fisher samples, buffer size) |
| `distillation`, `oneshot`, `iterative`, `hybrid`, `whc` | Method-specific hyperparameters |
| `ewc`, `multitask` | Baseline hyperparameters |
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

The union action space is computed at runtime from the minimal action sets of all games in the task sequence. For {Breakout, SpaceInvaders, Pong}, the union is {NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE} = 6 actions. Per-game action masking sets invalid actions' Q-values to $-\infty$ during selection and training.

**Double DQN** is enabled by default — policy network selects actions, target network evaluates them.

## Outputs

After a full run:

| Path | Content |
|---|---|
| `results/checkpoints/<tag>/expert_*_best.pt` | Best expert per game |
| `results/checkpoints/<tag>/consolidated_distillation.pt` | Distillation model |
| `results/checkpoints/<tag>/consolidated_distillation_ep*.pt` | Epoch snapshots (10, 100, 500, 5K, 10K) |
| `results/checkpoints/<tag>/consolidated_oneshot.pt` | One-Shot model |
| `results/checkpoints/<tag>/consolidated_iterative.pt` | Iterative model |
| `results/checkpoints/<tag>/consolidated_hybrid.pt` | Hybrid model |
| `results/checkpoints/<tag>/consolidated_hybrid_ep*.pt` | Hybrid epoch snapshots |
| `results/checkpoints/<tag>/consolidated_whc.pt` | WHC model |
| `results/checkpoints/<tag>/consolidated_ewc.pt` | EWC model |
| `results/checkpoints/<tag>/htcl_fisher_log.json` | Fisher / Hessian diagnostics |
| `results/figures/png/` | All comparison plots (PNG) |
| `results/figures/svg/` | All comparison plots (SVG) |
| `results/figures/eval_*_<tag>.json` | Per-method evaluation results |
| `results/figures/comparison_results_<tag>.json` | Numerical results summary |
| `results/figures/comparison_full_<tag>.json` | Full results with per-episode rewards |
| `results/logs/` | TensorBoard events + CSV metrics |

## Shell Script Options

```
./run_consolidate.sh [OPTIONS]

Options:
  --debug              Debug mode (fast, uses reduced settings)
  --tag TAG            Experiment tag (default: 'default')
  --device DEVICE      Force device: cpu, cuda, mps
  --config PATH        Config file (default: configs/base.yaml)
  --skip-consolidate   Skip consolidation (only evaluate & plot)
  --eval-episodes N    Override evaluation episode count
```

## References

- Nag, Raghavan, Narayanan, "Mitigating Task-Order Sensitivity and Forgetting via Hierarchical Second-Order Consolidation", 2026 (One-Shot, Iterative, Hybrid, WHC)
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017 (EWC)
- Schwarz et al., "Progress & compress: A scalable framework for continual learning", ICML 2018 (Online EWC)
- Hinton et al., "Distilling the knowledge in a neural network", NeurIPS Workshop 2014 (Knowledge Distillation)
- Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015 (DQN)
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2016 (Double DQN)
- van Hasselt et al., "Learning values across many orders of magnitude", NeurIPS 2016 (PopArt)
