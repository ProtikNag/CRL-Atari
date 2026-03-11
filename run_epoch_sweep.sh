#!/bin/sh
#SBATCH --job-name=crl_epoch_sweep
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p gpu

# =============================================================================
# CRL-Atari  Epoch Count Sweep
# =============================================================================
#
# Sweeps distill_epochs for both "distillation" (pure KD) and "hybrid"
# (HTCL + KD Phase 2) consolidation methods.
#
# Strategy:  Train ONCE for max(EPOCH_VALUES) epochs per method and save
#            intermediate snapshots at each milestone.  This is more
#            efficient (one training loop, not five) and more scientifically
#            valid (same initialization, same data ordering for all points).
#
# Sweep values:  10  100  500  5000  10000
#
# For each method the script:
#   1. Consolidates (one run, snapshots at milestones)
#   2. Evaluates every snapshot checkpoint on all tasks
#   3. Aggregates results into a single JSON with geometric-mean retention
#
# Pre-requisite: expert checkpoints must exist under
#   results/checkpoints/<TAG>/expert_*_best.pt
#   results/checkpoints/<TAG>/expert_summary.json
#
# Usage (local / interactive):
#   ./run_epoch_sweep.sh
#   ./run_epoch_sweep.sh --debug --tag debug
#
# Usage (SLURM):
#   sbatch run_epoch_sweep.sh --tag full_run_v1
#
# =============================================================================

hostname
date

# -- GPU / Environment Setup ---------------------------------------------------

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.3 2>/dev/null || true
module load python3/anaconda/2023.9 2>/dev/null || true

if [ -d "/work/pnag/envs/ml_env" ]; then
    source activate /work/pnag/envs/ml_env/
elif [ -d "venv" ]; then
    . venv/bin/activate
fi

python --version
python -c "import torch; print('torch=' + torch.__version__ + ', CUDA=' + str(torch.cuda.is_available()))"

if [ -d "/work/pnag/CRL-Atari" ]; then
    cd /work/pnag/CRL-Atari/
fi

set -e

# -- Parse arguments -----------------------------------------------------------

DEBUG=""
TAG="default"
DEVICE=""
CONFIG="configs/base.yaml"
EVAL_EPISODES=""
METHODS="distillation hybrid"

while [ $# -gt 0 ]; do
    case $1 in
        --debug)           DEBUG="--debug";           shift ;;
        --tag)             TAG="$2";                  shift 2 ;;
        --device)          DEVICE="--device $2";      shift 2 ;;
        --config)          CONFIG="$2";               shift 2 ;;
        --eval-episodes)   EVAL_EPISODES="--episodes $2"; shift 2 ;;
        --methods)         METHODS="$2";              shift 2 ;;
        -h|--help)
            echo "Usage: ./run_epoch_sweep.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug                Enable debug mode"
            echo "  --tag TAG              Experiment tag (default: 'default')"
            echo "  --device DEVICE        Force device: cpu, cuda, mps"
            echo "  --config PATH          Config file (default: configs/base.yaml)"
            echo "  --eval-episodes N      Override evaluation episode count"
            echo "  --methods \"m1 m2\"      Methods to sweep (default: 'distillation hybrid')"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

COMMON_ARGS="--config ${CONFIG} --tag ${TAG} ${DEBUG} ${DEVICE}"

# -- Sweep Configuration ------------------------------------------------------

EPOCH_VALUES="10,100,500,5000,10000"

# -- Logging -------------------------------------------------------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/logs/epoch_sweep_${TAG}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/pipeline_master.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

run_step() {
    STEP_NAME="$1"; shift
    STEP_LOG="${LOG_DIR}/${STEP_NAME}.log"
    log_msg "START: ${STEP_NAME}"
    log_msg "CMD: $*"
    "$@" 2>&1 | tee -a "${STEP_LOG}" "${MASTER_LOG}"
    EXIT_CODE=${PIPESTATUS[0]:-$?}
    if [ ${EXIT_CODE} -ne 0 ]; then
        log_msg "FAILED: ${STEP_NAME} (exit code ${EXIT_CODE})"
        exit ${EXIT_CODE}
    fi
    log_msg "DONE: ${STEP_NAME}"
    echo "" >> "${MASTER_LOG}"
}

log_msg "============================================================"
log_msg " CRL-Atari  Epoch Count Sweep"
log_msg "============================================================"
log_msg " Tag:     ${TAG}"
log_msg " Config:  ${CONFIG}"
log_msg " Debug:   ${DEBUG:-off}"
log_msg " Device:  ${DEVICE:-auto}"
log_msg " Methods: ${METHODS}"
log_msg " Epochs:  ${EPOCH_VALUES}"
log_msg " Strategy: single run per method with intermediate snapshots"
log_msg " Log:     ${LOG_DIR}"
log_msg "============================================================"

CKPT_DIR="results/checkpoints/${TAG}"

# -- Step 1: Consolidation (one run per method, snapshots at milestones) -------

log_msg "---- Step 1: Consolidation with Snapshots ----"

for METHOD in ${METHODS}; do
    log_msg "--- ${METHOD}: training for max epoch with snapshots at {${EPOCH_VALUES}} ---"
    run_step "consolidate_${METHOD}" \
        python scripts/consolidate.py \
            --method "${METHOD}" \
            --snapshot-epochs "${EPOCH_VALUES}" \
            ${COMMON_ARGS}
done

# -- Step 2: Evaluate Each Snapshot Checkpoint ---------------------------------

log_msg "---- Step 2: Evaluation ----"

# Evaluate experts (only once, skip if already exists)
for EXPERT_CKPT in ${CKPT_DIR}/expert_*_best.pt; do
    if [ -f "${EXPERT_CKPT}" ]; then
        GAME=$(basename "${EXPERT_CKPT}" | sed 's/expert_//;s/_best.pt//')
        OUT_JSON="results/figures/eval_expert_${GAME}_${TAG}.json"
        if [ -f "${OUT_JSON}" ]; then
            log_msg "SKIP: expert ${GAME} (already evaluated)"
            continue
        fi
        log_msg "Evaluating expert: ${GAME}"
        run_step "eval_expert_${GAME}" \
            python scripts/evaluate.py \
                --model-path "${EXPERT_CKPT}" --all-tasks \
                --config ${CONFIG} ${DEBUG} ${DEVICE} ${EVAL_EPISODES} \
                --output "${OUT_JSON}"
    fi
done

# Evaluate each snapshot checkpoint
for METHOD in ${METHODS}; do
    for EP in $(echo "${EPOCH_VALUES}" | tr ',' ' '); do
        CKPT="${CKPT_DIR}/consolidated_${METHOD}_ep${EP}.pt"
        OUT_JSON="results/figures/eval_${METHOD}_ep${EP}_${TAG}.json"
        if [ -f "${CKPT}" ]; then
            log_msg "Evaluating: ${METHOD} ep=${EP}"
            run_step "eval_${METHOD}_ep${EP}" \
                python scripts/evaluate.py \
                    --model-path "${CKPT}" --all-tasks \
                    --config ${CONFIG} ${DEBUG} ${DEVICE} ${EVAL_EPISODES} \
                    --output "${OUT_JSON}"
        else
            log_msg "WARN: no snapshot checkpoint for ${METHOD} ep=${EP}"
        fi
    done

    # Also evaluate the final (full-run) checkpoint
    FINAL_CKPT="${CKPT_DIR}/consolidated_${METHOD}.pt"
    FINAL_JSON="results/figures/eval_${METHOD}_${TAG}.json"
    if [ -f "${FINAL_CKPT}" ]; then
        log_msg "Evaluating: ${METHOD} (final)"
        run_step "eval_${METHOD}_final" \
            python scripts/evaluate.py \
                --model-path "${FINAL_CKPT}" --all-tasks \
                --config ${CONFIG} ${DEBUG} ${DEVICE} ${EVAL_EPISODES} \
                --output "${FINAL_JSON}"
    fi
done

# -- Step 3: Aggregate Sweep Results ------------------------------------------

log_msg "---- Step 3: Aggregate Results ----"

python - "${TAG}" "${METHODS}" "${EPOCH_VALUES}" <<'PYEOF'
import json, glob, os, sys, math

tag = sys.argv[1]
methods = sys.argv[2].split()
epochs = [int(x) for x in sys.argv[3].split(",")]
fig_dir = "results/figures"

# Load expert baselines
expert_rewards = {}
for p in sorted(glob.glob(os.path.join(fig_dir, f"eval_expert_*_{tag}.json"))):
    with open(p) as f:
        data = json.load(f)
    for task_name, info in data.get("results", {}).items():
        game = task_name.split("NoFrameskip")[0]
        expert_rewards[game] = info["mean_reward"]

print(f"\nExpert baselines: {expert_rewards}")

sweep_results = {}
for method in methods:
    sweep_results[method] = {}
    for ep in epochs:
        json_path = os.path.join(fig_dir, f"eval_{method}_ep{ep}_{tag}.json")
        if not os.path.exists(json_path):
            print(f"  MISSING: {json_path}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        per_task = {}
        for task_name, info in data.get("results", {}).items():
            game = task_name.split("NoFrameskip")[0]
            reward = info["mean_reward"]
            expert_r = expert_rewards.get(game, 1.0)
            pct = (reward / expert_r * 100) if expert_r != 0 else 0.0
            per_task[game] = {"reward": round(reward, 2), "retention_pct": round(pct, 2)}
        pcts = [v["retention_pct"] for v in per_task.values()]
        avg_pct = sum(pcts) / len(pcts) if pcts else 0.0
        log_vals = [math.log(max(p, 0.01)) for p in pcts]
        gmean_pct = math.exp(sum(log_vals) / len(log_vals)) if log_vals else 0.0
        sweep_results[method][ep] = {
            "per_task": per_task,
            "avg_retention_pct": round(avg_pct, 2),
            "gmean_retention_pct": round(gmean_pct, 2),
        }

out_path = os.path.join(fig_dir, f"epoch_sweep_{tag}.json")
with open(out_path, "w") as f:
    json.dump({"expert_rewards": expert_rewards, "sweep": sweep_results}, f, indent=2)
print(f"\nSweep results saved to {out_path}")

# Print summary table
games = sorted(expert_rewards)
print("\n" + "=" * 72)
print(f"  Epoch Sweep Summary (tag={tag})")
print("=" * 72)
header = f"{'Method':<14} {'Epochs':>7}  " + "  ".join(f"{g:>12}" for g in games) + "  Avg%  GMean%"
print(header)
print("-" * len(header))
for method in methods:
    for ep in epochs:
        entry = sweep_results.get(method, {}).get(ep)
        if entry is None:
            continue
        row = f"{method:<14} {ep:>7}  "
        for g in games:
            pct = entry["per_task"].get(g, {}).get("retention_pct", 0)
            row += f"  {pct:>10.1f}%"
        row += f"  {entry['avg_retention_pct']:>5.1f} {entry['gmean_retention_pct']:>6.1f}"
        print(row)
print("=" * 72)
PYEOF

# -- Summary -------------------------------------------------------------------

log_msg "============================================================"
log_msg " Epoch Sweep Complete!"
log_msg "============================================================"
log_msg " Figures:     results/figures/epoch_sweep_${TAG}.json"
log_msg " Checkpoints: results/checkpoints/${TAG}/"
log_msg " Logs:        ${LOG_DIR}/"
log_msg "============================================================"

date
