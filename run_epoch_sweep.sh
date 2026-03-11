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
# Sweep values:  10  100  500  5000  10000
#
# For each epoch count the script:
#   1. Consolidates (trains the student/hybrid model)
#   2. Evaluates the resulting checkpoint on all tasks
#   3. Saves per-sweep JSON results
#
# After all sweep points are done a final comparison aggregates everything
# and reports geometric-mean retention.
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

EPOCH_VALUES="10 100 500 5000 10000"

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
log_msg " Log:     ${LOG_DIR}"
log_msg "============================================================"

CKPT_DIR="results/checkpoints/${TAG}"

# -- Step 1: Consolidation Sweep ----------------------------------------------

log_msg "---- Step 1: Consolidation Sweep ----"

for METHOD in ${METHODS}; do
    for EP in ${EPOCH_VALUES}; do
        SUFFIX="_ep${EP}"
        CKPT="${CKPT_DIR}/consolidated_${METHOD}${SUFFIX}.pt"
        if [ -f "${CKPT}" ]; then
            log_msg "SKIP: ${METHOD} ep=${EP} (checkpoint exists: ${CKPT})"
            continue
        fi
        log_msg "--- ${METHOD} ep=${EP} ---"
        run_step "consolidate_${METHOD}_ep${EP}" \
            python scripts/consolidate.py \
                --method "${METHOD}" \
                --distill-epochs "${EP}" \
                --save-suffix "${SUFFIX}" \
                ${COMMON_ARGS}
    done
done

# -- Step 2: Evaluate Each Sweep Checkpoint ------------------------------------

log_msg "---- Step 2: Evaluation ----"

# Evaluate experts (only once)
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

# Evaluate each sweep checkpoint
for METHOD in ${METHODS}; do
    for EP in ${EPOCH_VALUES}; do
        SUFFIX="_ep${EP}"
        CKPT="${CKPT_DIR}/consolidated_${METHOD}${SUFFIX}.pt"
        OUT_JSON="results/figures/eval_${METHOD}${SUFFIX}_${TAG}.json"
        if [ -f "${CKPT}" ]; then
            log_msg "Evaluating: ${METHOD} ep=${EP}"
            run_step "eval_${METHOD}_ep${EP}" \
                python scripts/evaluate.py \
                    --model-path "${CKPT}" --all-tasks \
                    --config ${CONFIG} ${DEBUG} ${DEVICE} ${EVAL_EPISODES} \
                    --output "${OUT_JSON}"
        else
            log_msg "WARN: no checkpoint for ${METHOD} ep=${EP}"
        fi
    done
done

# -- Step 3: Aggregate Sweep Results ------------------------------------------

log_msg "---- Step 3: Aggregate Results ----"

# Build a summary JSON with the sweep results using a small Python helper
python - <<'PYEOF'
import json, glob, os, sys, math

tag = "${TAG}"
methods = "${METHODS}".split()
epochs = [int(x) for x in "${EPOCH_VALUES}".split()]
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
        suffix = f"_ep{ep}"
        json_path = os.path.join(fig_dir, f"eval_{method}{suffix}_{tag}.json")
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
            per_task[game] = {"reward": reward, "retention_pct": pct}
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
print("\n" + "=" * 72)
print(f"  Epoch Sweep Summary (tag={tag})")
print("=" * 72)
header = f"{'Method':<14} {'Epochs':>7}  " + "  ".join(f"{g:>12}" for g in sorted(expert_rewards)) + "  Avg%  GMean%"
print(header)
print("-" * len(header))
for method in methods:
    for ep in epochs:
        entry = sweep_results.get(method, {}).get(ep)
        if entry is None:
            continue
        row = f"{method:<14} {ep:>7}  "
        for g in sorted(expert_rewards):
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
