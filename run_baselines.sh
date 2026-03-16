#!/bin/sh
#SBATCH --job-name=crl_baselines
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p gpu

# =============================================================================
# CRL-Atari: Train Baseline Methods
# =============================================================================
#
# Runs baseline continual learning methods for comparison against CRL
# consolidation approaches. Currently supported:
#
#   EWC        — Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017)
#   Online EWC — Running Fisher average variant (Schwarz et al., 2018)
#   Multi-Task — Joint training on all tasks simultaneously (upper bound)
#
# EWC protocol (sequential continual learning):
#   Breakout (load expert) → SpaceInvaders (train w/ penalty) → Pong (train)
#   Task 1 loaded from existing expert checkpoint (~5.5h saved).
#
# Multi-task protocol:
#   Round-robin across all tasks, 5M total steps (~1.67M per task).
#   No forgetting by construction — serves as the performance ceiling.
#
# Usage (interactive):
#   ./run_baselines.sh                          # Run all baselines
#   ./run_baselines.sh --debug                  # Fast debug run
#   ./run_baselines.sh --tag myexp              # Custom experiment tag
#   ./run_baselines.sh --online-ewc             # Online EWC variant
#   ./run_baselines.sh --skip-ewc               # Skip EWC, run multi-task only
#   ./run_baselines.sh --skip-multitask         # Skip multi-task, run EWC only
#   ./run_baselines.sh --mtl-steps 3000000      # Override multi-task step count
#   ./run_baselines.sh --skip-eval              # Train only, skip eval steps
#
# Usage (SLURM):
#   sbatch run_baselines.sh
#   sbatch run_baselines.sh --debug --tag quick_test
#
# =============================================================================

hostname
date

# ── GPU / Environment Setup ──────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.3 2>/dev/null || true
module load python3/anaconda/2023.9 2>/dev/null || true

if [ -d "/work/pnag/envs/ml_env" ]; then
    source activate /work/pnag/envs/ml_env/
elif [ -d "venv" ]; then
    . venv/bin/activate
fi

python --version
echo "PyTorch version:"
python -c "import torch; print('  torch=' + torch.__version__ + ', CUDA=' + str(torch.cuda.is_available()) + ', device_count=' + str(torch.cuda.device_count()))"

if [ -d "/work/pnag/CRL-Atari" ]; then
    cd /work/pnag/CRL-Atari/
fi

set -e

# ── Parse arguments ──────────────────────────────────────────────────────────

DEBUG=""
TAG="default"
DEVICE=""
CONFIG="configs/base.yaml"
ONLINE_EWC=""
EWC_LAMBDA=""
GAMMA_EWC=""
SKIP_EVAL=""
SKIP_EWC=""
SKIP_MULTITASK=""
MTL_STEPS=""
EXPERT_PATH=""

while [ $# -gt 0 ]; do
    case $1 in
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --device)
            DEVICE="--device $2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --online-ewc)
            ONLINE_EWC="--online-ewc"
            shift
            ;;
        --ewc-lambda)
            EWC_LAMBDA="--ewc-lambda $2"
            shift 2
            ;;
        --gamma-ewc)
            GAMMA_EWC="--gamma-ewc $2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL="1"
            shift
            ;;
        --skip-ewc)
            SKIP_EWC="1"
            shift
            ;;
        --skip-multitask)
            SKIP_MULTITASK="1"
            shift
            ;;
        --mtl-steps)
            MTL_STEPS="--total-steps $2"
            shift 2
            ;;
        --expert)
            EXPERT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run_baselines.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug               Enable debug mode (10K steps per task)"
            echo "  --tag TAG             Experiment tag (default: 'default')"
            echo "  --device DEVICE       Device: cpu, cuda, mps (auto-detected)"
            echo "  --config PATH         Config file (default: configs/base.yaml)"
            echo "  --online-ewc          Use Online EWC (running Fisher average)"
            echo "  --ewc-lambda FLOAT    Override EWC penalty strength"
            echo "  --gamma-ewc FLOAT     Online EWC decay factor (default: 0.95)"
            echo "  --skip-eval           Skip evaluation steps after training"
            echo "  --skip-ewc            Skip EWC training (run multi-task only)"
            echo "  --skip-multitask      Skip multi-task training (run EWC only)"
            echo "  --mtl-steps INT       Override multi-task total steps (default: 5M)"
            echo "  --expert PATH         Path to Task 1 expert checkpoint (for EWC)"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

COMMON_ARGS="--config ${CONFIG} --tag ${TAG} ${DEBUG} ${DEVICE}"

# ── Determine baseline variant name ─────────────────────────────────────────

if [ -n "${ONLINE_EWC}" ]; then
    VARIANT="online_ewc"
else
    VARIANT="ewc"
fi

# ── Logging ──────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/logs/baseline_${VARIANT}_${TAG}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/baselines.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_msg "============================================================"
log_msg " CRL-Atari: Baseline Training"
log_msg "============================================================"
log_msg " Variant:  ${VARIANT}"
log_msg " Tag:      ${TAG}"
log_msg " Config:   ${CONFIG}"
log_msg " Debug:    ${DEBUG:-off}"
log_msg " Device:   ${DEVICE:-auto}"
log_msg " Lambda:   ${EWC_LAMBDA:-from config}"
log_msg " Log:      ${LOG_FILE}"
log_msg "============================================================"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1–2: EWC Sequential Training + Evaluation
# ══════════════════════════════════════════════════════════════════════════════

if [ -z "${SKIP_EWC}" ]; then

    # ── Auto-detect Task 1 expert checkpoint ──────────────────────────────
    if [ -z "${EXPERT_PATH}" ]; then
        EXPERT_PATH="results/checkpoints/${TAG}/expert_Breakout_best.pt"
    fi

    if [ ! -f "${EXPERT_PATH}" ]; then
        log_msg "ERROR: Expert checkpoint not found: ${EXPERT_PATH}"
        log_msg "Train experts first:  ./run_train_experts.sh --tag ${TAG}"
        exit 1
    fi

    log_msg "Task 1 expert: ${EXPERT_PATH}"

    log_msg ""
    log_msg "────────────────────────────────────────────────────────────"
    log_msg " Step 1: EWC Sequential Training"
    log_msg "────────────────────────────────────────────────────────────"

    EWC_ARGS="${COMMON_ARGS} --first-task-expert ${EXPERT_PATH} ${ONLINE_EWC} ${EWC_LAMBDA} ${GAMMA_EWC}"

    log_msg "Running: python scripts/train_ewc.py ${EWC_ARGS}"
    python scripts/train_ewc.py ${EWC_ARGS} 2>&1 | tee -a "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]:-$?}

    if [ ${EXIT_CODE} -ne 0 ]; then
        log_msg "FAILED: EWC training (exit code ${EXIT_CODE})"
        exit ${EXIT_CODE}
    fi

    log_msg "EWC training complete."

    # ── Evaluate EWC model ────────────────────────────────────────────────
    if [ -z "${SKIP_EVAL}" ]; then
        log_msg ""
        log_msg "────────────────────────────────────────────────────────────"
        log_msg " Step 2: Evaluate EWC Model on All Tasks"
        log_msg "────────────────────────────────────────────────────────────"

        MODEL_PATH="results/checkpoints/${TAG}/consolidated_ewc.pt"
        EVAL_OUTPUT="results/figures/eval_ewc_${TAG}.json"

        if [ ! -f "${MODEL_PATH}" ]; then
            log_msg "ERROR: Model not found: ${MODEL_PATH}"
            exit 1
        fi

        log_msg "Model:  ${MODEL_PATH}"
        log_msg "Output: ${EVAL_OUTPUT}"

        python scripts/evaluate.py \
            --model-path "${MODEL_PATH}" \
            --all-tasks \
            --config ${CONFIG} ${DEBUG} ${DEVICE} \
            --output "${EVAL_OUTPUT}" \
            2>&1 | tee -a "${LOG_FILE}"

        EXIT_CODE=${PIPESTATUS[0]:-$?}

        if [ ${EXIT_CODE} -ne 0 ]; then
            log_msg "WARNING: EWC evaluation failed (exit code ${EXIT_CODE})"
        else
            log_msg "Evaluation results saved to ${EVAL_OUTPUT}"
        fi
    fi

else
    log_msg "Skipping EWC (--skip-ewc)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3–4: Multi-Task Joint Training + Evaluation
# ══════════════════════════════════════════════════════════════════════════════

if [ -z "${SKIP_MULTITASK}" ]; then

    log_msg ""
    log_msg "────────────────────────────────────────────────────────────"
    log_msg " Step 3: Multi-Task Joint Training"
    log_msg "────────────────────────────────────────────────────────────"

    MTL_ARGS="${COMMON_ARGS} ${MTL_STEPS}"

    log_msg "Running: python scripts/train_multitask.py ${MTL_ARGS}"
    python scripts/train_multitask.py ${MTL_ARGS} 2>&1 | tee -a "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]:-$?}

    if [ ${EXIT_CODE} -ne 0 ]; then
        log_msg "FAILED: Multi-task training (exit code ${EXIT_CODE})"
        exit ${EXIT_CODE}
    fi

    log_msg "Multi-task training complete."

    # ── Evaluate multi-task model ─────────────────────────────────────────
    if [ -z "${SKIP_EVAL}" ]; then
        log_msg ""
        log_msg "────────────────────────────────────────────────────────────"
        log_msg " Step 4: Evaluate Multi-Task Model on All Tasks"
        log_msg "────────────────────────────────────────────────────────────"

        MODEL_PATH="results/checkpoints/${TAG}/consolidated_multitask.pt"
        EVAL_OUTPUT="results/figures/eval_multitask_${TAG}.json"

        if [ ! -f "${MODEL_PATH}" ]; then
            log_msg "ERROR: Model not found: ${MODEL_PATH}"
            exit 1
        fi

        log_msg "Model:  ${MODEL_PATH}"
        log_msg "Output: ${EVAL_OUTPUT}"

        python scripts/evaluate.py \
            --model-path "${MODEL_PATH}" \
            --all-tasks \
            --config ${CONFIG} ${DEBUG} ${DEVICE} \
            --output "${EVAL_OUTPUT}" \
            2>&1 | tee -a "${LOG_FILE}"

        EXIT_CODE=${PIPESTATUS[0]:-$?}

        if [ ${EXIT_CODE} -ne 0 ]; then
            log_msg "WARNING: Multi-task evaluation failed (exit code ${EXIT_CODE})"
        else
            log_msg "Evaluation results saved to ${EVAL_OUTPUT}"
        fi
    fi

else
    log_msg "Skipping multi-task (--skip-multitask)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# [FUTURE] Additional baselines
# ══════════════════════════════════════════════════════════════════════════════
#
# To add a new baseline (e.g., L2 Regularization, Progressive Nets):
#   1. Create src/baselines/<method>.py
#   2. Create scripts/train_<method>.py
#   3. Add a new step section following the same pattern
#

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

log_msg ""
log_msg "============================================================"
log_msg " Baselines Complete!"
log_msg "============================================================"

if [ -z "${SKIP_EWC}" ]; then
    log_msg " EWC checkpoint:       results/checkpoints/${TAG}/consolidated_ewc.pt"
    if [ -z "${SKIP_EVAL}" ]; then
        log_msg " EWC eval JSON:        results/figures/eval_ewc_${TAG}.json"
    fi
fi

if [ -z "${SKIP_MULTITASK}" ]; then
    log_msg " Multi-task checkpoint: results/checkpoints/${TAG}/consolidated_multitask.pt"
    if [ -z "${SKIP_EVAL}" ]; then
        log_msg " Multi-task eval JSON:  results/figures/eval_multitask_${TAG}.json"
    fi
fi

log_msg " Full log:             ${LOG_FILE}"
log_msg "============================================================"
log_msg ""
log_msg "Next steps:"
log_msg "  Visualize:   python scripts/visualize.py --tag ${TAG}"
log_msg "  Report:      python scripts/generate_report.py --tag ${TAG}"
log_msg ""

date
