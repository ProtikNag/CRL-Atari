#!/bin/sh
#SBATCH --job-name=crl_consolidate
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
########SBATCH -p dgx_aic
#SBATCH -p gpu

# =============================================================================
# CRL-Atari  Consolidation Pipeline
# =============================================================================
#
# Consolidates pre-trained expert checkpoints into single models and produces
# comprehensive evaluation plots and visualisations.
#
# Steps:
#   1. Consolidate experts  (Distillation + HTCL)
#   2. Evaluate all models on all tasks
#   3. Generate comparison plots & Fisher diagnostics
#
# Pre-requisite: expert checkpoints must exist under
#   results/checkpoints/<TAG>/expert_*_best.pt
#   results/checkpoints/<TAG>/expert_summary.json
#
# Usage (local / interactive):
#   ./run_consolidate.sh                        # default tag
#   ./run_consolidate.sh --debug --tag debug    # fast debug run
#   ./run_consolidate.sh --skip-consolidate     # only re-evaluate & re-plot
#
# Usage (SLURM):
#   sbatch run_consolidate.sh --tag full_run_v1
#
# =============================================================================

hostname
date

# ── GPU / Environment Setup ──────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.3 2>/dev/null || true
module load python3/anaconda/2023.9 2>/dev/null || true

# Activate env
if [ -d "/work/pnag/envs/ml_env" ]; then
    source activate /work/pnag/envs/ml_env/
elif [ -d "venv" ]; then
    . venv/bin/activate
fi

python --version
python -c "import torch; print('torch=' + torch.__version__ + ', CUDA=' + str(torch.cuda.is_available()))"

# Change to project directory if on cluster
if [ -d "/work/pnag/CRL-Atari" ]; then
    cd /work/pnag/CRL-Atari/
fi

set -e

# ── Parse arguments ──────────────────────────────────────────────────────────

DEBUG=""
TAG="default"
DEVICE=""
CONFIG="configs/base.yaml"
SKIP_CONSOLIDATE=""
EVAL_EPISODES=""

while [ $# -gt 0 ]; do
    case $1 in
        --debug)           DEBUG="--debug";           shift ;;
        --tag)             TAG="$2";                  shift 2 ;;
        --device)          DEVICE="--device $2";      shift 2 ;;
        --config)          CONFIG="$2";               shift 2 ;;
        --skip-consolidate) SKIP_CONSOLIDATE="true";  shift ;;
        --eval-episodes)   EVAL_EPISODES="--episodes $2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./run_consolidate.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug                Enable debug mode"
            echo "  --tag TAG              Experiment tag (default: 'default')"
            echo "  --device DEVICE        Force device: cpu, cuda, mps"
            echo "  --config PATH          Config file (default: configs/base.yaml)"
            echo "  --skip-consolidate     Skip consolidation, only evaluate & plot"
            echo "  --eval-episodes N      Override evaluation episode count"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

COMMON_ARGS="--config ${CONFIG} --tag ${TAG} ${DEBUG} ${DEVICE}"

# ── Logging ──────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/logs/consolidate_${TAG}_${TIMESTAMP}"
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
log_msg " CRL-Atari  Consolidation Pipeline"
log_msg "============================================================"
log_msg " Tag:    ${TAG}"
log_msg " Config: ${CONFIG}"
log_msg " Debug:  ${DEBUG:-off}"
log_msg " Device: ${DEVICE:-auto}"
log_msg " Log:    ${LOG_DIR}"
log_msg "============================================================"

# ── Step 1: Consolidation ────────────────────────────────────────────────────

if [ -z "${SKIP_CONSOLIDATE}" ]; then
    log_msg "──── Step 1/3: Consolidation ────"

    log_msg "--- Knowledge Distillation ---"
    run_step "01a_distillation" \
        python scripts/consolidate.py --method distillation ${COMMON_ARGS}

    log_msg "--- HTCL ---"
    run_step "01b_htcl" \
        python scripts/consolidate.py --method htcl ${COMMON_ARGS}
else
    log_msg "Skipping consolidation (--skip-consolidate)."
fi

# ── Step 2: Evaluate ─────────────────────────────────────────────────────────

log_msg "──── Step 2/3: Evaluation ────"

CKPT_DIR="results/checkpoints/${TAG}"

for EXPERT_CKPT in ${CKPT_DIR}/expert_*_best.pt; do
    if [ -f "${EXPERT_CKPT}" ]; then
        GAME=$(basename "${EXPERT_CKPT}" | sed 's/expert_//;s/_best.pt//')
        log_msg "Evaluating expert: ${GAME}"
        run_step "02_eval_expert_${GAME}" \
            python scripts/evaluate.py \
                --model-path "${EXPERT_CKPT}" --all-tasks \
                --config ${CONFIG} ${DEBUG} ${DEVICE} ${EVAL_EPISODES} \
                --output "results/figures/eval_expert_${GAME}_${TAG}.json"
    fi
done

for METHOD in distillation htcl; do
    CKPT="${CKPT_DIR}/consolidated_${METHOD}.pt"
    if [ -f "${CKPT}" ]; then
        log_msg "Evaluating consolidated: ${METHOD}"
        run_step "02_eval_${METHOD}" \
            python scripts/evaluate.py \
                --model-path "${CKPT}" --all-tasks \
                --config ${CONFIG} ${DEBUG} ${DEVICE} ${EVAL_EPISODES} \
                --output "results/figures/eval_${METHOD}_${TAG}.json"
    fi
done

# ── Step 3: Comparison Plots ─────────────────────────────────────────────────

log_msg "──── Step 3/3: Comparison Plots ────"
run_step "03_compare" \
    python scripts/compare.py ${COMMON_ARGS} ${EVAL_EPISODES}

# ── Summary ──────────────────────────────────────────────────────────────────

log_msg "============================================================"
log_msg " Pipeline Complete!"
log_msg "============================================================"
log_msg " Figures:     results/figures/{png,svg}/"
log_msg " Checkpoints: results/checkpoints/${TAG}/"
log_msg " Logs:        ${LOG_DIR}/"
log_msg "============================================================"

date
