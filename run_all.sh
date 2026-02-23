#!/bin/sh
#SBATCH --job-name=crl_atari
#SBATCH -N 1                    ## Compute Node (Number of computers)
#SBATCH -n 24                   ## CPU Cores
#SBATCH --gres=gpu:1            ## Run on 1 GPU
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
########SBATCH -p dgx_aic
#SBATCH -p gpu

# =============================================================================
# CRL-Atari: Full Experiment Pipeline (Hyperion / SLURM)
# =============================================================================
#
# Run the complete continual RL experiment:
#   1. Train expert DQN agents on each Atari task sequentially
#   2. Consolidate experts using EWC, Distillation, and HTCL
#   3. Evaluate all models on all tasks
#   4. Generate comparison plots and visualizations
#   5. Generate HTML technical report
#
# Usage (interactive):
#   ./run_all.sh              # Full experiment
#   ./run_all.sh --debug      # Fast debug run (reduced training)
#   ./run_all.sh --tag myexp  # Custom experiment tag
#
# Usage (SLURM):
#   sbatch run_all.sh
#   sbatch run_all.sh --debug --tag quick_test
#
# =============================================================================

hostname
date

# ── GPU / Environment Setup ──────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.3 2>/dev/null || true
module load python3/anaconda/2023.9 2>/dev/null || true

# Activate the conda/venv environment
# Adjust the path below to match Hyperion setup
if [ -d "/work/pnag/envs/ml_env" ]; then
    source activate /work/pnag/envs/ml_env/
elif [ -d "venv" ]; then
    . venv/bin/activate
fi

python --version
echo "PyTorch version:"
python -c "import torch; print('  torch=' + torch.__version__ + ', CUDA=' + str(torch.cuda.is_available()) + ', device_count=' + str(torch.cuda.device_count()))"
echo "GPU info:"
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print('  GPU %d: %s (%.1f GB)' % (i, torch.cuda.get_device_name(i), props.total_mem / 1e9))
else:
    print('  No CUDA GPUs detected')
"

# Change to project directory (adjust for Hyperion)
if [ -d "/work/pnag/CRL-Atari" ]; then
    cd /work/pnag/CRL-Atari/
fi

set -e  # Exit on error

# ── Parse arguments ──────────────────────────────────────────────────────────

DEBUG=""
TAG="default"
DEVICE=""
CONFIG="configs/base.yaml"
SKIP_TRAIN=""
SKIP_CONSOLIDATE=""
EVAL_EPISODES=""

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
        --skip-train)
            SKIP_TRAIN="true"
            shift
            ;;
        --skip-consolidate)
            SKIP_CONSOLIDATE="true"
            shift
            ;;
        --eval-episodes)
            EVAL_EPISODES="--episodes $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Enable debug mode (fast training)"
            echo "  --tag TAG            Experiment tag (default: 'default')"
            echo "  --device DEVICE      Device: cpu, cuda, mps (auto-detected)"
            echo "  --config PATH        Config file path (default: configs/base.yaml)"
            echo "  --skip-train         Skip expert training (use existing checkpoints)"
            echo "  --skip-consolidate   Skip consolidation (only evaluate)"
            echo "  --eval-episodes N    Number of evaluation episodes"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

COMMON_ARGS="--config ${CONFIG} --tag ${TAG} ${DEBUG} ${DEVICE}"

# ── Set up logging ───────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/logs/pipeline_${TAG}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# Master log captures everything
MASTER_LOG="${LOG_DIR}/pipeline_master.log"

# Helper: log with timestamp to both stdout and master log
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

# Helper: run a pipeline step, tee output to both a step-specific log and master
run_step() {
    STEP_NAME="$1"
    shift
    STEP_LOG="${LOG_DIR}/${STEP_NAME}.log"
    log_msg "START: ${STEP_NAME}"
    log_msg "CMD: $*"
    log_msg "Step log: ${STEP_LOG}"

    # Run command, capture stdout+stderr, tee to step log and master log
    "$@" 2>&1 | tee -a "${STEP_LOG}" "${MASTER_LOG}"
    EXIT_CODE=${PIPESTATUS[0]:-$?}

    if [ ${EXIT_CODE} -ne 0 ]; then
        log_msg "FAILED: ${STEP_NAME} (exit code ${EXIT_CODE})"
        log_msg "Check ${STEP_LOG} for details."
        exit ${EXIT_CODE}
    else
        log_msg "DONE: ${STEP_NAME}"
    fi
    echo "" >> "${MASTER_LOG}"
}

# ── Log environment info ─────────────────────────────────────────────────────

log_msg "============================================================"
log_msg " CRL-Atari Experiment Pipeline"
log_msg "============================================================"
log_msg " Hostname:  $(hostname)"
log_msg " Date:      $(date)"
log_msg " Tag:       ${TAG}"
log_msg " Config:    ${CONFIG}"
log_msg " Debug:     ${DEBUG:-off}"
log_msg " Device:    ${DEVICE:-auto}"
log_msg " Log dir:   ${LOG_DIR}"
log_msg " Master log: ${MASTER_LOG}"
log_msg "============================================================"

# Log full environment snapshot for reproducibility
{
    echo "=== ENVIRONMENT SNAPSHOT ==="
    echo "USER: $(whoami)"
    echo "HOSTNAME: $(hostname)"
    echo "DATE: $(date)"
    echo "PWD: $(pwd)"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
    echo ""
    echo "=== PYTHON PACKAGES ==="
    pip list 2>/dev/null || pip3 list 2>/dev/null || echo "pip not found"
    echo ""
    echo "=== GIT STATUS ==="
    git log --oneline -5 2>/dev/null || echo "Not a git repo"
    git diff --stat 2>/dev/null || true
    echo ""
    echo "=== NVIDIA-SMI ==="
    nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
    echo ""
    echo "=== CONFIG FILE ==="
    cat "${CONFIG}"
} > "${LOG_DIR}/environment_snapshot.log" 2>&1

log_msg "Environment snapshot saved to ${LOG_DIR}/environment_snapshot.log"

# ── Step 1: Train Experts ────────────────────────────────────────────────────

if [ -z "${SKIP_TRAIN}" ]; then
    log_msg "============================================================"
    log_msg " Step 1/4: Training Expert Agents"
    log_msg "============================================================"
    run_step "01_train_experts" python scripts/train_experts.py ${COMMON_ARGS}
    log_msg ""
else
    log_msg "Skipping expert training (--skip-train)."
fi

# ── Step 2: Consolidate ─────────────────────────────────────────────────────

if [ -z "${SKIP_CONSOLIDATE}" ]; then
    log_msg "============================================================"
    log_msg " Step 2/4: Consolidation"
    log_msg "============================================================"

    log_msg "--- EWC ---"
    run_step "02a_consolidate_ewc" python scripts/consolidate.py --method ewc ${COMMON_ARGS}

    log_msg "--- Knowledge Distillation ---"
    run_step "02b_consolidate_distillation" python scripts/consolidate.py --method distillation ${COMMON_ARGS}

    log_msg "--- HTCL ---"
    run_step "02c_consolidate_htcl" python scripts/consolidate.py --method htcl ${COMMON_ARGS}

    log_msg "All consolidation methods complete."
else
    log_msg "Skipping consolidation (--skip-consolidate)."
fi

# ── Step 3: Evaluate All Models ──────────────────────────────────────────────

log_msg "============================================================"
log_msg " Step 3/4: Evaluating All Models"
log_msg "============================================================"

CKPT_DIR="results/checkpoints/${TAG}"

# Evaluate each expert on all tasks
for EXPERT_CKPT in ${CKPT_DIR}/expert_*_best.pt; do
    if [ -f "${EXPERT_CKPT}" ]; then
        GAME=$(basename "${EXPERT_CKPT}" | sed 's/expert_//;s/_best.pt//')
        log_msg "Evaluating expert: ${GAME}"
        run_step "03_eval_expert_${GAME}" python scripts/evaluate.py \
            --model-path "${EXPERT_CKPT}" \
            --all-tasks \
            --config ${CONFIG} \
            ${DEBUG} ${DEVICE} ${EVAL_EPISODES} \
            --output "results/figures/eval_expert_${GAME}_${TAG}.json"
    fi
done

# Evaluate consolidated models on all tasks
for METHOD in ewc distillation htcl; do
    CKPT="${CKPT_DIR}/consolidated_${METHOD}.pt"
    if [ -f "${CKPT}" ]; then
        log_msg "Evaluating consolidated: ${METHOD}"
        run_step "03_eval_${METHOD}" python scripts/evaluate.py \
            --model-path "${CKPT}" \
            --all-tasks \
            --config ${CONFIG} \
            ${DEBUG} ${DEVICE} ${EVAL_EPISODES} \
            --output "results/figures/eval_${METHOD}_${TAG}.json"
    fi
done

# ── Step 4: Compare and Plot ────────────────────────────────────────────────

log_msg "============================================================"
log_msg " Step 4/5: Generating Comparison Plots & Visualizations"
log_msg "============================================================"
run_step "04a_compare" python scripts/compare.py ${COMMON_ARGS} ${EVAL_EPISODES}
run_step "04b_visualize" python scripts/visualize.py --tag ${TAG}

# ── Step 5: Generate HTML Report ─────────────────────────────────────────────

log_msg "============================================================"
log_msg " Step 5/5: Generating HTML Technical Report"
log_msg "============================================================"
run_step "05_report" python scripts/generate_report.py --output docs/CRL_Atari_Technical_Report.html

# ── Summary ──────────────────────────────────────────────────────────────────

log_msg "============================================================"
log_msg " Pipeline Complete!"
log_msg "============================================================"
log_msg " Results:      results/figures/"
log_msg " Report:       docs/CRL_Atari_Technical_Report.html"
log_msg " Checkpoints:  results/checkpoints/${TAG}/"
log_msg " Logs:         ${LOG_DIR}/"
log_msg " Master log:   ${MASTER_LOG}"
log_msg "============================================================"

date

# =============================================================================
# Example configurations:
# =============================================================================
#
# Full experiment on Hyperion:
#   sbatch run_all.sh --tag full_run_v1
#
# Quick debug test:
#   sbatch run_all.sh --debug --tag debug_test
#
# Skip training, only re-consolidate and compare:
#   ./run_all.sh --skip-train --tag full_run_v1
#
# Force CPU (debug on login node):
#   ./run_all.sh --debug --device cpu --tag cpu_test
#
# =============================================================================
