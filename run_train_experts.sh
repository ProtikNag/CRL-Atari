#!/bin/sh
#SBATCH --job-name=crl_experts
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p dgx_aic

# =============================================================================
# CRL-Atari: Train Expert DQN Agents Only
# =============================================================================
#
# Trains one DQN expert per Atari task. No consolidation, evaluation matrix,
# or report generation — just expert training.
#
# Usage (interactive):
#   ./run_train_experts.sh                  # Full training (3M steps/expert)
#   ./run_train_experts.sh --debug          # Fast debug run (10K steps)
#   ./run_train_experts.sh --tag myexp      # Custom experiment tag
#
# Usage (SLURM):
#   sbatch run_train_experts.sh
#   sbatch run_train_experts.sh --debug --tag quick_test
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
        -h|--help)
            echo "Usage: ./run_train_experts.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug          Enable debug mode (10K steps per expert)"
            echo "  --tag TAG        Experiment tag (default: 'default')"
            echo "  --device DEVICE  Device: cpu, cuda, mps (auto-detected)"
            echo "  --config PATH    Config file path (default: configs/base.yaml)"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

COMMON_ARGS="--config ${CONFIG} --tag ${TAG} ${DEBUG} ${DEVICE}"

# ── Logging ──────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/logs/train_experts_${TAG}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_experts.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_msg "============================================================"
log_msg " CRL-Atari: Train Expert Agents"
log_msg "============================================================"
log_msg " Tag:     ${TAG}"
log_msg " Config:  ${CONFIG}"
log_msg " Debug:   ${DEBUG:-off}"
log_msg " Device:  ${DEVICE:-auto}"
log_msg " Log:     ${LOG_FILE}"
log_msg "============================================================"

# ── Train ────────────────────────────────────────────────────────────────────

log_msg "Starting expert training..."
python scripts/train_experts.py ${COMMON_ARGS} 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]:-$?}

if [ ${EXIT_CODE} -ne 0 ]; then
    log_msg "FAILED: Expert training (exit code ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi

log_msg "============================================================"
log_msg " Expert training complete!"
log_msg " Checkpoints: results/checkpoints/${TAG}/"
log_msg " Log:         ${LOG_FILE}"
log_msg "============================================================"

date
