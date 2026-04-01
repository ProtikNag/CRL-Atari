#!/bin/sh
#SBATCH --job-name=crl_classic
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p dgx_aic
#SBATCH --time=06:00:00

# =============================================================================
# CRL Classic Control: Full Multi-Seed Experiment
# =============================================================================
#
# Trains DQN experts, runs all consolidation and sequential methods (WHC,
# Distillation, Hybrid, EWC, P&C, TRAC, C-CHAIN, Multi-Task), evaluates
# with 500 episodes, and produces retention heatmap with confidence intervals
# across 5 seeds.
#
# Estimated time:
#   Single seed:  ~30-40 minutes on GPU
#   All 5 seeds:  ~3-4 hours on GPU
#   Debug mode:   ~5 minutes
#
# Usage (interactive):
#   ./run_classic_control.sh                      # Full 5-seed run
#   ./run_classic_control.sh --single-seed 42     # One seed only
#   ./run_classic_control.sh --debug              # Fast debug run
#
# Usage (SLURM):
#   sbatch run_classic_control.sh
#   sbatch run_classic_control.sh --debug
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
CONFIG="configs/classic_control.yaml"
SINGLE_SEED=""

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
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --single-seed)
            SINGLE_SEED="--single-seed $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run_classic_control.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Enable debug mode (~5 min)"
            echo "  --tag TAG            Experiment tag (default: 'default')"
            echo "  --config PATH        Config file (default: configs/classic_control.yaml)"
            echo "  --single-seed SEED   Run only one seed (skip multi-seed)"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  sbatch run_classic_control.sh                    # Full 5-seed run"
            echo "  sbatch run_classic_control.sh --single-seed 42   # Quick single seed"
            echo "  sbatch run_classic_control.sh --debug            # Fast debug"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PYTHON_ARGS="--config ${CONFIG} --tag ${TAG} ${DEBUG} ${SINGLE_SEED}"

# ── Logging ──────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results_classic_control/logs/run_${TAG}_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/experiment.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_msg "============================================================"
log_msg " CRL Classic Control Experiment"
log_msg "============================================================"
log_msg " Tag:         ${TAG}"
log_msg " Config:      ${CONFIG}"
log_msg " Debug:       ${DEBUG:-off}"
log_msg " Single seed: ${SINGLE_SEED:-all 5 seeds}"
log_msg " Log:         ${LOG_FILE}"
log_msg " Host:        $(hostname)"
log_msg " GPU:         ${CUDA_VISIBLE_DEVICES}"
log_msg "============================================================"

# ── Run experiment ───────────────────────────────────────────────────────────

log_msg "Starting experiment..."
python scripts/run_classic_control.py ${PYTHON_ARGS} 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]:-$?}

if [ ${EXIT_CODE} -ne 0 ]; then
    log_msg "FAILED: Experiment exited with code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

log_msg "============================================================"
log_msg " Experiment complete!"
log_msg " Results:     results_classic_control/figures/"
log_msg " Checkpoints: results_classic_control/checkpoints/"
log_msg " Log:         ${LOG_FILE}"
log_msg "============================================================"

date
