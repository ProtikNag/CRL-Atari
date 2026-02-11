#!/bin/sh
#SBATCH --job-name=crl_atari
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --time=24:00:00

hostname
date

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.3
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env/
python --version

cd /work/pnag/crl_atari/

# =============================================================================
# CRL-Atari Experiment Configuration
# =============================================================================
# Methods: individual, sequential, ewc, htcl
# Quick mode: --quick (50k steps for testing)
# =============================================================================

SEED=42
TRAIN_STEPS=500000
OUTPUT_DIR="./results"

echo "=============================================="
echo "CRL-Atari Experiment"
echo "Seed: ${SEED}"
echo "Train steps per game: ${TRAIN_STEPS}"
echo "=============================================="

# Full experiment (all methods)
python main.py \
  --config config.yaml \
  --seed ${SEED} \
  --train-steps ${TRAIN_STEPS} \
  --output-dir ${OUTPUT_DIR} \
  --methods individual sequential ewc htcl

echo "=============================================="
echo "Experiment Completed!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "=============================================="

date

# =============================================================================
# Quick test run (reduced steps):
# python main.py --config config.yaml --quick
#
# Run single method:
# python main.py --config config.yaml --methods htcl
#
# Custom training budget:
# python main.py --config config.yaml --train-steps 1000000
# =============================================================================
