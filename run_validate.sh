#!/bin/sh
#SBATCH --job-name=crl_validate
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p dgx_aic

hostname
date

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.3
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env/
python --version

cd /work/pnag/crl_atari/

echo "=============================================="
echo "CRL-Atari Validation Run"
echo "Testing FireResetEnv fix on Pong + Breakout"
echo "=============================================="

# Step 1: Run diagnostics
echo ""
echo "--- Running diagnostics ---"
python diagnose.py

# Step 2: Short training run (individual only, 200k steps, 2 games)
echo ""
echo "--- Running validation training ---"
python main.py \
  --config config_validate.yaml \
  --seed 42 \
  --methods individual

echo "=============================================="
echo "Validation Complete!"
echo "Check: Pong mean20 should climb above -20 by ~100k steps"
echo "Check: Breakout mean20 should be >1 by ~100k steps"
echo "=============================================="

date
