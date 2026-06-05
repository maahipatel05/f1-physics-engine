#!/bin/bash
#SBATCH --job-name=f1_lora_svd
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1          # Request 1× A100 (80 GB)
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================
# Rice University SLURM training script
# Usage:  sbatch slurm/train.sh
# ============================================================

set -euo pipefail

# ---- Environment ----------------------------------------------------
module purge
module load GCC/11.3.0 CUDA/12.1.0 cuDNN/8.9.2

# Activate your conda / venv environment
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate f1_physics

# ---- Paths ----------------------------------------------------------
PROJECT_DIR="${HOME}/f1_physics_engine"
cd "${PROJECT_DIR}"

mkdir -p logs checkpoints

# ---- HuggingFace cache (shared scratch to avoid redundant downloads) -
export HF_HOME="/scratch/${USER}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}"

# ---- WandB (optional – set your key in .env or here) ----------------
# export WANDB_API_KEY="your_key_here"

# ---- Memory / CUDA flags --------------------------------------------
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false

# ---- Accelerate config (single GPU, fp16) ---------------------------
cat > /tmp/accelerate_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# ---- Launch training ------------------------------------------------
echo "========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started:   $(date)"
echo "========================================="

accelerate launch \
    --config_file /tmp/accelerate_config.yaml \
    train.py \
    --config configs/default.yaml

echo "Training finished: $(date)"