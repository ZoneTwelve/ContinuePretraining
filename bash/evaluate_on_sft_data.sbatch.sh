#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --account="GOV112003"
#SBATCH --partition="gpNCHC_LLM"

export WANDB_MODE=disabled

MODEL_PATH=checkpoints/llama-7b-cw_stage1-qode_base-e1_stage2-qode_base-s3822
CKPT_PATH=None

TASK="
    python scripts/eval/evaluate.py \
        --model_path=$MODEL_PATH \
        --ckpt_path=$CKPT_PATH \
        --job_type=eval
"

srun bash -c "$TASK"
