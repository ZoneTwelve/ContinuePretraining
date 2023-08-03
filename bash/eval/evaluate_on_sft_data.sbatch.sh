#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --account="GOV112003"
#SBATCH --partition="gpNCHC_LLM"

export WANDB_MODE=disabled

MODEL_PATH=
DATA_PATH="data/sft/raw"
NAME=""
VERSION=""
SAVE_DIR="logs/eval"

TASK="
    python scripts/eval/evaluate.py \
        --model_path=$MODEL_PATH \
        --data_path=$DATA_PATH \
        --name=$NAME \
        --version=$VERSION \
        --save_dir=$SAVE_DIR \
"

srun bash -c "$TASK"
