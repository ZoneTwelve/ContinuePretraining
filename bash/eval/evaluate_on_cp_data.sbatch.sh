#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --account="GOV112003"
#SBATCH --partition="gpNCHC_LLM"
#SBATCH --output="logs/slurm/%j.log" \


MODEL_PATH="checkpoints/open_llama_7b"
DATA_PATH="data/cp/evaluation/perplexity/tw"
MAX_LENGTH=2048
BATCH_SIZE=1
NUM_DATAPOINTS=None
NAME="tstl"
VERSION="openllama-$MAX_LENGTH"


TASK="
    python scripts/eval/evaluate_on_cp_data.py \
        --name="$NAME" \
        --version="$VERSION" \
        --model_path="$MODEL_PATH" \
        --data_path="$DATA_PATH" \
        --max_length=$MAX_LENGTH \
        --batch_size=$BATCH_SIZE \
"

srun bash -c "$TASK"
