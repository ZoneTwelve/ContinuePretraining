#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --account="GOV112003"
#SBATCH --partition="gpNCHC_LLM"
#SBATCH --output="logs/slurm/%j-train_stage1.log" \

# export WANDB_MODE=disabled

MODEL_TYPE=llama
MODEL_PATH=checkpoints/open_llama_7b
TOKENIZER_PATH=checkpoints/tokenizer/open_llama-cw
MAX_LENGTH=2048
DATA_PATH=data/cp/group/g
DATASET_PATH=data/cp/tokenized/open_llama-cw/g
CKPT_PATH=None

# WandB
NAME="open_llama_7b-cw_stage1-g"
VERSION=""
TAGS="open_llama:7b, cp, cp:stage1, tokenizer:open_llama-cw, data:g"
NOTES=""

# Stage1
EXTEND_TOKENS=True
INITIALIZING_STRATEGY=mean # None, mean, sample
FREEZING_STRATEGY=exclude_new # None, exclude_new, exclude_all

# Common
MICRO_BATCH_SIZE=4
MICRO_BATCH_SIZE_VAL=8
ACCUMULATE_GRAD_BATCHES=1
LR=1e-4
LR_SCHEDULER_TYPE=None # None, linear, cosine
NUM_WARMUP_STEPS=500
MIN_LR_FACTOR=0.1
MAX_EPOCHS=1
MAX_STEPS=-1
CKPT_PATH=None
VAL_CHECK_INTERVAL=500
SEED=42


TASK="
    python scripts/cp/train.py \
        --model_type=$MODEL_TYPE \
        --model_path=$MODEL_PATH \
        --tokenizer_path=$TOKENIZER_PATH \
        --max_length=$MAX_LENGTH \
        --data_path=$DATA_PATH \
        --dataset_path=$DATASET_PATH \
        --name=\"$NAME\" \
        --version=$VERSION \
        --tags=\"$TAGS\" \
        --notes=\"$NOTES\" \
        --extend_tokens=$EXTEND_TOKENS \
        --initializing_strategy=$INITIALIZING_STRATEGY \
        --freezing_strategy=$FREEZING_STRATEGY \
        --lr=$LR \
        --lr_scheduler_type=$LR_SCHEDULER_TYPE \
        --num_warmup_steps=$NUM_WARMUP_STEPS \
        --min_lr_factor=$MIN_LR_FACTOR \
        --micro_batch_size=$MICRO_BATCH_SIZE \
        --micro_batch_size_val=$MICRO_BATCH_SIZE_VAL \
        --accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        --num_workers=4 \
        --max_epochs=$MAX_EPOCHS \
        --val_check_interval=$VAL_CHECK_INTERVAL \
        --ckpt_path=$CKPT_PATH \
        --seed=$SEED \
"

srun bash -c "$TASK"
