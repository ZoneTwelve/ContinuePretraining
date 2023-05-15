#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --account="GOV112003"
#SBATCH --partition="gpNCHC_LLM"


MODEL_TYPE=llama
MODEL_PATH=checkpoints/llama-13b
TOKENIZER_PATH=checkpoints/tokenizer/llama-extended
DATASET_PATH=data/cp/llama-extended/b-2048
CKPT_PATH=None

# WandB
NAME="llama-13b-extended_stage1--b-2048"
VERSION=""
TAGS="llama:13b, cp, cp:stage1, tokenizer:extended, data:b"
NOTES="LLaMA 13B 擴充中文字典 Stage1"

# Stage1
EXTEND_TOKENS=True
INITIALIZING_STRATEGY=sample # None, mean, sample
FREEZING_STRATEGY=exclude_new # None, exclude_new, exclude_all

# Common
MICRO_BATCH_SIZE=2
MICRO_BATCH_SIZE_VAL=None
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
