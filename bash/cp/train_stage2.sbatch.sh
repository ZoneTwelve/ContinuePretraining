#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --account="GOV112003"
#SBATCH --partition="gpNCHC_LLM"


MODEL_TYPE=llama
MODEL_PATH=
TOKENIZER_PATH=None
DATASET_PATH=
CKPT_PATH=None

# WandB
NAME=""
VERSION=""
TAGS=""
NOTES=""

# Common
MIRCO_BATCH_SIZE=8
MIRCO_BATCH_SIZE_VAL=4
ACCUMULATE_GRAD_BATCHES=1
LR=1e-4
LR_SCHEDULER_TYPE=None # None, linear, cosine
NUM_WARMUP_STEPS=500
MAX_EPOCHS=1
MAX_STEPS=-1
CKPT_PATH=None
VAL_CHECK_INTERVAL=500
SAVE_EVERY_N_STEPS=500
SEED=42

# Stage1
EXTEND_TOKENS=False
INITIALIZING_STRATEGY=None # None, mean, sample
FREEZING_STRATEGY=None # None, exclude_new, exclude_all


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
        --lr=$LR \
        --lr_scheduler_type=$LR_SCHEDULER_TYPE \
        --num_warmup_steps=$NUM_WARMUP_STEPS \
        --min_lr_factor=$MIN_LR_FACTOR \
        --micro_batch_size=$MIRCO_BATCH_SIZE \
        --micro_batch_size_val=$MIRCO_BATCH_SIZE_VAL \
        --accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES \
        --num_workers=4 \
        --max_epochs=$MAX_EPOCHS \
        --val_check_interval=$VAL_CHECK_INTERVAL \
        --ckpt_path=$CKPT_PATH \
"

srun bash -c "$TASK"
