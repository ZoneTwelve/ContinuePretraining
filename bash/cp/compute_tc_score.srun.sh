NODES=1
GPUS=8
TIME=
ACCOUNT="GOV112003"
PARTITOIN="gpNCHC_LLM"
NODELIST=
JOB_NAME=


MODEL_PATH=checkpoints/opt-1.3b-extended_stage1--d-2048--e1_stage2--d-2048--s10343
DATA_PATH=data/cp/evaluation/perplexity/tw/zhtw_news.jsonl
BATCH_SIZE=8
NUM_SAMPLES=10000

TASK="
    python scripts/cp/compute_tc_score.py \
        --model_path=$MODEL_PATH \
        --data_path=$DATA_PATH \
        --batch_size=$BATCH_SIZE \
        --num_samples=$NUM_SAMPLES
"

if [ $TIME ]
then
    TIME="--time=$TIME"
fi

if [ ! $JOB_NAME ]
then
    JOB_NAME=$(basename "$0")
fi

srun \
    --job-name=$JOB_NAME \
    --nodes=$NODES \
    --gpus-per-node=$GPUS \
    --ntasks-per-node=$GPUS \
    --cpus-per-task=4 \
    $TIME \
    --account=$ACCOUNT \
    --partition=$PARTITOIN \
    --nodelist=$NODELIST \
    bash -c "$TASK"
