NODES=1
GPUS=8
ACCOUNT="GOV112003"
PARTITOIN="gpNCHC_LLM"


MODEL_PATHS=(
    
)
DATA_PATH=data/cp/evaluation/perplexity/tw/zhtw_news.jsonl
BATCH_SIZE=8
NUM_SAMPLES=10000

for MODEL_PATH in "${MODEL_PATHS[@]}"
do
    MODEL_PATH=$MODEL_PATH

    MODEL_NAME=$(echo "$MODEL_PATH" | sed 's/checkpoints\///')
    JOB_NAME="compute_tc_score:$MODEL_NAME"
    echo $MODEL_NAME

    TASK="
        srun python scripts/cp/compute_tc_score.py \
            --model_path=$MODEL_PATH \
            --data_path=$DATA_PATH \
            --batch_size=$BATCH_SIZE \
            --num_samples=$NUM_SAMPLES \
    "

    f() {
        sbatch \
            --job-name=$JOB_NAME \
            --nodes=$NODES \
            --gpus-per-node=$GPUS \
            --ntasks-per-node=$GPUS \
            --cpus-per-task=4 \
            --account=$ACCOUNT \
            --partition=$PARTITOIN \
            --output="logs/slurm/compute_tc_score__$MODEL_NAME.log" \
            --wrap="$TASK"
    }

    until f; [[ $? -eq 0 ]];
    do
        echo "Failed to submit, retrying."
        sleep 3
    done
        
done
