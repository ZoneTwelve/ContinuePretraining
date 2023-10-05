JOB_NAME=train
NODES=12

# export WANDB_MODE=disabled

CONFIG="config/llama2-13b|tokenizer=ccw|data=k>cc.yaml"
COMMAND="
    python scripts/cp/main.py fit
        --config \"$CONFIG\"
"

sbatchx \
    --job_name $JOB_NAME \
    --nodes $NODES \
    "$COMMAND"
