JOB_NAME=train
NODES=4

# export WANDB_MODE=disabled

CONFIG="config/llama2-7b|tokenizer=ccw|data=j-v2+cc/spn.yaml"
VERSION=null
CKPT_PATH=null
LOAD_FULL_WEIGHTS=false


COMMAND="
    python scripts/main.py fit
        --config \"$CONFIG\"
        --ckpt_path \"$CKPT_PATH\"
        --trainer.logger.init_args.version \"$VERSION\"
        --trainer.strategy.init_args.load_full_weights \"$LOAD_FULL_WEIGHTS\"
"

sbatchx \
    --job_name $JOB_NAME \
    --nodes $NODES \
    "$COMMAND"
