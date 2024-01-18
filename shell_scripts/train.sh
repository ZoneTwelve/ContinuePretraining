JOB_NAME="cp-mistral-7b"
PARTITION="gpNCHC_H100"
# PARTITION=gp1d
NODES=2

# export WANDB_MODE=disabled
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG="INFO"
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"

SUBCOMMAND="fit"
CONFIG="config/mistral-7b|tokenizer=ccw|data=m.yaml"
VERSION="null"
CKPT_PATH="null"


COMMAND="
    python scripts/main.py $SUBCOMMAND
        --config \"$CONFIG\"
        --ckpt_path \"$CKPT_PATH\"
        --trainer.logger.init_args.version \"$VERSION\"
        --trainer.num_nodes $NODES
"

SBATCH_ARGS=""
add_sbatch_arg() {
    if [ ! -z "$2" ]; then
        SBATCH_ARGS="$SBATCH_ARGS $1 $2"
    fi
}
add_sbatch_arg "--job-name" $JOB_NAME
add_sbatch_arg "--partition" $PARTITION
add_sbatch_arg "--nodes" $NODES

sbatchx \
    $SBATCH_ARGS \
    "$COMMAND"
