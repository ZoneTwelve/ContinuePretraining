NODES=1
GPUS=1
TIME=
ACCOUNT="GOV112003"
PARTITOIN="gpNCHC_LLM"
NODELIST=
JOB_NAME="interactive.srun.sh"

if [ $TIME ]
then
    TIME="--time=$TIME"
fi

if [ $JOB_NAME ]
then
    JOB_NAME="--job-name=$JOB_NAME"
fi

srun \
    $JOB_NAME \
    --nodes=$NODES \
    --gpus-per-node=$GPUS \
    --ntasks-per-node=1 \
    --cpus-per-task=$(($GPUS * 4)) \
    $TIME \
    --account=$ACCOUNT \
    --partition=$PARTITOIN \
    --nodelist=$NODELIST \
    --pty bash