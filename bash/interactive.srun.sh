NODES=1
GPUS=1
TIME=
ACCOUNT="GOV112003"
PARTITOIN="gpNCHC_LLM" # gpNCHC_LLM gtest gp1d
NODELIST=
JOB_NAME="interactive.srun.sh"

if [ $1 ]
then
    GPUS=$1
fi

if [ $2 ]
then
    PARTITOIN=$2
fi

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
    --ntasks-per-node=1 \
    --cpus-per-task=$(($GPUS * 4)) \
    $TIME \
    --account=$ACCOUNT \
    --partition=$PARTITOIN \
    --nodelist=$NODELIST \
    --pty bash
