#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --account="GOV112003"
#SBATCH --partition="gpNCHC_LLM"


SWEEP_ID=hare1822/test/h6fgvrx9
COUNT=10

TASK="
    python scripts/test/test_sweep.py \
        --sweep_id=$SWEEP_ID \
        --count=$COUNT \
"

srun bash -c "$TASK"
