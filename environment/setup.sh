CONDA=conda
if type mamba > /dev/null 2>&1; then
    CONDA=mamba
fi

CONDA_ENV_NAME=taide-cp
CONDA_ENV_FILE=environment/default.yaml

$CONDA env create -f $CONDA_ENV_FILE
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

prompt() {
    echo -n $1 >&2
    read x
    echo -n $x
}

conda env config vars set \
    SLURM_DEFAULT_ACCOUNT=$(prompt "SLURM_DEFAULT_ACCOUNT=") \
    SLURM_DEFAULT_PARTITION=$(prompt "SLURM_DEFAULT_PARTITION=") \

add_command() {
    FILE_PATH=$CONDA_PREFIX/bin/$1
    echo $2 > $FILE_PATH
    chmod +x $FILE_PATH
}

SQUEUE_FORMAT="%.6i %.30j %.10u %.7Q %.7T %.10M %.5D %.5b %R"

add_command sq "squeue -a -u \$USER -o \"$SQUEUE_FORMAT\""
add_command sqa "squeue -a -A \$SLURM_DEFAULT_ACCOUNT -o \"$SQUEUE_FORMAT\""
add_command sqp "squeue -a -p \$SLURM_DEFAULT_PARTITION -o \"$SQUEUE_FORMAT\""
add_command si "sinfo -p \$SLURM_DEFAULT_PARTITION"


# Fix bash-completion error messages
ln $CONDA_PREFIX/share/bash-completion/completions/_yum $CONDA_PREFIX/share/bash-completion/completions/yum
