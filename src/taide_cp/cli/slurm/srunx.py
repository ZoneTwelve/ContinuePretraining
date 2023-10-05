import os
import subprocess
from pathlib import Path

from ..utils import Fire


@Fire
def main(
    job_name: str = Path(__file__).stem,
    nodes: int = 1,
    gpus_per_node: int = 1,
    cpus_per_task: int | None = None,
    ntasks_per_node: int = 1,
    account: str = os.environ.get('SLURM_DEFAULT_ACCOUNT'),
    partition: str = os.environ.get('SLURM_DEFAULT_PARTITION'),
):
    cpus_per_task = cpus_per_task or 4 * gpus_per_node

    args = [
        'srun',
        '--job-name', job_name,
        '--nodes', str(nodes),
        '--gpus-per-node', str(gpus_per_node),
        '--cpus-per-task', str(cpus_per_task),
        '--ntasks-per-node', str(ntasks_per_node),
        '--account', account,
        '--partition', partition,
        '--pty', 'bash'
    ]

    try:
        p = subprocess.Popen(args)
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
