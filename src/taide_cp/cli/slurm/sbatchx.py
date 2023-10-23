import logging
import os
import re
import subprocess
import time
from pathlib import Path

from ..utils import Fire

logging.basicConfig()
logger = logging.getLogger('sbatchx')


@Fire
def main(
    command: str,
    *,
    job_name: str = Path(__file__).stem,
    nodes: int = 1,
    gpus_per_node: int = 8,
    cpus_per_task: int = 4,
    ntasks_per_node: int = 8,
    account: str = os.environ.get('SLURM_DEFAULT_ACCOUNT'),
    partition: str = os.environ.get('SLURM_DEFAULT_PARTITION'),
    output: str = 'logs/slurm/%j-%x.log',
    verbosity: int | str = logging.WARN,
    attach: bool = True
):
    logger.setLevel(verbosity)

    command = command.strip()
    command = re.sub(r'\s+', ' ', command)
    command = f'srun {command}'
    logger.info(command)

    p = subprocess.run(
        [
            'sbatch',
            '--job-name', job_name,
            '--nodes', str(nodes),
            '--gpus-per-node', str(gpus_per_node),
            '--cpus-per-task', str(cpus_per_task),
            '--ntasks-per-node', str(ntasks_per_node),
            '--account', account,
            '--partition', partition,
            '--output', output,
            '--wrap', command
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    m = re.search(r'Submitted batch job (\d+)', p.stdout.decode())
    if m:
        job_id = m[1]
        print(job_id)
    else:
        print(p.stderr.decode())
        return
    
    if not attach:
        return

    t = time.time()
    while True:
        try:
            subprocess.run(['sattach', f'{job_id}.0'])

            if time.time() - t > 30:
                break

            time.sleep(1)
        except KeyboardInterrupt:
            break
