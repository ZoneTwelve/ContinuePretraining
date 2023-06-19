import re

from ..utils import parse_ev

__all__ = ['global_rank', 'get_host', 'get_port', 'get_host_and_port']


def global_rank():
    return parse_ev(int, 'SLURM_PROCID')

def world_size():
    return parse_ev(int, 'SLURM_NTASKS')

def get_host():
    nodelist = parse_ev(str, 'SLURM_NODELIST', '127.0.0.1')
    nodelist = re.sub(r'\[(.*?)[,-].*\]', '\\1', nodelist)
    nodelist = re.sub(r'\[(.*?)\]', '\\1', nodelist)
    root_node = nodelist.split(' ')[0].split(',')[0]
    return root_node

def get_port(offset: int = 10000):
    job_id = parse_ev(str, 'SLURM_JOB_ID', '1234')
    port = job_id[-4:]
    port = int(port) + offset
    return port

def get_host_and_port(port_offset: int = 10000):
    return get_host(), get_port(port_offset)
