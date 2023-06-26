import os
import re

from .utilities import parse_ev


class staticproperty(property):
    def __get__(self, cls, owner):
        return staticmethod(self.fget).__get__(None, owner)()

class SLURM:
    @staticproperty
    def is_slurm():
        return 'SLURM_JOB_ID' in os.environ

    @staticproperty
    def global_rank():
        return parse_ev(int, 'SLURM_PROCID')

    @staticproperty
    def local_rank():
        return parse_ev(int, 'SLURM_LOCALID')

    @staticproperty
    def world_size():
        return parse_ev(int, 'SLURM_NTASKS')
    
    @staticproperty
    def job_id():
        return parse_ev(str, 'SLURM_JOB_ID')

    @staticproperty
    def job_name():
        return parse_ev(str, 'SLURM_JOB_ID')
    
    @staticproperty
    def num_nodes():
        return parse_ev(int, 'SLURM_JOB_NUM_NODES')

    @staticproperty
    def num_tasks():
        return parse_ev(int, 'SLURM_NTASKS')

    @staticmethod
    def get_host():
        nodelist = parse_ev(str, 'SLURM_NODELIST', '127.0.0.1')
        nodelist = re.sub(r'\[(.*?)[,-].*\]', '\\1', nodelist)
        nodelist = re.sub(r'\[(.*?)\]', '\\1', nodelist)
        root_node = nodelist.split(' ')[0].split(',')[0]
        return root_node

    @staticmethod
    def get_port(offset: int = 10000):
        job_id = parse_ev(str, 'SLURM_JOB_ID', '1234')
        port = job_id[-4:]
        port = int(port) + offset
        return port

    @classmethod
    def get_host_and_port(cls, port_offset: int = 10000):
        return cls.get_host(), cls.get_port(port_offset)
