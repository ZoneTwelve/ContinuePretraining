import logging
import socket
import threading
import time
from enum import IntEnum, auto
from functools import partial
from typing import Callable, Optional

import flask
import requests
import wandb

from ..utils.slurm import get_host, get_port, global_rank, world_size

__all__ = ['SweepRunner', 'SweepRunnerState']

class SweepRunnerState(IntEnum):
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    FINISHED = auto()


class SweepRunner:
    __state_keys__ = ['config', 'state', 'ready_clients']
    
    @property
    def address(self):
        return f'http://{self.host}:{self.port}'

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, port_offset: int = 10000) -> None:
        self.host = host or get_host()
        self.port = port or get_port(port_offset)

        if global_rank() == 0:
            logging.getLogger('werkzeug').disabled = True
            self.create_server()

        self.config = None
        self.state: SweepRunnerState = SweepRunnerState.PENDING
        self.ready_clients = []

    def create_server(self):
        def index():
            return flask.jsonify({k: getattr(self, k) for k in self.__state_keys__})
        
        def set_ready():
            data = flask.request.json 
            if data is not None and data['global_rank'] not in self.ready_clients:
                self.ready_clients.append(data['global_rank'])
            return flask.Response(status=204)

        self.server = flask.Flask(self.__class__.__name__)
        self.server.add_url_rule('/', view_func=index, methods=['GET'])
        self.server.add_url_rule('/set_ready', view_func=set_ready, methods=['POST'])
        self.server_thread = threading.Thread(target=lambda: self.server.run(self.host, self.port), daemon=True)
        self.server_thread.start()    
    
    def wait_for_server(self):
        c = None
        while c != 0:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            c = s.connect_ex((self.host, self.port))
    
    def wait_for_clients(self):
        while len(self.ready_clients) < world_size():
            time.sleep(1)
    
    def set_ready(self):
        requests.post(f'{self.address}/set_ready', json=dict(global_rank=global_rank()))

    def sync_state(self):
        state = requests.get(f'{self.address}').json()
        self.__dict__.update(state)

    def agent_function(self, function: Callable[..., None]):
        run = wandb.init()

        self.config = dict(run.config)
        self.state = SweepRunnerState.READY
        self.set_ready()

        self.wait_for_clients()
        self.state = SweepRunnerState.PENDING
        self.ready_clients = []

        function(**self.config)

    def run(self, sweep_id: int, function: Callable[..., None], count: Optional[int] = None):
        if global_rank() == 0:
            wandb.agent(sweep_id, function=partial(self.agent_function, function), count=count)
            self.state = SweepRunnerState.FINISHED
        else:
            self.wait_for_server()
            
            while True:
                self.sync_state()

                if self.state == SweepRunnerState.FINISHED:
                    break
                elif self.state == SweepRunnerState.PENDING:
                    time.sleep(1)
                elif self.state == SweepRunnerState.READY:
                    self.set_ready()
                    function(**self.config)
