import multiprocess

from taide_cp.cli import TaideCPLightningCLI
from taide_cp.data import *
from taide_cp.lightning import *
from taide_cp.models import *
from taide_cp.patchers import *

if __name__ == '__main__':
    multiprocess.set_start_method('spawn')
    TaideCPLightningCLI()
