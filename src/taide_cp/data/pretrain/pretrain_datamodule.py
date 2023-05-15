from ...utils.data import LightningDataModuleX
from .pretrain_datacollator import PretrainDataCollator


class PretrainDataModule(LightningDataModuleX):
    datacollator_cls = PretrainDataCollator
