from ..datamodule import LightningDataModuleX
from .pretrain_datacollator import DataCollatorForPreTraining


class DataModuleForPreTraining(LightningDataModuleX):
    datacollator_cls = DataCollatorForPreTraining
