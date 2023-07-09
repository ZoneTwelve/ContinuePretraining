from .lightning_module_for_pre_training import LightningModuleForPreTraining
from .llama import LLaMALightningModuleForPreTraining
from .mpt import MPTLightningModuleForPreTraining
from .opt import OPTLightningModuleForPreTraining

MODELS_FOR_PRE_TRAINING = {v.name: v for v in globals().values() if isinstance(v, type) and issubclass(v, LightningModuleForPreTraining)}
