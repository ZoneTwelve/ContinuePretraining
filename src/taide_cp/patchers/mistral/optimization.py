from transformers.models.mistral.modeling_mistral import (
    MistralPreTrainedModel, MistralRMSNorm)

from ..patcher import Patcher


class MistralOptimizationPatcher(Patcher):
    def __init__(
        self,
        fused_rms_norm: bool = True
    ) -> None:
        super().__init__()

        self.fuesed_rms_norm = fused_rms_norm

    def _validate(self, target: MistralPreTrainedModel):
        assert isinstance(target, MistralPreTrainedModel)
    
    def patch(self, target: MistralPreTrainedModel):
        for n, m in target.named_modules():           
            if self.fuesed_rms_norm and isinstance(m, MistralRMSNorm):
                self.patch_module(target, n, _get_fused_rms_norm(m, target.config.hidden_size))


def _get_fused_rms_norm(module: MistralRMSNorm, normalized_shape: int):
    try:
        from apex.normalization import FusedRMSNorm
    except ImportError:
        raise ImportError("apex is not available, Please install apex from source (https://github.com/NVIDIA/apex) to use the fused layernorm kernel")

    fused_rms_norm = FusedRMSNorm(
        normalized_shape=normalized_shape,
        eps=module.variance_epsilon,
        elementwise_affine=True,
        memory_efficient=True
    )
    fused_rms_norm.weight = module.weight
    return fused_rms_norm
