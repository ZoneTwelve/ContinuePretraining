from typing import Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import (LlamaDecoderLayer,
                                                      LlamaPreTrainedModel)

from taide_cp.patchers.patcher import Patcher


def _clamp_fp16(x: torch.Tensor):
    if x.dtype == torch.half:
        max_dtype = torch.finfo(torch.half).max
        clamp_value = torch.where(torch.isinf(x).any(), max_dtype - 1000, max_dtype)
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def _decoder_layer_forward(self: LlamaDecoderLayer, *args, **kwargs):
    outputs = LlamaDecoderLayer.forward(self, *args, **kwargs)
    outputs = (_clamp_fp16(outputs[0]), *outputs[1:])
    return outputs


class LLaMAFP16Patcher(Patcher):
    def _validate(self, target: LlamaPreTrainedModel):
        assert isinstance(target, LlamaPreTrainedModel)

    def patch(self, target: LlamaPreTrainedModel):
        for module in target.modules():
            if isinstance(module, LlamaDecoderLayer):
                self.patch_method(module.forward, _decoder_layer_forward)
