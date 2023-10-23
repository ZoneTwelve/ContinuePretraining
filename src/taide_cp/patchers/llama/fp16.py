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


def _decoder_layer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    outputs = LlamaDecoderLayer.forward(
        self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    outputs = (_clamp_fp16(outputs[0]), *outputs[1:])
    return outputs


class LLaMAFP16Patcher(Patcher):
    def _validate(self, target: LlamaPreTrainedModel):
        assert isinstance(target, LlamaPreTrainedModel)

    def patch(self, target: LlamaPreTrainedModel):
        for module in target.modules():
            if isinstance(module, LlamaDecoderLayer):
                self.patch_method(module.forward, _decoder_layer_forward)
