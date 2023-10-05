import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaDecoderLayer,
                                                      LlamaMLP, LlamaModel,
                                                      LlamaPreTrainedModel,
                                                      apply_rotary_pos_emb,
                                                      repeat_kv)

from taide_cp.patchers.patcher import Patcher


def _clamp_fp16(x: torch.Tensor):
    if x.dtype == torch.half:
        max_dtype = torch.finfo(torch.half).max
        clamp_value = torch.where(torch.isinf(x).any(), max_dtype - 1000, max_dtype)
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    combined_attention_mask = LlamaModel._prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

    dtype_min = torch.tensor(
        torch.finfo(inputs_embeds.dtype).min,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device
    )
    combined_attention_mask = torch.clamp_min(combined_attention_mask, dtype_min)
    # for every prompt instance, examine if there is any token who has a full masking attention tensor
    fully_masked = (combined_attention_mask == dtype_min).all(dim=-1)
    if torch.any(fully_masked):
        # This can happen when your prompts are left padded and you want to masked out the left padded tokens
        # However the operation 'expanded_attn_mask + combined_attention_mask' above will cause those left padded tokens
        # to have a fully masked attention tensor, and cause numerical instability(inf) during inference. Therefore
        # for those tokens who has attention tensor fully masked, force them to at least attend to itself.
        indices = torch.arange(
            combined_attention_mask.size(2) + past_key_values_length, device=combined_attention_mask.device
        )
        indices = indices.unsqueeze(0).unsqueeze(0).expand_as(fully_masked)
        # Set the n-th elements in the n-th tensors to be zero
        combined_attention_mask[fully_masked, indices[fully_masked]] = 0.0
    return combined_attention_mask


def _attention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    hidden_states = _clamp_fp16(hidden_states)

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = _clamp_fp16(attn_weights)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return _clamp_fp16(attn_output), attn_weights, past_key_value


# def _attention_forward(
#         self: LlamaAttention,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         attn_output, attn_weights, past_key_value = LlamaAttention.forward(
#             self,
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache
#         )

#         attn_output = _clamp_fp16(attn_output)
#         return attn_output, attn_weights, past_key_value


def _mlp_forward(self: LlamaMLP, x: torch.Tensor):
    x = LlamaMLP.forward(self, x)
    x = _clamp_fp16(x)
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
            # if isinstance(module, LlamaModel):
            #     self.patch_method(module._prepare_decoder_attention_mask, _prepare_decoder_attention_mask)

            # if isinstance(module, LlamaAttention):
            #     self.patch_method(module.forward, _attention_forward)

            # if isinstance(module, LlamaMLP):
            #     self.patch_method(module.forward, _mlp_forward)

            if isinstance(module, LlamaDecoderLayer):
                self.patch_method(module.forward, _decoder_layer_forward)
