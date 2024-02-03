import copy

import torch
from transformers.models.llama.modeling_llama import *

from ..patcher import Patcher


class LlamaOptimizationPatcher(Patcher):
    def __init__(
        self,
        fused_cross_entropy: bool = True,
        fused_rms_norm: bool = True,
        fused_rotary_embedding: bool = True,
        fused_swiglu: bool = True
    ) -> None:
        super().__init__()

        self.fused_cross_entropy = fused_cross_entropy
        self.fuesed_rms_norm = fused_rms_norm
        self.fused_rotary_embedding = fused_rotary_embedding
        self.fused_swiglu = fused_swiglu

    def _validate(self, target: LlamaPreTrainedModel):
        assert isinstance(target, LlamaPreTrainedModel)

    def extra_repr(self) -> str:
        return '\n'.join([
            f'fused_cross_entropy={self.fused_cross_entropy}',
            f'fuesed_rms_norm={self.fuesed_rms_norm}',
            f'fused_rotary_embedding={self.fused_rotary_embedding}',
            f'fused_swiglu={self.fused_swiglu}',
        ])
    
    def patch(self, target: LlamaPreTrainedModel):
        rotary_emb = None
        for n, m in target.named_modules():
            if self.fused_cross_entropy and isinstance(m, LlamaForCausalLM):
                self.patch_method(m.forward, _llama_for_causal_lm_forward)
            
            if self.fuesed_rms_norm and isinstance(m, LlamaRMSNorm):
                self.patch_method(m.forward, _llama_rms_norm_forward)

            if self.fused_rotary_embedding:
                ATTENTION_FORWARD_FN_MAPPING = {
                    LlamaAttention: _llama_attention_forward,
                    LlamaSdpaAttention: _llama_sdpa_attention_forward,
                    LlamaFlashAttention2: _llama_flash_attention_2_forward
                }

                if isinstance(m, LlamaAttention):
                    rotary_emb = rotary_emb or _get_rotary_embedding(m.rotary_emb)
                    del m.rotary_emb
                    m.rotary_emb = rotary_emb
                    self.patch_method(m.forward, ATTENTION_FORWARD_FN_MAPPING[m.__class__])
            
            if self.fused_swiglu and isinstance(m, LlamaMLP):
                self.patch_method(m.forward, _llama_mlp_forward)
        
        return target


def _llama_for_causal_lm_forward(
    self: LlamaForCausalLM,
    *args,
    labels: torch.LongTensor | None = None,
    return_dict: bool | None = None,
    **kwargs
):
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    outputs = LlamaForCausalLM.forward(self, *args, return_dict=return_dict, **kwargs)
    logits = outputs.logits if return_dict else outputs[0]

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss, _ = cross_entropy_loss(shift_logits, shift_labels)
        loss = loss.mean()

    if return_dict:
        outputs.loss = loss
    elif loss is not None:
        outputs = (loss,) + outputs

    return outputs


def _llama_rms_norm_forward(self: LlamaRMSNorm, x: torch.Tensor):
    from flash_attn.ops.triton.layer_norm import rms_norm_fn
    return rms_norm_fn(
        x,
        weight=self.weight,
        bias=None,
        eps=self.variance_epsilon
    )


def _set_cos_sin_cache(self: LlamaRotaryEmbedding, seq_len: int, device: torch.device, dtype: torch.dtype):
    self.__class__._set_cos_sin_cache(self, seq_len, device, dtype)
    self.cos_cached = self.cos_cached[:, :self.dim // 2]
    self.sin_cached = self.sin_cached[:, :self.dim // 2]


def _get_rotary_embedding(rotary_emb: LlamaRotaryEmbedding):
    rotary_emb = copy.deepcopy(rotary_emb)
    rotary_emb._set_cos_sin_cache = _set_cos_sin_cache.__get__(rotary_emb)
    rotary_emb._set_cos_sin_cache(rotary_emb.max_position_embeddings, rotary_emb.cos_cached.device, rotary_emb.cos_cached.dtype)
    return rotary_emb


def _llama_attention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    from flash_attn.layers.rotary import apply_rotary_emb

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

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

    kv_seq_len = key_states.shape[-2]
    seqlen_offsets = 0
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        seqlen_offsets = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        kv_seq_len += seqlen_offsets

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states = apply_rotary_emb(query_states, cos, sin, inplace=True, seqlen_offsets=seqlen_offsets)
    key_states = apply_rotary_emb(key_states, cos, sin, inplace=True, seqlen_offsets=seqlen_offsets)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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

    return attn_output, attn_weights, past_key_value


def _llama_sdpa_attention_forward(
    self: LlamaSdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Cache | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
):
    from flash_attn.layers.rotary import apply_rotary_emb

    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super(LlamaSdpaAttention, self).forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    kv_seq_len = key_states.shape[-2]
    seqlen_offsets = 0
    if past_key_value is not None:
        seqlen_offsets = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        kv_seq_len += seqlen_offsets

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    query_states = apply_rotary_emb(query_states, cos, sin, inplace=True, seqlen_offsets=seqlen_offsets)
    key_states = apply_rotary_emb(key_states, cos, sin, inplace=True, seqlen_offsets=seqlen_offsets)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def _llama_flash_attention_2_forward(
    self: LlamaFlashAttention2,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    from flash_attn.layers.rotary import apply_rotary_emb

    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    kv_seq_len = key_states.shape[-2]
    seqlen_offsets = 0
    if past_key_value is not None:
        seqlen_offsets = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        kv_seq_len += seqlen_offsets

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states = apply_rotary_emb(query_states, cos, sin, inplace=True, seqlen_offsets=seqlen_offsets)
    key_states = apply_rotary_emb(key_states, cos, sin, inplace=True, seqlen_offsets=seqlen_offsets)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states.transpose(1, 2), value_states.transpose(1, 2), self.layer_idx, cache_kwargs)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _llama_mlp_forward(self: LlamaMLP, x: torch.Tensor):
    from flash_attn.ops.activations import swiglu
    return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
