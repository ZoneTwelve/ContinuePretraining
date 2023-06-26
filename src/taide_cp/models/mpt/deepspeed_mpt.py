import math
import warnings
from typing import List, Tuple

import deepspeed
import torch
from torch import ByteTensor, FloatTensor, LongTensor, nn

from taide_cp.models.mpt.configuration_mpt import MPTConfig

from .modeling_mpt import (BaseModelOutputWithPast, MPTForCausalLM, MPTModel,
                           logger)


class DeepSpeedMPTModel(MPTModel):
    def forward(self, input_ids: LongTensor, past_key_values: List[Tuple[FloatTensor]] | None = None, attention_mask: ByteTensor | None = None, prefix_mask: ByteTensor | None = None, sequence_id: LongTensor | None = None, labels: LongTensor | None = None, return_dict: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, use_cache: bool | None = None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()
        if not return_dict:
            raise NotImplementedError('return_dict False is not implemented yet for MPT')
        if output_attentions:
            if self.attn_impl != 'torch':
                raise NotImplementedError('output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`.')
        if attention_mask is not None and attention_mask[:, 0].sum() != attention_mask.shape[0] and self.training:
            raise NotImplementedError('MPT does not support training with left padding.')
        if self.prefix_lm and prefix_mask is None:
            raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')
        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError('sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True ' + 'and the model is in train mode.')
            elif self.attn_uses_sequence_id is False and sequence_id is not None:
                warnings.warn('MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' + 'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.')
        S = input_ids.size(1)
        assert S <= self.config.max_seq_len, f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'
        tok_emb = self.wte(input_ids)
        if self.alibi:
            x = tok_emb
        else:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(f'past_key_values must provide a past_key_value for each attention ' + f'layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.config.n_layers!r}).')
                past_position = past_key_values[0][0].size(1)
                if self.attn_impl == 'torch':
                    past_position = past_key_values[0][0].size(3)
            if S + past_position > self.config.max_seq_len:
                raise ValueError(f'Cannot forward input with past sequence length {past_position} and current sequence length {S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.')
            pos = torch.arange(past_position, S + past_position, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            if attention_mask is not None:
                pos = torch.clamp(pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, past_position:], min=0)
            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb
        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            x_shrunk = x * self.embedding_fraction + x.detach() * (1 - self.embedding_fraction)
            assert isinstance(self.emb_drop, nn.Module)
            x = self.emb_drop(x_shrunk)
        (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=torch.float32, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for (b_idx, block) in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = past_key_values[b_idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*args, **kwargs):
                        return module(*args, **kwargs)
                    return custom_forward

                (x, attn_weights, past_key_value) = deepspeed.checkpointing.checkpoint(
                    create_custom_forward(block),
                    x,
                    None,
                    attn_bias,
                    attention_mask,
                    self.is_causal,
                )
            else:
                (x, attn_weights, past_key_value) = block(x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=self.is_causal)
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value
            if output_attentions:
                assert all_self_attns is not None
                all_self_attns = all_self_attns + (attn_weights,)
        
        x = self.norm_f(x)
        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (x,)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values, hidden_states=all_hidden_states, attentions=all_self_attns)


class DeepSpeedMPTForCausalLM(MPTForCausalLM):
    def __init__(self, config: MPTConfig):
        super(MPTForCausalLM, self).__init__(config)

        if not config.tie_word_embeddings:
            raise ValueError('MPTForCausalLM only supports tied word embeddings')
        self.transformer = DeepSpeedMPTModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, device=config.init_device)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale
