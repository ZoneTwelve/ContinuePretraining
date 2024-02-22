import inspect
import logging
from contextlib import nullcontext
from typing import TypedDict

import deepspeed
import lightning as L
import torch
import torch.distributed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch import nn
from torch.optim import AdamW
from torchmetrics import Metric
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.modeling_utils import no_init_weights

from ...lightning import EnhancedDeepSpeedStrategy
from ...metrics import Perplexity
from ..hf import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ..lr_schedulers import get_lr_scheduler
from ..modules import PartiallyFrozenEmbedding, PartiallyFrozenLinear
from .causal_lm_config import *

logger = logging.getLogger(__name__)

class _BatchType(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class LitCausalLM(L.LightningModule):
    model_class = AutoModelForCausalLM
    model_config_class = AutoConfig
    tokenizer_class = AutoTokenizer

    model: PreTrainedModel
    model_config: PretrainedConfig
    tokenizer: PreTrainedTokenizerBase

    @property
    def strategy(self):
        return None if self._trainer is None else self.trainer.strategy
    
    @property
    def is_load_from_checkpoint(self) -> bool:
        if self._trainer is None:
            return False
        
        if self.trainer.ckpt_path is not None:
            return True

        current_frame = inspect.currentframe().f_back
        while current_frame is not None:
            f_locals = current_frame.f_locals
            current_frame = current_frame.f_back

            if isinstance(f_locals.get('self', None), L.Trainer) and 'ckpt_path' in f_locals:
                return f_locals['ckpt_path'] is not None
    
    def __init__(self, config: LitCausalLMConfig) -> None:       
        super().__init__()

        self.save_hyperparameters({'config': config})

        self.config = config
        self.model_config = self.model_config_class.from_pretrained(self.config.model_path, **config.model_kwargs)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.config.tokenizer_path, **config.tokenizer_kwargs)

        assert self.config.extend_vocab or len(self.tokenizer) <= self.model_config.vocab_size

        self.batch_perplexity = Perplexity(ignore_index=-100)
        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)
    
    def _call_patchers(self):
        for patcher in self.config.patchers:
            patcher(self.model)

    def _construct_model_from_config(self) -> PreTrainedModel:
        ctx = nullcontext()
        if isinstance(self.strategy, EnhancedDeepSpeedStrategy):
            ctx = self.strategy.model_init_context()

        with (
            no_init_weights(),
            ctx
        ):
            if self.model_class is AutoModelForCausalLM:
                return self.model_class.from_config(
                    self.model_config,
                    trust_remote_code=self.config.model_kwargs.get('trust_remote_code', None),
                    attn_implementation=self.config.model_kwargs.get('attn_implementation', None),
                )
            return self.model_class(self.model_config)
    
    def _construct_model_from_pretrained(self) -> PreTrainedModel:
        kwargs = dict(
            low_cpu_mem_usage=True
        )
        kwargs |= self.config.model_kwargs
        return self.model_class.from_pretrained(self.config.model_path, **kwargs)
    
    def _get_pretrained_weights(self):
        model = self._construct_model_from_pretrained()
        state_dict = model.state_dict()
        return state_dict
    
    def _zero3_load_state_dict_into_model(self, model: PreTrainedModel, state_dict: dict[str, torch.Tensor] | None):
        def chunk(l: list, n: int):
            x = []
            s = -(len(l) // -n)
            for i in range(0, len(l), s):
                x += [l[i:i + s]]
            return x
        
        keys = [n for n, _ in model.named_parameters()]
        key_shards = chunk(keys, 10)        

        if self.global_rank == 0:
            current = 0
            total = len(list(model.parameters()))

        for sharded_keys in key_shards:
            sharded_params = {n: p for n, p in model.named_parameters() if n in sharded_keys}
            with deepspeed.zero.GatheredParameters(list(sharded_params.values()), modifier_rank=0):
                if self.global_rank == 0:
                    for n, p in sharded_params.items():
                        p.data.copy_(state_dict[n], non_blocking=True)
                        
                        current += 1
                        logger.debug(f'{current}/{total} {n}')
        
        return model
    
    def tie_partially_frozen_weights(self):
        if getattr(self.model.config, 'tie_word_embeddings'):
            input_embeddings: PartiallyFrozenEmbedding = self.model.get_input_embeddings()
            output_embeddings: PartiallyFrozenLinear = self.model.get_output_embeddings()
            assert isinstance(input_embeddings, PartiallyFrozenEmbedding)
            assert isinstance(output_embeddings, PartiallyFrozenLinear)
            output_embeddings.frozen_linear.weight = input_embeddings.frozen_embedding.weight
            output_embeddings.trainable_linear.weight = input_embeddings.trainable_embedding.weight
    
    def _maybe_extend_vocabs(self):
        config = self.config.extend_vocab
        model = self.model
        new_num_tokens = len(self.tokenizer)
        old_num_tokens = self.model.config.vocab_size

        if not config:
            return
        
        logger.info(f'[{self.global_rank}] Extend vocabs')

        is_zero3 = getattr(self.strategy, 'zero_stage_3', False)
        if is_zero3:
            hf_ds_config = HfDeepSpeedConfig(self.strategy.config)
        
        model.resize_token_embeddings(new_num_tokens, config.pad_to_multiple_of)
        
        input_embeddings: nn.Embedding = model.get_input_embeddings()
        output_embeddings: nn.Linear = model.get_output_embeddings()

        with deepspeed.zero.GatheredParameters(
            [input_embeddings.weight, output_embeddings.weight],
            modifier_rank=0,
            enabled=is_zero3
        ):
            if config.initializing_strategy == InitializingStrategy.MEAN:
                input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight.data[:old_num_tokens].mean(0))
                output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight.data[:old_num_tokens].mean(0))
            elif config.initializing_strategy == InitializingStrategy.SAMPLE:
                mask = torch.randperm(old_num_tokens, device=model.device)[:new_num_tokens - old_num_tokens]
                input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight.data[mask])
                output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight.data[mask])
        
        input_embeddings.weight.data[len(self.tokenizer):].zero_()
        if 'pad_token' in self.tokenizer.special_tokens_map:
            input_embeddings.weight.data[self.tokenizer.pad_token_id].zero_()

        if config.training_strategy == TrainingStrategy.NEW_TOKENS_ONLY:
            model.requires_grad_(False)
            model.set_input_embeddings(PartiallyFrozenEmbedding(input_embeddings, old_num_tokens))
            model.set_output_embeddings(PartiallyFrozenLinear(output_embeddings, old_num_tokens))
            self.tie_partially_frozen_weights()
        elif config.training_strategy == TrainingStrategy.ALL_TOKENS:
            model.requires_grad_(False)
            model.get_input_embeddings().requires_grad_(True)
            model.get_output_embeddings().requires_grad_(True)
        else:
            model.requires_grad_(True)

    def _convert_metrics(self) -> None:
        for m in self.modules():
            if isinstance(m, Metric):
                m.set_dtype(torch.float)

    def configure_model(self) -> None:
        is_zero3 = getattr(self.strategy, 'zero_stage_3', False)

        if self.is_load_from_checkpoint:
            logger.info('Construct un-initialized model')
            self.model = self._construct_model_from_config()
        elif is_zero3:
            logger.info('Construct un-initialized model')
            self.model = self._construct_model_from_config()
            logger.info('Load pretrained weights from disk')
            state_dict = self._get_pretrained_weights() if self.global_rank == 0 else None
            logger.info('Load pretrained weights into model')
            self._zero3_load_state_dict_into_model(self.model, state_dict)
        else:
            logger.info('Load pretrained model')
            self.model = self._construct_model_from_pretrained()

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable({'use_reentrant': False})
                
        self._maybe_extend_vocabs()
        self._call_patchers()
        self._convert_metrics()

        if self.global_rank == 0:
            logger.info(f'Model Architecture:\n{self.model}')

    def configure_optimizers(self):
        optimizer_config = {}

        optimizer_cls = AdamW
        if isinstance(self.strategy, EnhancedDeepSpeedStrategy):
            optimizer_cls = DeepSpeedCPUAdam if self.strategy.offload_optimizer else FusedAdam

        optimizer_config['optimizer'] = optimizer_cls(
            self.parameters(),
            lr=self.config.optimizer_config.lr,
            betas=self.config.optimizer_config.betas,
            eps=self.config.optimizer_config.eps,
            weight_decay=self.config.optimizer_config.weight_decay
        )

        optimizer_config['lr_scheduler'] = {
            'scheduler': get_lr_scheduler(
                scheduler_type=self.config.optimizer_config.lr_scheduler_type,
                optimizer=optimizer_config['optimizer'],
                num_warmup_steps=self.config.optimizer_config.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                min_lr_factor=self.config.optimizer_config.min_lr_factor,
            ),
            'interval': 'step',
        }

        return optimizer_config

    def compute_loss(self, batch: _BatchType) -> torch.Tensor:
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            use_cache=False,
        )
        return output.loss.float()
    
    def on_train_start(self) -> None:
        if self.is_load_from_checkpoint:
            if self.config.extend_vocab and self.config.extend_vocab.training_strategy == TrainingStrategy.NEW_TOKENS_ONLY:
                self.tie_partially_frozen_weights()
            else:
                self.model.tie_weights()

    def training_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['input_ids'].size(0)
        loss = self.compute_loss(batch)

        self.log('loss', loss, prog_bar=True, logger=False)
        self.log('Loss/Train/Step', loss)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.train_perplexity.update(loss, batch['labels'])
        self.batch_perplexity.update(loss, batch['labels'])
        self.log('Perplexity/Train/Step', self.batch_perplexity.compute())
        self.batch_perplexity.reset()
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log('Perplexity/Train', self.train_perplexity, sync_dist=True)

    def validation_step(self, batch: _BatchType, batch_idx: int, dataloader_idx: int = 0):
        batch_size = batch['input_ids'].size(0)
        loss = self.compute_loss(batch)

        self.val_perplexity.update(loss, batch['labels'])
        self.log('loss', loss, prog_bar=True, logger=False, sync_dist=True)
        self.log('Loss/Val', loss, sync_dist=True)
        self.log('Perplexity/Val', self.val_perplexity, batch_size=batch_size, sync_dist=True)
