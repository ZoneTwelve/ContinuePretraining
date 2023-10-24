from typing import TypedDict

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.modeling_utils import no_init_weights

from ...lightning import EnhancedDeepSpeedStrategy
from ...metrics import Perplexity
from ...utils import ContextManagers
from ..hf import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ..lr_schedulers import get_lr_scheduler
from ..modules import PartiallyFrozenEmbedding, PartiallyFrozenLinear
from .causal_lm_config import *


class _BatchType(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def tie_partially_frozen_weights(model: PreTrainedModel):
    if getattr(model.config, 'tie_word_embeddings'):
        input_embeddings: PartiallyFrozenEmbedding = model.get_input_embeddings()
        output_embeddings: PartiallyFrozenLinear = model.get_output_embeddings()
        output_embeddings.frozen_linear.weight = input_embeddings.frozen_embedding.weight
        output_embeddings.trainable_linear.weight = input_embeddings.trainable_embedding.weight


def extend_vocabs(
    model: PreTrainedModel,
    new_num_tokens: int,
    config: ExtendVocabConfig,
):
    import deepspeed
    from transformers.integrations import is_deepspeed_zero3_enabled

    old_num_tokens = model.config.vocab_size
    model.resize_token_embeddings(new_num_tokens, config.pad_to_multiple_of)
    input_embeddings: nn.Embedding = model.get_input_embeddings()
    output_embeddings: nn.Linear = model.get_output_embeddings()

    with deepspeed.zero.GatheredParameters(
        [input_embeddings.weight, output_embeddings.weight],
        modifier_rank=0,
        enabled=is_deepspeed_zero3_enabled()
    ):
        if config.initializing_strategy == InitializingStrategy.MEAN:
            input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight.data[:old_num_tokens].mean(0))
            output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight.data[:old_num_tokens].mean(0))
        elif config.initializing_strategy == InitializingStrategy.SAMPLE:
            mask = torch.randperm(old_num_tokens, device=model.device)[:new_num_tokens - old_num_tokens]
            input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight.data[mask])
            output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight.data[mask])

    if config.training_strategy == TrainingStrategy.NEW_TOKENS_ONLY:
        model.requires_grad_(False)
        model.set_input_embeddings(PartiallyFrozenEmbedding(input_embeddings, old_num_tokens))
        model.set_output_embeddings(PartiallyFrozenLinear(output_embeddings, old_num_tokens))
        tie_partially_frozen_weights(model)
    elif config.training_strategy == TrainingStrategy.ALL_TOKENS:
        model.requires_grad_(False)
        model.get_input_embeddings().requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)
    else:
        model.requires_grad_(True)
    

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

    def __init__(self, config: LitCausalLMConfig) -> None:       
        super().__init__()

        self.save_hyperparameters({'config': config})

        self.config = config
        self.model_config = self.model_config_class.from_pretrained(self.config.model_path, revision=config.revision)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.config.tokenizer_path, revision=config.revision)

        assert self.config.extend_vocab or len(self.tokenizer) <= self.model_config.vocab_size

        self.batch_perplexity = Perplexity(ignore_index=-100)
        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)

    def _get_model_kwargs(self):
        kwargs = dict(
            low_cpu_mem_usage=True,
            config=self.model_config,
            revision=self.config.revision
        )
        if isinstance(self.strategy, EnhancedDeepSpeedStrategy):
            kwargs['low_cpu_mem_usage'] = not self.strategy.zero_stage_3
        return kwargs
    
    def _call_patchers(self):
        for patcher in self.config.patchers:
            patcher(self.model)

    def configure_sharded_model(self) -> None:
        context_managers = []
        
        load_from_checkpoint = self._trainer is not None and self.trainer.ckpt_path is not None
        if load_from_checkpoint:
            context_managers.append(no_init_weights())    
        
        if isinstance(self.strategy, EnhancedDeepSpeedStrategy):
            import deepspeed
            if load_from_checkpoint and self.strategy.zero_stage_3:
                context_managers.append(deepspeed.zero.Init(config_dict_or_path=self.strategy.config))

        if not load_from_checkpoint:
            self.model: PreTrainedModel = self.model_class.from_pretrained(self.config.model_path, **self._get_model_kwargs())
        else:
            with ContextManagers(context_managers):
                self.model = self.model_class(self.model_config)

        self._call_patchers()

        if self.config.extend_vocab:
            extend_vocabs(
                self.model,
                len(self.tokenizer),
                config=self.config.extend_vocab,
            )
        
    def configure_optimizers(self):
        optimizer_config = {}

        optimizer_cls = AdamW
        if isinstance(self.strategy, EnhancedDeepSpeedStrategy):
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
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
        self.model.gradient_checkpointing_enable()

        if self.trainer.ckpt_path is not None:
            if self.config.extend_vocab and self.config.extend_vocab.training_strategy == TrainingStrategy.NEW_TOKENS_ONLY:
                tie_partially_frozen_weights(self.model)
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

    def on_train_end(self) -> None:
        self.model.gradient_checkpointing_disable()

    def on_validation_start(self) -> None:
        self.model.gradient_checkpointing_disable()

    def validation_step(self, batch: _BatchType, batch_idx: int, dataloader_idx: int = 0):
        batch_size = batch['input_ids'].size(0)
        loss = self.compute_loss(batch)

        self.val_perplexity.update(loss, batch['labels'])
        self.log('loss', loss, prog_bar=True, logger=False, sync_dist=True)
        self.log('Loss/Val', loss, sync_dist=True)
        self.log('Perplexity/Val', self.val_perplexity, batch_size=batch_size, sync_dist=True)

    def on_validation_end(self) -> None:
        self.model.gradient_checkpointing_enable()
