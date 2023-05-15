from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Type, TypedDict

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch import nn
from torch.optim import AdamW
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.modeling_utils import no_init_weights

from ...lightning import DeepSpeedStrategy
from ...metrics import Perplexity
from ...utils import ContextManagers
from ...utils.training import (LightningModuleX, PartiallyFrozenEmbedding,
                               PartiallyFrozenLinear, get_lr_scheduler)


class _BatchType(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


InitializingStrategy = Literal[None, 'mean', 'sample']
FreezingStrategy = Literal[None, 'exclude_new', 'exclude_all']


def tie_partially_frozen_weights(model: PreTrainedModel):
    if getattr(model.config, 'tie_word_embeddings'):
        input_embeddings: PartiallyFrozenEmbedding = model.get_input_embeddings()
        output_embeddings: PartiallyFrozenLinear = model.get_output_embeddings()
        output_embeddings.w1 = input_embeddings.w1
        output_embeddings.w2 = input_embeddings.w2


def extend_tokens(
    model: PreTrainedModel,
    new_num_tokens: int,
    initializing_strategy: InitializingStrategy = None,
    freeze_old: bool = False,
):
    old_num_tokens = model.config.vocab_size
    model.resize_token_embeddings(new_num_tokens)
    input_embeddings: nn.Embedding = model.get_input_embeddings()
    output_embeddings: nn.Linear = model.get_output_embeddings()
    
    if initializing_strategy == 'mean':
        input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight.data[:old_num_tokens].mean(0))
        output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight.data[:old_num_tokens].mean(0))
    elif initializing_strategy == 'sample':
        mask = torch.randperm(old_num_tokens, device=model.device)[:new_num_tokens - old_num_tokens]
        input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight.data[mask])
        output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight.data[mask])

    if freeze_old:
        model.set_input_embeddings(PartiallyFrozenEmbedding(input_embeddings, old_num_tokens))
        model.set_output_embeddings(PartiallyFrozenLinear(output_embeddings, old_num_tokens))
        tie_partially_frozen_weights(model)


class LightningModuleForPreTraining(LightningModuleX):
    model_class: Type[PreTrainedModel]
    ds_model_class: Type[PreTrainedModel]
    config_class: Type[PretrainedConfig]
    tokenizer_class: Type[PreTrainedTokenizerBase]

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        extend_tokens: bool = False,
        initializing_strategy: InitializingStrategy = None,
        freezing_strategy: FreezingStrategy = None,
        lr: Optional[float] = 1e-5,
        betas: Optional[Tuple[float, float]] = (0.9, 0.95),
        weight_decay: Optional[float] = 1e-1,
        lr_scheduler_type: Literal[None, 'linear', 'cosine'] = None,
        num_warmup_steps: int = 0,
        min_lr_factor: float = 0.1,
        _load_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['_load_from_checkpoint'])

        self.model_path = Path(model_path)
        tokenizer_path = tokenizer_path or model_path

        self.config: PretrainedConfig = self.config_class.from_pretrained(self.model_path)
        self.tokenizer: PreTrainedTokenizerBase = self.tokenizer_class.from_pretrained(tokenizer_path)

        self.extend_tokens = extend_tokens
        self.initializing_strategy = initializing_strategy
        self.freezing_strategy = freezing_strategy

        assert self.extend_tokens or len(self.tokenizer) <= self.config.vocab_size

        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.min_lr_factor = min_lr_factor

        self._load_from_checkpoint = _load_from_checkpoint

        self.batch_perplexity = Perplexity(ignore_index=-100)
        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)

    def configure_sharded_model(self) -> None:       
        context_managers = []
        
        if self._load_from_checkpoint:
            context_managers.append(no_init_weights())    
        
        model_cls = self.model_class
        if isinstance(self.strategy, DeepSpeedStrategy):
            context_managers.append(self.strategy.deepspeed_init_context)
            model_cls = self.ds_model_class or model_cls

        if not self._load_from_checkpoint:
            self.model: PreTrainedModel = model_cls.from_pretrained(self.model_path)
        else:
            with ContextManagers(context_managers):
                self.model = model_cls(self.config)

        if self.extend_tokens:           
            extend_tokens(
                self.model,
                len(self.tokenizer),
                initializing_strategy=self.initializing_strategy,
                freeze_old=self.freezing_strategy == 'exclude_new',
            )

            if self.freezing_strategy == 'exclude_new':
                self.freeze()
                input_embeddings: PartiallyFrozenEmbedding = self.model.get_input_embeddings()
                output_embeddings: PartiallyFrozenLinear = self.model.get_output_embeddings()
                input_embeddings.w1.requires_grad_(False)
                input_embeddings.w2.requires_grad_(True)
                output_embeddings.w1.requires_grad_(False)
                output_embeddings.w2.requires_grad_(True)
            elif self.freezing_strategy == 'exclude_all':
                self.freeze()
                self.model.get_input_embeddings().requires_grad_(True)
                self.model.get_output_embeddings().requires_grad_(True)

    def configure_optimizers(self):
        optimizer_config = {}
        parameters = [p for p in self.parameters() if p.requires_grad]

        optimizer_cls = AdamW

        if isinstance(self.strategy, DeepSpeedStrategy):
            optimizer_cls = DeepSpeedCPUAdam if self.strategy.is_using_offload else FusedAdam

        optimizer_config['optimizer'] = optimizer_cls(parameters, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        optimizer_config['lr_scheduler'] = {
            'scheduler': get_lr_scheduler(
                scheduler_type=self.lr_scheduler_type,
                optimizer=optimizer_config['optimizer'],
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                min_lr_factor=self.min_lr_factor,
            ),
            'interval': 'step',
        }

        return optimizer_config
    
    def on_train_start(self) -> None:
        self.model.gradient_checkpointing_enable()

        if self._load_from_checkpoint:
            if self.extend_tokens and self.freezing_strategy == 'exclude_new':
                tie_partially_frozen_weights(self.model)
            self.model.tie_weights()

    def training_step(self, batch: _BatchType, batch_idx: int):
        batch_size = batch['input_ids'].size(0)
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            use_cache=False,
        )
        loss = output.loss

        self.log('loss', loss, prog_bar=True, logger=False)

        self.log('Loss/Train/Step', loss)
        self.log('Loss/Train', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.train_perplexity.update(loss, batch['labels'], by_loss=True)
        self.batch_perplexity.update(loss, batch['labels'], by_loss=True)
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
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = output.loss
        self.val_perplexity.update(loss, batch['labels'], by_loss=True)
        
        self.log('loss', loss, prog_bar=True, logger=False, sync_dist=True)
        self.log('Loss/Val', loss, sync_dist=True)
        self.log('Perplexity/Val', self.val_perplexity, batch_size=batch_size, sync_dist=True)

    def on_validation_end(self) -> None:
        self.model.gradient_checkpointing_enable()
