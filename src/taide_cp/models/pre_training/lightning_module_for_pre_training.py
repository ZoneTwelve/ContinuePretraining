import os
from functools import partial
from types import MethodType
from typing import Callable, Optional, Set, Type, TypedDict

import lightning as L
import torch
from peft import LoraConfig, PeftModelForCausalLM, prepare_model_for_kbit_training
from torch import nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.modeling_utils import no_init_weights

from taide_cp.patchers import Patcher

from ...lightning import DeepSpeedStrategy, FSDPStrategy
from ...metrics import Perplexity
from ...patchers import Patcher
from ...utils import ContextManagers
from ..hf import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ..lr_schedulers import get_lr_scheduler
from ..modules import PartiallyFrozenEmbedding, PartiallyFrozenLinear
from .pre_training_config import *


__all__ = [
    'LightningModuleForPreTraining',
    'LightningModuleForPreTrainingWithLoRA'
]

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

def extend_tokens(
    model: PreTrainedModel,
    new_num_tokens: int,
    initializing_strategy: InitializingStrategy,
    training_strategy: TrainingStrategy,
):
    import deepspeed
    from transformers.deepspeed import is_deepspeed_zero3_enabled

    old_num_tokens = model.config.vocab_size
    model.resize_token_embeddings(new_num_tokens)
    input_embeddings: nn.Embedding = model.get_input_embeddings()
    output_embeddings: nn.Linear = model.get_output_embeddings()

    with deepspeed.zero.GatheredParameters(
        [input_embeddings.weight, output_embeddings.weight],
        modifier_rank=0,
        enabled=is_deepspeed_zero3_enabled()
    ):
        if initializing_strategy == InitializingStrategy.MEAN:
            input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight.data[:old_num_tokens].mean(0))
            output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight.data[:old_num_tokens].mean(0))
        elif initializing_strategy == InitializingStrategy.SAMPLE:
            mask = torch.randperm(old_num_tokens, device=model.device)[:new_num_tokens - old_num_tokens]
            input_embeddings.weight.data[old_num_tokens:].copy_(input_embeddings.weight.data[mask])
            output_embeddings.weight.data[old_num_tokens:].copy_(output_embeddings.weight.data[mask])

    if training_strategy == TrainingStrategy.NEW_TOKENS_ONLY:
        model.requires_grad_(False)
        model.set_input_embeddings(PartiallyFrozenEmbedding(input_embeddings, old_num_tokens))
        model.set_output_embeddings(PartiallyFrozenLinear(output_embeddings, old_num_tokens))
        tie_partially_frozen_weights(model)
    elif training_strategy == TrainingStrategy.ALL_TOKENS:
        model.requires_grad_(False)
        model.get_input_embeddings().requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)
    else:
        model.requires_grad_(True)


def _fsdp_configure_gradient_clipping(
    self: "LightningModuleForPreTraining",
    optimizer: Optimizer,
    gradient_clip_val: int | float | None = None,
    gradient_clip_algorithm: str | None = None
) -> None:
    assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
    self.model.clip_grad_norm_(gradient_clip_val)
    

class LightningModuleForPreTraining(L.LightningModule):
    model_class = AutoModelForCausalLM
    model_config_class = AutoConfig
    tokenizer_class = AutoTokenizer

    model: PreTrainedModel
    model_config: PretrainedConfig
    tokenizer: PreTrainedTokenizerBase

    # Fully Sharded Data Parallel
    fsdp_param_init_fn: Optional[Callable[[nn.Module, bool, int], bool]] = None
    fsdp_transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None

    def __init__(self, config: PreTrainingConfig, patchers: Patcher | list[Patcher] | None = None) -> None:       
        super().__init__()

        self.save_hyperparameters(config.asdict())

        self.config = config
        self.model_config = self.model_config_class.from_pretrained(self.config.model_path, revision=config.revision)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.config.tokenizer_path, revision=config.revision)

        assert self.config.extend_vocab or len(self.tokenizer) <= self.model_config.vocab_size
        
        if patchers is None:
            self.patchers = []
        elif isinstance(patchers, Patcher): 
            self.patchers = [patchers]
        else:
            self.patchers = patchers
        
        self.batch_perplexity = Perplexity(ignore_index=-100)
        self.train_perplexity = Perplexity(ignore_index=-100)
        self.val_perplexity = Perplexity(ignore_index=-100)

    def _move_metrics_to_device(self):
        for m in self.modules():
            if isinstance(m, Metric):
                m.to(self.trainer.strategy.root_device)

    def _get_model_kwargs(self):
        kwargs = dict(
            low_cpu_mem_usage=True,
            config=self.model_config,
            revision=self.config.revision
        )
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            kwargs['low_cpu_mem_usage'] = not self.trainer.strategy.zero_stage_3
        return kwargs

    def configure_sharded_model(self) -> None:
        context_managers = []
        
        load_from_checkpoint = self.trainer.ckpt_path is not None
        if load_from_checkpoint:
            context_managers.append(no_init_weights())    
        
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            if load_from_checkpoint:
                context_managers.append(self.trainer.strategy.ds_init_context)

        if not load_from_checkpoint:
            self.model: PreTrainedModel = self.model_class.from_pretrained(self.config.model_path, **self._get_model_kwargs())
        else:
            with ContextManagers(context_managers):
                self.model = self.model_class(self.model_config)

        for patcher in self.patchers:
            patcher(self.model)

        if self.config.extend_vocab:
            extend_tokens(
                self.model,
                len(self.tokenizer),
                initializing_strategy=self.config.extend_vocab.initializing_strategy,
                training_strategy=self.config.extend_vocab.training_strategy,
            )

        if isinstance(self.trainer.strategy, FSDPStrategy):
            from torch.distributed.fsdp.wrap import (
                transformer_auto_wrap_policy, wrap)

            self.model = wrap(
                self.model,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={self.fsdp_transformer_layer_cls}
                )
            )
            self.configure_gradient_clipping = MethodType(_fsdp_configure_gradient_clipping, self)

        self._move_metrics_to_device()
        
    def configure_optimizers(self):
        optimizer_config = {}

        optimizer_cls = AdamW
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
            optimizer_cls = DeepSpeedCPUAdam if self.trainer.strategy.offload_optimizer else FusedAdam

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

    @classmethod
    def convert_to_hf(
        cls,
        checkpoint_path: str,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        dtype: torch.dtype = torch.half
    ) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        from accelerate import init_empty_weights

        from ...utils import rsetattr
        from ...utils.deepspeed import \
            get_lightning_checkpoint_from_zero_checkpoint
        
        def patch_state_dict(state_dict: dict[str, torch.Tensor]):
            return {k.removeprefix('model.'): v for k, v in state_dict.items()}

        def patch_partial_embeddings(model: PreTrainedModel, state_dict: dict[str, torch.Tensor]):
            new_state_dict: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.endswith('.w1') or k.endswith('frozen_embedding.weight') or k.endswith('frozen_linear.weight'):
                    continue

                if k.endswith('.w2') or k.endswith('trainable_embedding.weight') or k.endswith('trainable_linear.weight'):
                    new_key = k.replace('.w2', '.weight').replace('.trainable_embedding', '').replace('.trainable_linear', '')
                    old_embeddings = model.get_parameter(new_key).data
                    num_old_embeddings = old_embeddings.size(0)
                    num_embeddings = num_old_embeddings + v.size(0)
                    embedding_size = v.size(1)
                    x = torch.empty(num_embeddings, embedding_size)
                    x[:num_old_embeddings].copy_(old_embeddings)
                    x[num_old_embeddings:].copy_(v)
                    k, v = new_key, x
                new_state_dict[k] = v
            return new_state_dict

        def load_state_dict(model: PreTrainedModel, state_dict: dict[str, torch.Tensor]):
            for k, v in state_dict.items():
                rsetattr(model, k, torch.nn.Parameter(v))
            model.tie_weights()
            return model

        if os.path.isdir(checkpoint_path):
            checkpoint = get_lightning_checkpoint_from_zero_checkpoint(checkpoint_path, dtype=dtype)
        else:
            checkpoint = torch.load(checkpoint_path, 'cpu')

        hparams = checkpoint['hyper_parameters']
        model_path = model_path or hparams['model_path']
        tokenizer_path = tokenizer_path or hparams['tokenizer_path'] or model_path

        state_dict = patch_state_dict(checkpoint['state_dict'])
        
        tokenizer = cls.tokenizer_class.from_pretrained(tokenizer_path)
        config = cls.model_config_class.from_pretrained(model_path)

        with init_empty_weights():
            model: PreTrainedModel = cls.model_class.from_config(config)

        if hparams['extend_vocab']:
            model.resize_token_embeddings(len(tokenizer))

            if hparams['extend_vocab']['training_strategy'] != TrainingStrategy.FULL:
                model = cls.model_class.from_pretrained(model_path, torch_dtype='auto', low_cpu_mem_usage=True, config=config)

            if hparams['extend_vocab']['training_strategy'] != TrainingStrategy.NEW_TOKENS_ONLY:
                state_dict = patch_partial_embeddings(model, state_dict)
    
        model = load_state_dict(model, state_dict)
        model.to(model.dtype)
        
        if 'pad_token' in tokenizer.special_tokens_map:
            model.config.pad_token_id = tokenizer.pad_token_id

        return tokenizer, model


class LightningModuleForPreTrainingWithLoRA(LightningModuleForPreTraining):
    config: PreTrainingWithLoRAConfig

    def __init__(self, config: PreTrainingWithLoRAConfig, patchers: Patcher | list[Patcher] | None = None) -> None:
        super().__init__(config, patchers)

        if config.quantization_config is not None:
            if not hasattr(self.model_config, 'quantization_config'):
                self.model_config.quantization_config = {}
            self.model_config.quantization_config |= config.quantization_config

    def _get_model_kwargs(self):
        kwargs = super()._get_model_kwargs()
        kwargs['torch_dtype'] = 'auto'
        return kwargs

    def configure_sharded_model(self) -> None:
        self.model: PreTrainedModel = self.model_class.from_pretrained(self.config.model_path, **self._get_model_kwargs())
        self.model = prepare_model_for_kbit_training(self.model)
        if self.config.lora_config:
            self.model = PeftModelForCausalLM(self.model, self.config.lora_config)
        else:
            self.model = PeftModelForCausalLM.from_pretrained(self.model, self.config.lora_path, is_trainable=True)

        self.model.base_model.enable_input_require_grads()

        for patcher in self.patchers:
            patcher(self.model.base_model)

        self._move_metrics_to_device()

    @classmethod
    def convert_to_hf(
        cls,
        checkpoint_path: str,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        dtype: torch.dtype = torch.half,
    ) -> tuple[PreTrainedTokenizerBase, PeftModelForCausalLM]:
        from ...utils.deepspeed import \
            get_lightning_checkpoint_from_zero_checkpoint
        
        def patch_state_dict(state_dict: dict[str, torch.Tensor]):
            return {k.removeprefix('model.'): v for k, v in state_dict.items()}

        if os.path.isdir(checkpoint_path):
            checkpoint = get_lightning_checkpoint_from_zero_checkpoint(checkpoint_path, dtype=dtype)
        else:
            checkpoint = torch.load(checkpoint_path, 'cpu')

        hparams = checkpoint['hyper_parameters']
        model_path = model_path or hparams['model_path']
        tokenizer_path = tokenizer_path or hparams['tokenizer_path'] or model_path

        tokenizer = cls.tokenizer_class.from_pretrained(tokenizer_path)
        base_model = cls.model_class.from_pretrained(
            model_path,
            torch_dtype='auto',
            low_cpu_mem_usage=True
        )

        peft_model = PeftModelForCausalLM(base_model, LoraConfig(**hparams['lora_config']))
        state_dict = patch_state_dict(checkpoint['state_dict'])
        incompatible_keys = peft_model.load_state_dict(state_dict, strict=False)
        return tokenizer, peft_model
