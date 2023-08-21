import os
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional, Set, Tuple, Type, TypedDict

import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, wrap
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from transformers import (PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)
from transformers.modeling_utils import no_init_weights

from ...lightning import DeepSpeedStrategy, FSDPStrategy
from ...metrics import Perplexity
from ...utils import ContextManagers
from ..layers import PartiallyFrozenEmbedding, PartiallyFrozenLinear
from ..lightning_module import LightningModuleX
from ..lr_schedulers import get_lr_scheduler


class _BatchType(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


InitializingStrategy = Literal['mean', 'sample']
FreezingStrategy = Literal['exclude_new', 'exclude_all']


def tie_partially_frozen_weights(model: PreTrainedModel):
    if getattr(model.config, 'tie_word_embeddings'):
        input_embeddings: PartiallyFrozenEmbedding = model.get_input_embeddings()
        output_embeddings: PartiallyFrozenLinear = model.get_output_embeddings()
        output_embeddings.frozen_linear.weight = input_embeddings.frozen_embedding.weight
        output_embeddings.trainable_linear.weight = input_embeddings.trainable_embedding.weight

def extend_tokens(
    model: PreTrainedModel,
    new_num_tokens: int,
    initializing_strategy: InitializingStrategy | None = None,
    freezing_strategy: FreezingStrategy | None = None,
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

    if freezing_strategy == 'exclude_new':
        model.requires_grad_(False)
        model.set_input_embeddings(PartiallyFrozenEmbedding(input_embeddings, old_num_tokens))
        model.set_output_embeddings(PartiallyFrozenLinear(output_embeddings, old_num_tokens))
        tie_partially_frozen_weights(model)
    elif freezing_strategy == 'exclude_all':
        model.requires_grad_(False)
        model.get_input_embeddings().requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)

class LightningModuleForPreTraining(LightningModuleX):
    model_class: Optional[Type[PreTrainedModel]] = None
    config_class: Optional[Type[PretrainedConfig]] = None
    tokenizer_class: Optional[Type[PreTrainedTokenizerBase]] = None

    # DeepSpeed
    ds_model_class: Optional[Type[PreTrainedModel]] = None

    # Fully Sharded Data Parallel
    fsdp_auto_wrap_policy: Optional[Callable[[nn.Module, bool, int], bool]] = None
    fsdp_param_init_fn: Optional[Callable[[nn.Module, bool, int], bool]] = None
    fsdp_transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        extend_tokens: bool = False,
        initializing_strategy: InitializingStrategy | None = None,
        freezing_strategy: FreezingStrategy | None = None,
        max_length: int = 2048,
        lr: Optional[float] = 1e-5,
        # optimizer: Literal['adamw', 'lion'] = 'adamw',
        betas: Optional[Tuple[float, float]] = (0.9, 0.95),
        eps: float = 1e-8,
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

        self.max_length = max_length

        self.lr = lr
        self.betas = betas
        self.eps = eps
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
        
        kwargs = dict(
            low_cpu_mem_usage=True,
            config=self.config
        )
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            if self._load_from_checkpoint:
                context_managers.append(self.trainer.strategy.ds_init_context)
            kwargs['low_cpu_mem_usage'] = not self.trainer.strategy.zero_stage_3

        if not self._load_from_checkpoint:
            self.model: PreTrainedModel = self.model_class.from_pretrained(self.model_path, **kwargs)
        else:
            with ContextManagers(context_managers):
                self.model = self.model_class(self.config)

        if isinstance(self.strategy, FSDPStrategy):
            self.model = wrap(
                self.model,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=self.fsdp_transformer_layer_cls
                )
            )

        if self.extend_tokens:
            extend_tokens(
                self.model,
                len(self.tokenizer),
                initializing_strategy=self.initializing_strategy,
                freezing_strategy=self.freezing_strategy,
            )

        if isinstance(self.strategy, FSDPStrategy):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import (
                transformer_auto_wrap_policy, wrap)

            self.model: FSDP = wrap(
                self.model,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={self.fsdp_transformer_layer_cls}
                )
            )
        
        for m in self.modules():
            if isinstance(m, Metric):
                m.to(self.strategy.root_device)
        
    def configure_optimizers(self):
        optimizer_config = {}

        optimizer_cls = AdamW
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
            optimizer_cls = DeepSpeedCPUAdam if self.trainer.strategy.offload_optimizer else FusedAdam

        optimizer_config['optimizer'] = optimizer_cls(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )
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
    
    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:

        if isinstance(self.strategy, FSDPStrategy):
            assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
            self.model.clip_grad_norm_(gradient_clip_val)
        else:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm
            )

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

        if self._load_from_checkpoint:
            if self.extend_tokens and self.freezing_strategy == 'exclude_new':
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

    @staticmethod
    def convert_to_hf(
        checkpoint_path: str,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        dtype: torch.dtype = torch.half
    ) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        from accelerate import init_empty_weights

        from ...models import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from ...utils.deepspeed import \
            get_lightning_checkpoint_from_zero_checkpoint
        from ...utils.utilities import rsetattr
        
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

        hyper_parameters = checkpoint['hyper_parameters']
        model_path = model_path or hyper_parameters['model_path']
        tokenizer_path = tokenizer_path or hyper_parameters['tokenizer_path'] or model_path

        state_dict = patch_state_dict(checkpoint['state_dict'])
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        config = AutoConfig.from_pretrained(hyper_parameters['model_path'])

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        if hyper_parameters['extend_tokens']:
            model.resize_token_embeddings(len(tokenizer))

            if hyper_parameters['freezing_strategy'] is not None:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', low_cpu_mem_usage=True, config=config)

                if hyper_parameters['freezing_strategy'] == 'exclude_new':
                    state_dict = patch_partial_embeddings(model, state_dict)
    
        model = load_state_dict(model, state_dict)
        model.to(model.dtype)
        
        if 'pad_token' in tokenizer.special_tokens_map:
            model.config.pad_token_id = tokenizer.pad_token_id

        return tokenizer, model
