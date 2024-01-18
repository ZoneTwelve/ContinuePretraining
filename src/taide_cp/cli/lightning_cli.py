import logging
from typing import Any, Callable, Dict, Optional, Type, Union

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import (ArgsType, LightningArgumentParser,
                                   LightningCLI, SaveConfigCallback)

from taide_cp.utils.slurm import SLURM

from .save_config_callback import TaideCPSaveConfigCallback
from .training_routine_callback import TrainingRoutineCallback


class TaideCPLightningCLI(LightningCLI):
    def __init__(
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = None,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
    ) -> None:
        save_config_callback = TaideCPSaveConfigCallback if save_config_callback is None else save_config_callback
        save_config_kwargs = save_config_kwargs or {}
        save_config_kwargs = {'overwrite': True} | save_config_kwargs
        parser_kwargs = parser_kwargs or {}
        parser_kwargs = {'parser_mode': 'omegaconf'} | parser_kwargs

        super().__init__(
            model_class=model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
            save_config_kwargs=save_config_kwargs,
            trainer_class=trainer_class,
            trainer_defaults=trainer_defaults,
            seed_everything_default=seed_everything_default,
            parser_kwargs=parser_kwargs,
            subclass_mode_model=subclass_mode_model,
            subclass_mode_data=subclass_mode_data,
            args=args,
            run=run,
            auto_configure_optimizers=auto_configure_optimizers
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments('data.init_args.config.batch_size', 'trainer.strategy.init_args.logging_batch_size_per_gpu')
        parser.link_arguments('model.tokenizer', 'data.init_args.config.tokenizer', apply_on='instantiate')

        parser.add_argument('--float32_matmul_precision', type=Optional[str], default=None)
        parser.add_argument('--logging_level', type=Union[str, int], default=logging.INFO)
        parser.add_argument('--save_config', type=bool, default=True)

        parser.add_lightning_class_args(TrainingRoutineCallback, 'training_routine_callback')

    def _setup_extra_args(self):
        float32_matmul_precision = self._get(self.config, 'float32_matmul_precision')
        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)
        
        logging_level = self._get(self.config, 'logging_level')
        if isinstance(logging_level, str):
            logging_level = getattr(logging, logging_level.upper())
        
        logging.getLogger('taide_cp').setLevel(logging_level)
        logging.getLogger('lightning').setLevel(logging_level)

        if not self._get(self.config, 'save_config'):
            self.save_config_callback = False

    def before_instantiate_classes(self) -> None:
        self._setup_extra_args()

        config = self.config.get(self.config.get('subcommand'))

        if config is None:
            return

        extra_plugins = []
        if SLURM.is_slurm and SLURM.num_tasks > 1:
            extra_plugins += [{'class_path': 'SLURMEnvironment', 'init_args': {'auto_requeue': False}}]
        else:
            extra_plugins += [{'class_path': 'LightningEnvironment'}]
        config.trainer.plugins = config.trainer.plugins or []
        config.trainer.plugins += extra_plugins
