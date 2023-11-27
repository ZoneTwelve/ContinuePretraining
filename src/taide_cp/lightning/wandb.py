from argparse import Namespace
from typing import Any, Dict, Literal, Optional, Union
from lightning.fabric.utilities.types import _PATH

from lightning.pytorch.loggers.wandb import (WandbLogger, _convert_params,
                                             _sanitize_callable_params,
                                             rank_zero_only)
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from ..utils.config import ConfigBase

class EnhancedWandbLogger(WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        entity: str | None = None,
        tags: list | None = None,
        save_code: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            entity=entity,
            tags=tags,
            save_code=save_code,
            **kwargs
        )

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        
        for k, v in params.items():
            if isinstance(v, ConfigBase):
                params[k] = v.to_dict()

        self.experiment.config.update(params, allow_val_change=True)
