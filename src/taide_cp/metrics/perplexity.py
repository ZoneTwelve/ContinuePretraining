from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

_TORCH_FLOAT_OR_DOUBLE = (torch.float32, torch.float64)


def  _check_type_consistency(preds: Tensor):
    if preds.dtype not in _TORCH_FLOAT_OR_DOUBLE:
        raise TypeError(
            f'Input tensor `preds` is expected to be of a type one of {_TORCH_FLOAT_OR_DOUBLE} but got {preds.dtype}.'
        )


def _check_shape_consistency(preds: Tensor, target: Tensor):
    if preds.dim() != 3:
        raise ValueError(
            'Input tensor `preds` is expected to have 3 dimensions, [batch_size, seq_len, vocab_size],'
            f' but got {len(preds.shape)}.'
        )
    if target.dim() != 2:
        raise ValueError(
            'Input tensor `target` is expected to have 2 dimensions, [batch_size, seq_len],'
            f' but got {len(target.shape)}.'
        )
    if preds.shape[:2] != target.shape:
        raise ValueError(
            'Input tensors `preds` and `target` are expected to have equaling first two dimensions,'
            f' [batch_size, seq_len], but got {preds.shape[:2]} and {target.shape}.'
        )


def _perplexity_update(preds: Tensor, target: Tensor, ignore_index: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    preds = preds.reshape(-1, preds.shape[-1])
    target = target.reshape(-1)

    total_log_probs = F.cross_entropy(preds, target, ignore_index=ignore_index, reduction='sum')
    count = torch.count_nonzero(target != ignore_index)

    return total_log_probs, count


class Perplexity(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    total_log_probs: Tensor
    count: Tensor

    def __init__(
        self,
        ignore_index: int | None = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__(**kwargs)

        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError(f'Argument `ignore_index` expected to either be `None` or an `int` but got {ignore_index}')
        self.ignore_index = ignore_index
        self.add_state('total_log_probs', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, logits_or_loss: Tensor, target: Tensor) -> None:        
        _check_type_consistency(logits_or_loss)
        
        if logits_or_loss.dim() == 0:
            count = torch.count_nonzero(target != self.ignore_index)
            total_log_probs = logits_or_loss * count
        else:
            _check_shape_consistency(logits_or_loss, target)
            total_log_probs, count = _perplexity_update(logits_or_loss, target, self.ignore_index)
        
        self.total_log_probs += total_log_probs
        self.count += count

    def compute(self):
        return torch.exp(self.total_log_probs / self.count)
