from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Perplexity
from torchmetrics.functional.text.perplexity import \
    _check_shape_and_type_consistency


def _perplexity_update(preds: Tensor, target: Tensor, ignore_index: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    _check_shape_and_type_consistency(preds, target)

    preds = preds.reshape(-1, preds.shape[-1])
    target = target.reshape(-1)

    total_log_probs = F.cross_entropy(preds, target, ignore_index=ignore_index, reduction='sum')
    count = torch.count_nonzero(target != ignore_index)

    return total_log_probs, count


class Perplexity(Perplexity):
    def update(self, preds_or_loss: Tensor, target: Tensor, by_loss: bool = False) -> None:
        if by_loss:
            preds_or_loss = preds_or_loss.float()
            if preds_or_loss.isnan() or preds_or_loss.isinf():
                return

            count = torch.count_nonzero(target != self.ignore_index)
            total_log_probs = preds_or_loss * count

        else:
            total_log_probs, count = _perplexity_update(preds_or_loss, target, self.ignore_index)
        
        self.total_log_probs += total_log_probs
        self.count += count
