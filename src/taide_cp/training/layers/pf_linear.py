
import torch
from torch import Tensor, nn


class PartiallyFrozenLinear(nn.Module):
    @property
    def weight(self):
        return torch.cat([self.frozen_linear.weight, self.trainable_linear.weight], dim=0)
    
    @property
    def bias(self):
        if self.frozen_linear.bias:
            return torch.cat([self.frozen_linear.bias, self.trainable_linear.bias], dim=0)
        return None

    def __init__(self, linear: nn.Linear, pivot: int) -> None:
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.pivot = pivot

        self.frozen_linear = nn.Linear(
            self.in_features,
            self.pivot,
            bias=linear.bias is not None
        ).requires_grad_(False)

        self.trainable_linear = nn.Linear(
            self.in_features,
            self.out_features - self.pivot,
            bias=linear.bias is not None
        ).requires_grad_(True)

        self.frozen_linear.weight.data = linear.weight[:pivot]
        self.trainable_linear.weight.data = linear.weight[pivot:]

        if linear.bias:
            self.frozen_linear.bias.data = linear.bias[:pivot]
            self.trainable_linear.bias.data = linear.bias[pivot:]

    def forward(self, x: Tensor):
        x1 = self.frozen_linear(x)
        x2 = self.trainable_linear(x)
        return torch.cat([x1.to(x2.device, x2.dtype), x2], dim=-1)
