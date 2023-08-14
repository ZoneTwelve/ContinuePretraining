import torch
from torch import LongTensor, nn


class PartiallyFrozenEmbedding(nn.Module):
    @property
    def weight(self):
        return torch.cat([self.frozen_embedding.weight, self.trainable_embedding.weight], dim=0)

    def __init__(
        self,
        embeddings: nn.Embedding,
        pivot: int,
    ) -> None:
        super().__init__()

        self.num_embeddings = embeddings.num_embeddings
        self.embedding_dim = embeddings.embedding_dim
        self.padding_idx = embeddings.padding_idx
        self.max_norm = embeddings.max_norm
        self.norm_type = embeddings.norm_type
        self.scale_grad_by_freq = embeddings.scale_grad_by_freq
        self.sparse = embeddings.sparse

        self.pivot = pivot

        self.frozen_embedding = nn.Embedding.from_pretrained(
            embeddings.weight[:pivot],
            padding_idx=self.padding_idx if self.padding_idx is not None and self.padding_idx < pivot else None,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            freeze=True
        )

        self.trainable_embedding = nn.Embedding.from_pretrained(
            embeddings.weight[:pivot],
            padding_idx=self.padding_idx - self.pivot if self.padding_idx is not None and self.padding_idx >= pivot else None,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            freeze=False
        )
    
    def forward(self, x: LongTensor):
        mask = x < self.pivot
        e = torch.empty(*x.shape, self.embedding_dim, device=self.trainable_embedding.weight.device, dtype=self.trainable_embedding.weight.dtype)
        e[mask] = self.frozen_embedding(x[mask]).to(e.device, e.dtype)
        e[~mask] = self.trainable_embedding(x[~mask] - self.pivot)
        return e
