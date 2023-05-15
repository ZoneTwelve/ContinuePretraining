from typing import Dict, Optional

import torch
from torch import Generator, Tensor
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        num_items: int,
        generator: Optional[Generator] = None,
    ) -> None:
        super().__init__()

        self.data = torch.randint(
            0,
            vocab_size,
            (num_items, sequence_length),
            generator=generator
        )
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        input_ids = self.data[index]
        return dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
        )
