from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataSet(Dataset):
    def __init__(self, data_fpath: str) -> None:
        self.data_fpath = data_fpath
        self.data = self.load_data()

    def load_data(self):
        return pd.read_csv(self.data_fpath)

    def __getitem__(self, idx: int) -> str:
        return self.data.iloc[idx]["code-switched"]

    def __len__(self) -> int:
        return len(self.data)


def collate_fn(
    batch: List[str], tokenizer: AutoTokenizer
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Tokenize the batch of sentences
    tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized
