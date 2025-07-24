import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Union, Dict, Any, Tuple
from core.data import BlockTabularData


class BlockTabularDataset(Dataset):
    def __init__(
        self,
        data: BlockTabularData,
        labels: List[Any],
        vocab: Dict[str, Dict[Any, int]],
        pad_token_id: int = 0,
        null_block_id: int = -1
    ):
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.null_block_id = null_block_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_seq = self.data.get_token_sequence(idx)
        label = self.labels[idx]

        token_ids = self._encode_token_sequence(token_seq)
        block_ids = self._get_block_id_sequence(token_seq)

        return token_ids, block_ids, label

    def _encode_token_sequence(self, token_seq: List[Union[Any, Tuple[str, Tuple[Any], int]]]) -> List[int]:
        token_ids = []

        for i, token in enumerate(token_seq):
            if isinstance(token, tuple) and token[0] == "BLOCK":
                _, block_values, _ = token
                ids = []

                for j, val in enumerate(block_values):
                    col_name = self.data.column_names[j]
                    key = val.item() if isinstance(val, (np.integer, np.floating)) else val  # ✅ Coerce to native type
                    val_id = self.vocab[col_name][key]
                    ids.append(val_id)
                block_token_id = sum(ids) // len(ids)  # or use hash(tuple(ids))
                token_ids.append(block_token_id)
            else:
                col_name = self.data.column_names[i]
                token_ids.append(self.vocab[col_name][token])

        return token_ids

    def _get_block_id_sequence(self, token_seq: List[Union[Any, Tuple[str, Tuple[Any], int]]]) -> List[int]:
        block_ids = []

        for token in token_seq:
            if isinstance(token, tuple) and token[0] == "BLOCK":
                _, _, block_id = token
                block_ids.append(block_id)
            else:
                block_ids.append(self.null_block_id)

        return block_ids
