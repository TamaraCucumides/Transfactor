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

    def _encode_token_sequence(self, token_seq: List[Union[Any, Tuple[str, Tuple[Any], List[str], int]]]) -> List[int]:
        token_ids = []

        for i, token in enumerate(token_seq):
            if isinstance(token, tuple) and token[0] == "BLOCK":
                _, block_values, block_columns, _ = token
                ids = []

                for col, val in zip(block_columns, block_values):
                    key = val.item() if isinstance(val, (np.integer, np.floating)) else val
                    try:
                        val_id = self.vocab[col][key]
                    except KeyError:
                        print(f"[KeyError in block] Column: {col}, Key: {key}, Available: {self.vocab[col].keys()}")
                        raise
                    ids.append(val_id)

                block_token_id = sum(ids) // len(ids)  # or use hash(tuple(ids))
                token_ids.append(block_token_id)
            else:
                col_name = self.data.column_names[i]
                key = token.item() if isinstance(token, (np.integer, np.floating)) else token
                try:
                    token_ids.append(self.vocab[col_name][key])
                except KeyError:
                    print(f"[KeyError single token] Column: {col_name}, Key: {key}, Available: {self.vocab[col_name].keys()}")
                    raise

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
