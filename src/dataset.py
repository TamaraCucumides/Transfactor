import torch
from torch.utils.data import Dataset
from typing import List, Union, Dict, Any
from data import BlockTabularData


class BlockTabularDataset(Dataset):
    def __init__(
        self,
        data: BlockTabularData,
        labels: List[Any],
        vocab: Dict[str, Dict[Any, int]],
        pad_token_id: int = 0,
        null_block_id: int = -1
    ):
        """
        Args:
            data: BlockTabularData instance
            labels: List of target labels (already encoded as ints)
            vocab: Dict mapping column names to vocabularies {value -> token_id}
            pad_token_id: Token ID used for padding
            null_block_id: The block ID for singleton/non-block tokens
        """
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.null_block_id = null_block_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_seq = self.data.get_token_sequence(idx)  # List[Union[value, List[value]]]
        label = self.labels[idx]

        token_ids = self._encode_token_sequence(token_seq)
        block_ids = self._get_block_id_sequence(token_seq, idx)

        return token_ids, block_ids, label

    def _encode_token_sequence(self, token_seq: List[Union[Any, List[Any]]]) -> List[int]:
        token_ids = []

        for i, token in enumerate(token_seq):
            if isinstance(token, list):  # block token
                ids = []
                for j, val in enumerate(token):
                    col_name = self.data.column_names[j]  # use block columns
                    val_id = self.vocab[col_name][val]
                    ids.append(val_id)
                block_id = sum(ids) // len(ids)  # placeholder pooling
                token_ids.append(block_id)
            else:  # single value token
                col_name = self.data.column_names[i]
                token_ids.append(self.vocab[col_name][token])

        return token_ids

    def _get_block_id_sequence(self, token_seq: List[Union[Any, List[Any]]], row_idx: int) -> List[int]:
        """
        Returns a list of block IDs per token.
        For singleton tokens, return self.null_block_id.
        """
        block_ids = []
        row_blocks = self.data.row_blocks[row_idx]  # List of (cols, block_id)
        block_lookup = {tuple(cols): block_id for cols, block_id in row_blocks}

        for token in token_seq:
            if isinstance(token, list):  # it's a block
                # find matching block_id by matching values
                for cols, block_id in row_blocks:
                    if len(cols) == len(token):
                        block_ids.append(block_id)
                        break
                else:
                    block_ids.append(self.null_block_id)
            else:
                block_ids.append(self.null_block_id)
        return block_ids