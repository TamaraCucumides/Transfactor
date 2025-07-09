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
    ):
        """
        Args:
            data: BlockTabularData instance
            labels: List of target labels (to be predicted); encoded as ints
            vocab: Dict mapping column names to vocabularies {value -> token_id}
            pad_token_id: Token ID used for padding
        """
        self.data = data
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vocab = vocab
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_seq = self.data.get_token_sequence(idx)  # List of values or list of values
        label = self.labels[idx]
        token_ids = self._encode_token_sequence(token_seq)
        return token_ids, label

    def _encode_token_sequence(self, token_seq: List[Union[Any, List[Any]]]) -> List[int]:
        """
        Converts each value (or list of values for a block) to a token ID.
        Aggregates block token IDs using simple mean pooling for now.
        """
        token_ids = []

        for token in token_seq:
            if isinstance(token, list):  # block token
                ids = []
                for i, val in enumerate(token):
                    col_name = self.data.column_names[i]
                    val_id = self.vocab[col_name][val]
                    ids.append(val_id)
                block_id = sum(ids) // len(ids)  # TODO:change after
                token_ids.append(block_id)
            else:  # single value token
                # Find corresponding column index
                col_index = token_seq.index(token)
                col_name = self.data.column_names[col_index]
                token_ids.append(self.vocab[col_name][token])

        return token_ids