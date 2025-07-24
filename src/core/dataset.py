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

                print(f"[DEBUG] Encoding block token: values={block_values}, columns={block_columns}")

                ids = []

                for col, val in zip(block_columns, block_values):
                    # Normalize the value to a hashable native Python type
                    if isinstance(val, pd.Index):
                        key = val[0] if len(val) > 0 else None
                    elif isinstance(val, (np.integer, np.floating)):
                        key = val.item()
                    else:
                        key = val

                    try:
                        val_id = self.vocab[col][key]
                    except KeyError:
                        print(f"[KeyError in BLOCK] Column: {col}, Key: {key}, Vocab: {list(self.vocab[col].keys())[:10]}...")
                        raise

                    ids.append(val_id)

                block_token_id = sum(ids) // len(ids) if ids else 0
                token_ids.append(block_token_id)

            else:
                col_name = self.data.column_names[i]
                val = token
                # Normalize singleton value
                if isinstance(val, pd.Index):
                    key = val[0] if len(val) > 0 else None
                elif isinstance(val, (np.integer, np.floating)):
                    key = val.item()
                else:
                    key = val

                try:
                    token_ids.append(self.vocab[col_name][key])
                except KeyError:
                    print(f"[KeyError in singleton] Column: {col_name}, Key: {key}, Vocab: {list(self.vocab[col_name].keys())[:10]}...")
                    raise

        return token_ids



    def _get_block_id_sequence(self, token_seq: List[Union[Any, Tuple[str, Tuple[Any], Tuple[str], int]]]) -> List[int]:
        block_ids = []

        for token in token_seq:
            if isinstance(token, tuple) and token[0] == "BLOCK":
                _, _, _, block_id = token  # <-- updated to unpack 4 elements
                block_ids.append(block_id)
            else:
                block_ids.append(self.null_block_id)

        return block_ids

