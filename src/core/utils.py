import torch
from typing import List, Union, Dict, Any
from sklearn.preprocessing import LabelEncoder

def build_vocab_from_df(df):
    vocab = {}
    label_encoders = {}

    for col in df.columns:
        le = LabelEncoder()
        le.fit(df[col])
        vocab[col] = {val: i for i, val in enumerate(le.classes_)}
        label_encoders[col] = le

    return vocab, label_encoders

def pad_sequences(seqs: List[List[int]], pad_value: int = 0):
    max_len = max(len(seq) for seq in seqs)
    padded = []
    masks = []
    for seq in seqs:
        pad_len = max_len - len(seq)
        padded_seq = seq + [pad_value] * pad_len
        mask = [1] * len(seq) + [0] * pad_len
        padded.append(padded_seq)
        masks.append(mask)
    return padded, masks


def collate_fn_pad(batch, pad_token_id: int = 0, pad_block_id: int = 99999):
    token_seqs, block_id_seqs, labels = zip(*batch)

    tokens, attn_mask = pad_sequences(token_seqs, pad_value=pad_token_id)
    block_ids, _ = pad_sequences(block_id_seqs, pad_value=pad_block_id)

    return {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "block_ids": torch.tensor(block_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

## Block utils

def encode_block_definitions(raw_block_defs, label_encoders):
    """
    Converts string-based block definitions to encoded integer values using label_encoders.

    Args:
        raw_block_defs: List of dicts like {"block_id": ..., "columns": [...], "values": [...]}
        label_encoders: Dict[col_name -> LabelEncoder]

    Returns:
        encoded_block_defs: List of same dicts, but with values encoded to ints
    """
    encoded_defs = []

    for block in raw_block_defs:
        cols = block["columns"]
        raw_vals = block["values"]
        encoded_vals = [
            label_encoders[col].transform([val])[0]
            for col, val in zip(cols, raw_vals)
        ]
        encoded_defs.append({
            "block_id": block["block_id"],
            "columns": cols,
            "values": encoded_vals
        })

    return encoded_defs

class SafeLabelEncoder:
    def __init__(self, unk_token="__UNK__"):
        self.le = LabelEncoder()
        self.unk_token = unk_token
        self.classes_ = None
        self._unk_index = None

    def fit(self, values):
        unique_vals = pd.Series(values).astype(str).unique().tolist()
        if self.unk_token in unique_vals:
            raise ValueError(f"'{self.unk_token}' already in data.")
        values_with_unk = unique_vals + [self.unk_token]
        self.le.fit(values_with_unk)
        self.classes_ = set(self.le.classes_)
        self._unk_index = self.le.transform([self.unk_token])[0]
        return self

    def transform(self, values):
        vals = pd.Series(values).astype(str)
        safe_vals = np.where(vals.isin(self.classes_), vals, self.unk_token)
        return self.le.transform(safe_vals)

    def fit_transform(self, values):
        return self.fit(values).transform(values)

    def inverse_transform(self, values):
        return self.le.inverse_transform(values)

    def get_classes(self):
        return self.le.classes_

    @property
    def unk_index(self):
        return self._unk_index
        

def build_vocab_from_label_encoders(label_encoders, restrict_to_cols=None):
    """
    Build a vocab that contains *every* integer id an encoder can emit,
    including the UNK id, even if that id never appears in df_train.
    """
    cols = restrict_to_cols or label_encoders.keys()
    vocab = {}
    for col in cols:
        le = label_encoders[col]
        # SafeLabelEncoder exposes .get_classes(); sklearn's LabelEncoder has .classes_
        classes = le.get_classes() if hasattr(le, "get_classes") else le.classes_
        n = len(classes)
        vocab[col] = {int(i): int(i) for i in range(n)}  # identity mapping
    return vocab


