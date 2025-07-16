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

