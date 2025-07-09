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

def pad_sequences(seqs: List[List[int]], pad_token_id: int = 0):
    max_len = max(len(seq) for seq in seqs)
    padded = []
    masks = []

    for seq in seqs:
        pad_len = max_len - len(seq)
        padded_seq = seq + [pad_token_id] * pad_len
        mask = [1] * len(seq) + [0] * pad_len
        padded.append(padded_seq)
        masks.append(mask)

    return padded, masks


def collate_fn_pad(batch, pad_token_id: int = 0):
    token_seqs, labels = zip(*batch)
    padded, masks = pad_sequences(token_seqs, pad_token_id)
    return {
        "tokens": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long)
    }
