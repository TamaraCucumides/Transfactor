#train_transfactor.py

from model.transfactor import Transfactor
from core.data import BlockTabularData
from core.dataset import BlockTabularDataset
from core.utils import build_vocab_from_df, encode_block_definitions, collate_fn_pad
from core.block_finding import fast_blocks_numpy
from core.utils import SafeLabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Utility Functions ===

def encode_dataframe(df, existing_encoders=None):
    df = df.copy()
    label_encoders = existing_encoders or {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # skip numeric columns

        if col in label_encoders:
            le = label_encoders[col]
            df[col] = le.transform(df[col])
        else:
            le = SafeLabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    return df, label_encoders


def encode_target(target, existing_encoder=None):
    le = existing_encoder or LabelEncoder()
    labels = le.fit_transform(target) if existing_encoder is None else le.transform(target)
    return labels, le

def prepare_vocab_and_blocks(df, raw_block_defs, label_encoders):
    import numpy as np

    categorical_cols = set(label_encoders.keys())

    encoded_blocks = []
    used_cols = set()

    for block in raw_block_defs:
        cols = block["columns"]
        raw_vals = block["values"]

        # Only keep blocks whose columns are all categorical
        if not all(col in categorical_cols for col in cols):
            continue

        try:
            encoded_vals = []
            for col, val in zip(cols, raw_vals):
                le = label_encoders[col]
                val_str = str(val)
                if val_str not in le.classes_:
                    raise ValueError(f"Unseen value {val_str} in column {col}")
                encoded_val = le.transform([val_str])[0]
                encoded_vals.append(encoded_val)

            encoded_blocks.append({
                "block_id": block["block_id"],
                "columns": cols,
                "values": encoded_vals
            })

            used_cols.update(cols)
        except Exception:
            continue  # skip block if any error

    if not encoded_blocks:
        raise ValueError("No valid blocks remaining after filtering and encoding.")

    # ⚠️ Restrict DataFrame strictly to used categorical columns
    df_for_vocab = df[list(used_cols)].copy()

    vocab, _ = build_vocab_from_df(df_for_vocab)
    return vocab, encoded_blocks



def prepare_dataset(df, block_defs, labels, vocab, pad_token_id=0, null_block_id=-1):
    # Restrict df to columns present in vocab
    encoded_cols = vocab.keys()
    df = df[list(encoded_cols)].copy()

    block_data = BlockTabularData(df, block_defs)
    dataset = BlockTabularDataset(
        data=block_data,
        labels=labels,
        vocab=vocab,
        pad_token_id=pad_token_id,
        null_block_id=null_block_id
    )
    return dataset

def create_dataloader(dataset, batch_size, pad_token_id, pad_block_id):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_pad(batch, pad_token_id=pad_token_id, pad_block_id=pad_block_id)
    )

def train_model(model, dataloader, criterion, optimizer, device, label="Train", train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in dataloader:
            token_ids = batch["tokens"].to(device)
            block_ids = batch["block_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if train:
                optimizer.zero_grad()

            logits = model(token_ids, block_ids, mask)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / total
    print(f"[{label}] Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

def test_model(model, dataloader, criterion, device):
    train_model(model, dataloader, criterion, optimizer=None, device=device, label="Test", train=False)


# Main

def run_pipeline(df_train, target_train, df_val, target_val, df_test, target_test,
                 min_support=10, max_cols=3, batch_size=2, epochs=10):

    # Block mining only on training data
    raw_block_defs = fast_blocks_numpy(df_train, min_support=min_support, max_cols=max_cols)
    print("Blocks:", raw_block_defs)

    # Encode features
    df_train_encoded, label_encoders = encode_dataframe(df_train)
    df_val_encoded, _ = encode_dataframe(df_val, existing_encoders=label_encoders)
    df_test_encoded, _ = encode_dataframe(df_test, existing_encoders=label_encoders)

    # Encode targets
    target_labels_train, target_le = encode_target(target_train)
    target_labels_val, _ = encode_target(target_val, existing_encoder=target_le)
    target_labels_test, _ = encode_target(target_test, existing_encoder=target_le)

    # Build vocab and encode blocks
    vocab, block_defs = prepare_vocab_and_blocks(df_train_encoded, raw_block_defs, label_encoders)

    # Datasets
    dataset_train = prepare_dataset(df_train_encoded, block_defs, target_labels_train, vocab)
    dataset_val = prepare_dataset(df_val_encoded, block_defs, target_labels_val, vocab)
    dataset_test = prepare_dataset(df_test_encoded, block_defs, target_labels_test, vocab)

    # Dataloaders
    #num_blocks = len(raw_block_defs)
    num_blocks = len(block_defs)
    pad_block_id = num_blocks

    dataloader_train = create_dataloader(dataset_train, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    dataloader_val = create_dataloader(dataset_val, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    dataloader_test = create_dataloader(dataset_test, batch_size, pad_token_id=0, pad_block_id=pad_block_id)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transfactor(
        vocab_size=max(v for col in vocab.values() for v in col.values()) + 1,
        num_blocks=num_blocks,
        d_model=32,
        nhead=4,
        num_layers=2,
        num_classes=len(set(target_labels_train)),
        max_seq_len=100
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        train_model(model, dataloader_train, criterion, optimizer, device, label="Train")
        test_model(model, dataloader_val, criterion, device)

    print("\nFinal evaluation on test set:")
    test_model(model, dataloader_test, criterion, device)

    return model, label_encoders, target_le, vocab

