#train_transfactor.py

from model.transfactor import Transfactor
from core.data import BlockTabularData
from core.dataset import BlockTabularDataset
from core.utils import build_vocab_from_df, encode_block_definitions, collate_fn_pad, build_vocab_from_label_encoders
from core.block_finding import fast_blocks_numpy
from core.utils import SafeLabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Utility Functions ===

def encode_dataframe(df, existing_encoders=None):
    df = df.copy()
    label_encoders = existing_encoders or {}

    for col in df.columns:
        le = label_encoders.get(col, SafeLabelEncoder())
        df[col] = le.fit_transform(df[col]) if col not in label_encoders else le.transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


def encode_target(target, existing_encoder=None):
    le = existing_encoder or LabelEncoder()
    labels = le.fit_transform(target) if existing_encoder is None else le.transform(target)
    return labels, le

def prepare_vocab_and_blocks(df, raw_block_defs, label_encoders):
    encoded_blocks = []
    print(f"[DEBUG] Columns in df: {list(df.columns)}")
    print(f"[DEBUG] Number of raw blocks: {len(raw_block_defs)}")

    for block in raw_block_defs:
        cols = block["columns"]
        raw_vals = block["values"]
        print(f"\n[DEBUG] Processing block_id={block['block_id']}, cols={cols}, raw_vals={raw_vals}")

        # Skip blocks where any column is missing from encoders
        if not all(col in label_encoders for col in cols):
            print(f"[SKIP] Missing label encoder for some columns: {cols}")
            continue

        try:
            encoded_vals = []
            for col, val in zip(cols, raw_vals):
                le = label_encoders[col]
                print(f"[ENCODING] Column={col}, Value={val}, Classes={le.classes_}")
                encoded_val = le.transform([val])[0]  # this may throw
                encoded_vals.append(encoded_val)

            encoded_blocks.append({
                "block_id": block["block_id"],
                "columns": cols,
                "values": encoded_vals
            })
            print(f"[OK] Encoded block {block['block_id']} → {encoded_vals}")

        except Exception as e:
            print(f"[ERROR] Failed to encode block {block['block_id']} with values {raw_vals}: {e}")
            continue  # Skip bad block

    if not encoded_blocks:
        raise ValueError("No valid blocks remaining after filtering and encoding.")

    vocab = build_vocab_from_label_encoders(label_encoders, restrict_to_cols=df.columns)
    print(f"[INFO] Vocab built with {len(vocab)} columns")

    return vocab, encoded_blocks


def prepare_dataset(df, block_defs, labels, vocab, pad_token_id=0, null_block_id=-1):
    encoded_cols = vocab.keys()
    print(f"[prepare_dataset] Using encoded columns: {list(encoded_cols)}")
    print(f"[prepare_dataset] First row:\n{df.iloc[0]}")
    print(f"[prepare_dataset] Creating BlockTabularData...")
    
    df = df[list(encoded_cols)].copy()
    block_data = BlockTabularData(df, block_defs)

    print(f"[prepare_dataset] BlockTabularData done. Creating dataset...")

    dataset = BlockTabularDataset(
        data=block_data,
        labels=labels,
        vocab=vocab,
        pad_token_id=pad_token_id,
        null_block_id=null_block_id
    )
    print(f"[prepare_dataset] Dataset done. Length: {len(dataset)}")

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

    print("[STEP] Starting block mining")
    raw_block_defs = fast_blocks_numpy(df_train, min_support=min_support, max_cols=max_cols)
    print(f"[INFO] Finished block mining with {len(raw_block_defs)} blocks")
    print("Blocks:", raw_block_defs)

    print("[STEP] Encoding training features")
    df_train_encoded, label_encoders = encode_dataframe(df_train)
    print("[STEP] Encoding validation features")
    df_val_encoded, _ = encode_dataframe(df_val, existing_encoders=label_encoders)
    print("[STEP] Encoding test features")
    df_test_encoded, _ = encode_dataframe(df_test, existing_encoders=label_encoders)

    print("[STEP] Encoding targets")
    target_labels_train, target_le = encode_target(target_train)
    target_labels_val, _ = encode_target(target_val, existing_encoder=target_le)
    target_labels_test, _ = encode_target(target_test, existing_encoder=target_le)

    print("[STEP] Preparing vocab and encoded blocks")
    vocab, block_defs = prepare_vocab_and_blocks(df_train_encoded, raw_block_defs, label_encoders)
    print(f"[INFO] Encoded {len(block_defs)} valid blocks")
    print(f"[DEBUG] Vocab keys: {list(vocab.keys())}")

    print("[STEP] Creating datasets")
    dataset_train = prepare_dataset(df_train_encoded, block_defs, target_labels_train, vocab)
    print("[INFO] Training dataset created")
    dataset_val = prepare_dataset(df_val_encoded, block_defs, target_labels_val, vocab)
    print("[INFO] Validation dataset created")
    dataset_test = prepare_dataset(df_test_encoded, block_defs, target_labels_test, vocab)
    print("[INFO] Test dataset created")

    num_blocks = len(block_defs)
    pad_block_id = num_blocks
    print(f"[INFO] Number of blocks: {num_blocks}, pad_block_id: {pad_block_id}")

    print("[STEP] Creating dataloaders")
    dataloader_train = create_dataloader(dataset_train, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    dataloader_val = create_dataloader(dataset_val, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    dataloader_test = create_dataloader(dataset_test, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    print("[INFO] Dataloaders ready")

    print("[STEP] Initializing model")
    vocab_size = max(v for col in vocab.values() for v in col.values()) + 1
    print(f"[DEBUG] Vocab size: {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transfactor(
        vocab_size=vocab_size,
        num_blocks=num_blocks,
        d_model=32,
        nhead=4,
        num_layers=2,
        num_classes=len(set(target_labels_train)),
        max_seq_len=100
    ).to(device)
    
    print("[INFO] Model initialized")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("[STEP] Starting training")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        train_model(model, dataloader_train, criterion, optimizer, device=model.device, label="Train")
        test_model(model, dataloader_val, criterion, device=model.device)

    print("\n[STEP] Final evaluation on test set")
    test_model(model, dataloader_test, criterion, device=model.device)

    return model, label_encoders, target_le, vocab

