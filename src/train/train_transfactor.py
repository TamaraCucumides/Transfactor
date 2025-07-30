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
import copy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score


# === Utility Functions ===

def evaluate_metrics(y_true, y_pred, y_prob=None, average='macro'):
    metrics = {
        "accuracy": (y_true == y_pred).mean(),
        "f1": f1_score(y_true, y_pred, average=average),
    }
    if y_prob is not None and len(set(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            metrics["roc_auc"] = 0.0  # fallback if class missing in batch
    else:
        metrics["roc_auc"] = None
    return metrics


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

    for block in raw_block_defs:
        cols = block["columns"]
        raw_vals = block["values"]

        if not all(col in label_encoders for col in cols):
            continue

        try:
            encoded_vals = [label_encoders[col].transform([val])[0] for col, val in zip(cols, raw_vals)]
            encoded_blocks.append({
                "block_id": block["block_id"],
                "columns": cols,
                "values": encoded_vals
            })
        except Exception as e:
            print(f"[ERROR] Failed to encode block {block['block_id']} with values {raw_vals}: {e}")
            continue

    if not encoded_blocks:
        print("No valid blocks remaining after filtering and encoding.")

    vocab = build_vocab_from_label_encoders(label_encoders, restrict_to_cols=df.columns)
    return vocab, encoded_blocks

def prepare_dataset(df, block_defs, labels, vocab, pad_token_id=0, null_block_id=-1):
    df = df[list(vocab.keys())].copy()
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
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    all_labels, all_preds, all_probs = [], [], []

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

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(logits.softmax(dim=1).detach().cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    all_probs = np.concatenate(all_probs, axis=0)

    print(f"[{label}] Loss={avg_loss:.4f}, Accuracy={acc:.4f}")
    return avg_loss, acc, all_labels, all_preds, all_probs

def test_model(model, dataloader, criterion, device):
    return train_model(model, dataloader, criterion, optimizer=None, device=device, label="Test", train=False)


# === Main Pipeline ===
def run_transfactor(df_train, target_train, df_val, target_val, df_test, target_test,
                 min_support=10, max_cols=3, batch_size=2, epochs=10,
                 selection_metric='loss'):

    raw_block_defs = fast_blocks_numpy(df_train, min_support=min_support, max_cols=max_cols)
    print(f"[INFO] Finished block mining with {len(raw_block_defs)} blocks")
    print(raw_block_defs)

    df_train_encoded, label_encoders = encode_dataframe(df_train)
    df_val_encoded, _ = encode_dataframe(df_val, existing_encoders=label_encoders)
    df_test_encoded, _ = encode_dataframe(df_test, existing_encoders=label_encoders)

    target_labels_train, target_le = encode_target(target_train)
    target_labels_val, _ = encode_target(target_val, existing_encoder=target_le)
    target_labels_test, _ = encode_target(target_test, existing_encoder=target_le)

    vocab, block_defs = prepare_vocab_and_blocks(df_train_encoded, raw_block_defs, label_encoders)
    print(f"[INFO] Encoded {len(block_defs)} valid blocks")

    dataset_train = prepare_dataset(df_train_encoded, block_defs, target_labels_train, vocab)
    dataset_val = prepare_dataset(df_val_encoded, block_defs, target_labels_val, vocab)
    dataset_test = prepare_dataset(df_test_encoded, block_defs, target_labels_test, vocab)

    num_blocks = len(block_defs)
    pad_block_id = num_blocks

    print("[STEP] Creating dataloaders")
    dataloader_train = create_dataloader(dataset_train, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    dataloader_val = create_dataloader(dataset_val, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    dataloader_test = create_dataloader(dataset_test, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    print("[INFO] Dataloaders ready")

    print("[STEP] Initializing model")
    vocab_size = max(v for col in vocab.values() for v in col.values()) + 1
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_state = None
    best_val_score = float("inf") if selection_metric == "loss" else -float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        train_model(model, dataloader_train, criterion, optimizer, device=device, label="Train", train=True)

        val_loss, val_acc, y_true, y_pred, y_prob = train_model(model, dataloader_val, criterion, optimizer=None, device=device, label="Val", train=False)
        metrics = evaluate_metrics(y_true, y_pred, y_prob)

        metric_value = {
            "loss": val_loss,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"] if metrics["roc_auc"] is not None else -1
        }[selection_metric]

        print(f"[VAL] All Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v is not None))


        is_better = (metric_value < best_val_score) if selection_metric == "loss" else (metric_value > best_val_score)

        if is_better:
            best_val_score = metric_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(f"[BEST] New best model at epoch {epoch+1} with {selection_metric}={metric_value:.4f}")


    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Loaded best model from epoch {best_epoch + 1}")

    print("\nFinal evaluation on test set:")
    _, _, y_true_test, y_pred_test, y_prob_test = test_model(model, dataloader_test, criterion, device=device)
    test_metrics = evaluate_metrics(y_true_test, y_pred_test, y_prob_test)

    print("[TEST METRICS]")
    for k, v in test_metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    return model, label_encoders, target_le, vocab

def run_transfactor_no_blocks(df_train, target_train, df_val, target_val,
                               df_test, target_test, batch_size=2, epochs=10):
    print(f"[INFO] Running Transfactor baseline with no blocks (1 token per value)")

    # Encode features
    df_train_encoded, label_encoders = encode_dataframe(df_train)
    df_val_encoded, _ = encode_dataframe(df_val, existing_encoders=label_encoders)
    df_test_encoded, _ = encode_dataframe(df_test, existing_encoders=label_encoders)

    # Encode targets
    target_labels_train, target_le = encode_target(target_train)
    target_labels_val, _ = encode_target(target_val, existing_encoder=target_le)
    target_labels_test, _ = encode_target(target_test, existing_encoder=target_le)

    # Build vocab only
    vocab = build_vocab_from_label_encoders(label_encoders, restrict_to_cols=df_train_encoded.columns)

    # === SKIP BLOCK MINING ===
    block_defs = []  # empty = treat all columns as singleton/default blocks

    dataset_train = prepare_dataset(df_train_encoded, block_defs, target_labels_train, vocab)
    dataset_val = prepare_dataset(df_val_encoded, block_defs, target_labels_val, vocab)
    dataset_test = prepare_dataset(df_test_encoded, block_defs, target_labels_test, vocab)

    pad_block_id = 0  # only null block id will be used
    vocab_size = max(v for col in vocab.values() for v in col.values()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader_train = create_dataloader(dataset_train, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    dataloader_val = create_dataloader(dataset_val, batch_size, pad_token_id=0, pad_block_id=pad_block_id)
    dataloader_test = create_dataloader(dataset_test, batch_size, pad_token_id=0, pad_block_id=pad_block_id)

    model = Transfactor(
        vocab_size=vocab_size,
        num_blocks=0,  # no real blocks
        d_model=32,
        nhead=4,
        num_layers=2,
        num_classes=len(set(target_labels_train)),
        max_seq_len=100
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}")
        train_model(model, dataloader_train, criterion, optimizer, device=device, label="Train", train=True)
        val_loss, val_acc = train_model(model, dataloader_val, criterion, optimizer=None, device=device, label="Val", train=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(f"[BEST] New best model at epoch {epoch+1} with val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Loaded best model from epoch {best_epoch + 1}")

    print("\nFinal evaluation on test set:")
    test_model(model, dataloader_test, criterion, device=device)

    return model, label_encoders, target_le, vocab