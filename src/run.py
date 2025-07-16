import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.transfactor import Transfactor
from data import BlockTabularData
from dataset import BlockTabularDataset
from utils import build_vocab_from_df, encode_block_definitions, collate_fn_pad

from sklearn.preprocessing import LabelEncoder
import pandas as pd

# === 1. Create toy data ===
df = pd.DataFrame([
    ["A", "A", "X", "Y"],
    ["A", "A", "B", "Y"],
    ["C", "D", "B", "Z"],
    ["A", "A", "X", "Y"],
    ["C", "D", "X", "Z"],
], columns=["c1", "c2", "c3", "c4"])

target = pd.Series(["yes", "no", "yes", "yes", "no"])

# === 2. Encode features and labels ===
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_le = LabelEncoder()
target_labels = target_le.fit_transform(target)

# === 3. Build vocab ===
vocab, _ = build_vocab_from_df(df)

# === 4. Define blocks and encode them ===
raw_block_defs = [{"block_id": 0, "columns": ["c1", "c2"], "values": ["A", "A"]}]
block_defs = encode_block_definitions(raw_block_defs, label_encoders)

# === 5. Dataset and DataLoader ===
block_data = BlockTabularData(df, block_defs)
dataset = BlockTabularDataset(
    data=block_data,
    labels=target_labels,
    vocab=vocab,
    pad_token_id=0,
    null_block_id=-1
)

num_blocks = 1
pad_block_id = num_blocks  # same as null_block_id in model

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda batch: collate_fn_pad(batch, pad_token_id=0, pad_block_id=pad_block_id)
)


# === 6. Model, optimizer, loss ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transfactor(
    vocab_size=max(v for col in vocab.values() for v in col.values()) + 1,
    num_blocks=1,
    d_model=32,
    nhead=4,
    num_layers=2,
    num_classes=2,
    max_seq_len=10
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === 7. Training loop ===
for epoch in range(10):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        token_ids = batch["tokens"].to(device)
        block_ids = batch["block_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(token_ids, block_ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / total
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")






