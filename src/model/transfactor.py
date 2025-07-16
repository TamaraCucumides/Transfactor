# Model definition

import torch
import torch.nn as nn
import torch.nn.functional as F

class Transfactor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_blocks: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        max_seq_len: int = 50
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.null_block_id = num_blocks  # Use this for singleton/unknown tokens

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.block_id_embedding = nn.Embedding(num_blocks + 1, d_model)  # +1 for null block
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Row-wise Transformer
        encoder_layer_row = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.row_transformer = nn.TransformerEncoder(encoder_layer_row, num_layers=num_layers)

        # Column-wise Transformer
        encoder_layer_col = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.col_transformer = nn.TransformerEncoder(encoder_layer_col, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, token_ids, block_ids, attention_mask):
        """
        Args:
            token_ids: [B, T]
            block_ids: [B, T]  (use self.null_block_id for singleton or padded)
            attention_mask: [B, T]
        Returns:
            logits: [B, num_classes]
        """
        B, T = token_ids.shape

        block_ids = block_ids.masked_fill(block_ids == -1, self.null_block_id)

        if torch.any(block_ids >= self.block_id_embedding.num_embeddings):
            raise ValueError(
                f"block_ids contain value(s) >= {self.block_id_embedding.num_embeddings}. "
                f"Max block_id in batch: {block_ids.max().item()}"
            )

        # Embeddings
        x = self.token_embedding(token_ids)  # [B, T, D]
        x += self.block_id_embedding(block_ids)  # [B, T, D]
        x += self.position_embedding[:, :T, :]  # [1, T, D]

        # Row-wise Transformer
        row_out = self.row_transformer(x, src_key_padding_mask=~attention_mask)  # [B, T, D]

        # Column-wise Transformer
        col_out = self.col_transformer(row_out.transpose(0, 1)).transpose(0, 1)  # [B, T, D]

        # Combine views
        combined = row_out + col_out  # [B, T, D]

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        summed = (combined * mask).sum(dim=1)  # [B, D]
        count = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = summed / count  # [B, D]

        return self.classifier(pooled)


#TODO: Add column_ids?
#TODO: Add [CLS] token?
