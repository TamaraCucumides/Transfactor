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
        max_seq_len: int = 50,
        mix_type: str = "concat"  # options: 'sum', 'concat', 'row', 'col'
    ):
        super().__init__()

        assert mix_type in {"sum", "concat", "row", "col"}, "Invalid mix_type"

        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.null_block_id = num_blocks
        self.mix_type = mix_type
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.block_id_embedding = nn.Embedding(num_blocks + 1, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer_row = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.row_transformer = nn.TransformerEncoder(encoder_layer_row, num_layers=num_layers)

        encoder_layer_col = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.col_transformer = nn.TransformerEncoder(encoder_layer_col, num_layers=num_layers)

        if mix_type == "concat":
            classifier_input_dim = d_model * 2
        else:
            classifier_input_dim = d_model

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, token_ids, block_ids, attention_mask):
        B, T = token_ids.shape

        block_ids = block_ids.masked_fill(block_ids == -1, self.null_block_id)

        if torch.any(block_ids >= self.block_id_embedding.num_embeddings):
            raise ValueError(
                f"block_ids contain value(s) >= {self.block_id_embedding.num_embeddings}. "
                f"Max block_id in batch: {block_ids.max().item()}"
            )

        x = self.token_embedding(token_ids)
        x += self.block_id_embedding(block_ids)
        x += self.position_embedding[:, :T, :]

        row_out = self.row_transformer(x, src_key_padding_mask=~attention_mask)

        col_out = self.col_transformer(row_out.transpose(0, 1)).transpose(0, 1)

        # Mix strategies
        if self.mix_type == "sum":
            combined = row_out + col_out
        elif self.mix_type == "concat":
            combined = torch.cat([row_out, col_out], dim=-1)
        elif self.mix_type == "row":
            combined = row_out
        elif self.mix_type == "col":
            combined = col_out
        else:
            raise ValueError(f"Unsupported mix_type: {self.mix_type}")

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1)
        summed = (combined * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        pooled = summed / count

        return self.classifier(pooled)

    @property
    def device(self):
        return next(self.parameters()).device

