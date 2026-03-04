import math

import torch
import torch.nn as nn


def causal_mask(size, device):
    return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.3,
        max_len=512,
        pad_idx=0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        seq_len = tgt.size(1)
        tgt_emb = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        tgt_emb = tgt_emb + self.pe[:seq_len].unsqueeze(0)
        tgt_emb = self.dropout(tgt_emb).permute(1, 0, 2)

        tgt_mask = causal_mask(seq_len, tgt.device)
        tgt_key_padding_mask = tgt.eq(self.pad_idx)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        out = out.permute(1, 0, 2)
        return self.fc(out)
