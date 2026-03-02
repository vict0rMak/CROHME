import torch
import torch.nn as nn

def causal_mask(size, device):
    return torch.triu(torch.ones(size, size, device=device), 1).bool()

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(512, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        tgt_emb = self.embed(tgt) + self.pos[:tgt.size(1)]
        tgt_emb = tgt_emb.permute(1, 0, 2)
        mask = causal_mask(tgt_emb.size(0), tgt.device)
        out = self.decoder(tgt_emb, memory, tgt_mask=mask)
        out = out.permute(1, 0, 2)
        return self.fc(out)
