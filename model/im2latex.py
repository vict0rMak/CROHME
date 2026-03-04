import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .transformer_decoder import TransformerDecoder


class Im2Latex(nn.Module):
    def __init__(self, vocab_size, d_model=256, pad_idx=0):
        super().__init__()
        self.encoder = CNNEncoder(d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model, pad_idx=pad_idx)

    def forward(self, img, tgt):
        memory = self.encoder(img)
        out = self.decoder(tgt, memory)
        return out
