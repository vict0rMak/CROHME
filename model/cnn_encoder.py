import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        backbone = models.resnet18(weights="DEFAULT")
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

    def forward(self, x):
        feat = self.cnn(x)          # (B, 512, H, W)
        feat = self.proj(feat)      # (B, d_model, H, W)
        B, C, H, W = feat.shape
        return feat.flatten(2).permute(2, 0, 1)
