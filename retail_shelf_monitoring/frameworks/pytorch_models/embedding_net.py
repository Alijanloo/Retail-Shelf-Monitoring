import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=256, target_shape=224):
        super().__init__()
        self.target_shape = target_shape
        # torchvision's mobilenet_v3_large
        self.backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        # features channel count: inspect by forwarding a dummy through features
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, target_shape, target_shape)
            feat = self.backbone.features(dummy)
            feat_channels = feat.shape[1]

        self.fc = nn.Linear(feat_channels, embedding_size)

    @property
    def input_shape(self):
        """Returns the expected input shape for the model (C, H, W)."""
        return (3, self.target_shape, self.target_shape)

    def forward(self, x):
        x = self.backbone.features(x)  # shape (B, C, H, W)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # shape (B, C)
        x = self.fc(x)  # (B, embedding_size)
        x = F.normalize(x, p=2, dim=1)  # l2-normalize per sample
        return x
