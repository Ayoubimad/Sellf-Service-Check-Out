import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet50(num_classes=2000, weights=None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # 512

    def forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        return F.normalize(x, p=2, dim=1)  # L2 normalize

    def forward(self, anchor, positive, negative):
        anchor_emb = self.forward_one(anchor)
        positive_emb = self.forward_one(positive)
        negative_emb = self.forward_one(negative)
        return anchor_emb, positive_emb, negative_emb
