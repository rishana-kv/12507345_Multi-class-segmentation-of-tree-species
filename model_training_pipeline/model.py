
# model.py
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Two conv layers with ReLU."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class UNetSmall(nn.Module):
    """Small UNet for semantic segmentation."""
    def __init__(self, n_classes):
        super().__init__()
        self.down1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = ConvBlock(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        mid = self.bridge(p2)
        u2 = self.up2(mid)
        u2 = torch.cat([u2, c2], dim=1)
        c3 = self.dec2(u2)
        u1 = self.up1(c3)
        u1 = torch.cat([u1, c1], dim=1)
        c4 = self.dec1(u1)
        return self.out(c4)
