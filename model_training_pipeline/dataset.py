
# dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

class TileDataset(Dataset):
    """Dataset for image tiles and masks."""

    def __init__(self, img_dir, mask_dir, size=256):
        self.size = size
        self.imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

        if len(self.imgs) != len(self.masks):
            raise ValueError(f"Mismatch: {len(self.imgs)} images vs {len(self.masks)} masks")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        img = torch.from_numpy(np.array(img, dtype=np.float32)/255.0).permute(2,0,1)

        mask = Image.open(self.masks[idx]).resize((self.size, self.size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img, mask

