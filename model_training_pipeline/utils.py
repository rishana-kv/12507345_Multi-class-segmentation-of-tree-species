
# utils.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def colorize_mask(mask):
    """Convert integer mask to RGB"""
    colors = np.array([
        [0, 0, 0],    
        [255, 0, 0],    
        [0, 255, 0],    
        [0, 0, 255],    
        [255, 255, 0]    
    ], dtype=np.uint8)
    return colors[mask]

def save_loss_curve(train_history, val_history, path):
    plt.figure(figsize=(8,5))
    plt.plot(train_history, label="Train Loss")
    plt.plot(val_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid()
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.show()

def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        inter = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        ious.append(float(inter)/float(union) if union != 0 else np.nan)
    return ious

def compute_dice(pred, target, num_classes):
    dices = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        inter = 2*(pred_inds & target_inds).sum()
        total = pred_inds.sum() + target_inds.sum()
        dices.append(float(inter)/float(total) if total != 0 else np.nan)
    return dices
