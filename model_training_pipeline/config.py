
# config.py
import torch

# ------------------------
# Paths
# ------------------------
DATASET_DIR = "/content/drive/MyDrive/All_data/datasets"
OUTPUT_MODEL = "/content/drive/MyDrive/All_data/results/unet_model_final.pth"
LOSS_CURVE = "/content/drive/MyDrive/All_data/results/loss_curve.png"
METRICS_FILE = "/content/drive/MyDrive/All_data/results/metrics.txt"

# ------------------------
# Training hyperparameters
# ------------------------
IMG_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 30
NUM_CLASSES = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

# ------------------------
# Device
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
