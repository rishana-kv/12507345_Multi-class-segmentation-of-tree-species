import os
import gc
import time
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import *
from dataset import TileDataset
from model import UNetSmall
from utils import save_loss_curve, colorize_mask, compute_iou, compute_dice

def main():
    start_time = time.time()
    print("Device:", DEVICE)

    # ----------------------
    # Datasets & Loaders
    # ----------------------
    train_set = TileDataset(f"{DATASET_DIR}/train/img", f"{DATASET_DIR}/train/mask", IMG_SIZE)
    val_set = TileDataset(f"{DATASET_DIR}/val/img", f"{DATASET_DIR}/val/mask", IMG_SIZE)
    test_img_dir = f"{DATASET_DIR}/test/img"
    test_mask_dir = f"{DATASET_DIR}/test/mask"

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ----------------------
    # Model, optimizer, loss
    # ----------------------
    model = UNetSmall(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    train_history, val_history = [], []

    # ----------------------
    # Training loop
    # ----------------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for img, mask in train_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del img, mask, out
            gc.collect()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                val_loss += criterion(model(img), mask).item()
                del img, mask
                gc.collect()

        train_avg = train_loss / len(train_loader)
        val_avg = val_loss / len(val_loader)
        train_history.append(train_avg)
        val_history.append(val_avg)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_avg:.4f} | Val: {val_avg:.4f}")

    # ----------------------
    # Save model & loss curve
    # ----------------------
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print("Model saved:", OUTPUT_MODEL)
    save_loss_curve(train_history, val_history, LOSS_CURVE)

    # ----------------------
    # Test predictions & metrics (first 5 images)
    # ----------------------
    if os.path.exists(test_img_dir):
        model.eval()
        test_imgs = sorted(os.listdir(test_img_dir))[:5]
        all_ious, all_dices = [], []

        species_colors = np.array([
            [0,0,0],        # Background
            [255,0,0],      # Beech
            [0,255,0],      # Pine
            [0,0,255],      # Birch
            [255,255,0]     # Spruce
        ], dtype=np.uint8)
        species_names = ["Background", "Beech", "Pine", "Birch", "Spruce"]

        os.makedirs("/content/drive/MyDrive/All_data/results", exist_ok=True)

        with open(METRICS_FILE, "w") as f:
            f.write("Filename\tIoU_per_class\tDice_per_class\n")

            # Create a figure for all 5 images
            fig, axs = plt.subplots(len(test_imgs), 3, figsize=(15, 5*len(test_imgs)))

            for row_idx, fname in enumerate(test_imgs):
                img_path = os.path.join(test_img_dir, fname)
                mask_path = os.path.join(test_mask_dir, fname)

                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                mask = Image.open(mask_path).resize((IMG_SIZE, IMG_SIZE))

                arr = np.array(img, dtype=np.float32)/255.0
                tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred = model(tensor)
                    pred_np = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
                mask_np = np.array(mask)

                # Compute metrics
                ious = compute_iou(pred_np, mask_np, NUM_CLASSES)
                dices = compute_dice(pred_np, mask_np, NUM_CLASSES)
                all_ious.append(ious)
                all_dices.append(dices)
                f.write(f"{fname}\t{ious}\t{dices}\n")

                # Plot each row
                axs[row_idx, 0].imshow(img)
                axs[row_idx, 0].set_title(f"Image: {fname}")
                axs[row_idx, 0].axis("off")

                axs[row_idx, 1].imshow(species_colors[mask_np])
                axs[row_idx, 1].set_title("Ground Truth")
                axs[row_idx, 1].axis("off")

                axs[row_idx, 2].imshow(species_colors[pred_np])
                axs[row_idx, 2].set_title("Prediction")
                axs[row_idx, 2].axis("off")

                # Save individual prediction
                pred_save_path = os.path.join("/content/drive/MyDrive/All_data/results", fname)
                Image.fromarray(species_colors[pred_np]).save(pred_save_path)
                print("Saved individual prediction:", pred_save_path)

            # Single legend for last column
            legend_elements = [Patch(facecolor=np.array(c)/255, edgecolor='k', label=name)
                               for c, name in zip(species_colors, species_names)]
            axs[-1, 2].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            combined_path = "/content/drive/MyDrive/All_data/results/combined_test_images.png"
            plt.savefig(combined_path, dpi=150)
            plt.show()
            print("Saved combined figure:", combined_path)

        # Mean metrics
        mean_iou = np.nanmean(all_ious, axis=0)
        mean_dice = np.nanmean(all_dices, axis=0)
        print("Mean IoU per class:", mean_iou)
        print("Mean Dice per class:", mean_dice)

        with open(METRICS_FILE, "a") as f:
            f.write("\nMean IoU per class:\t{}\n".format(mean_iou.tolist()))
            f.write("Mean Dice per class:\t{}\n".format(mean_dice.tolist()))
        print("Metrics saved:", METRICS_FILE)

    print(f"Total training time: {(time.time()-start_time)/60:.2f} min")


if __name__ == "__main__":
    main()
