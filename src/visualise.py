# src/visualise.py
import os
import math
import random
import yaml
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.dataset import CaptchaDataset, collate_fn
from src.model import CRNN
from src.decode import greedy_decode
from src.utils import set_seed
from src.vocab import ITOCH
from src.transforms import add_edge_channel    # IMPORTANT


GRID_ROWS = 5
GRID_COLS = 5
N_SAMPLES = GRID_ROWS * GRID_COLS


# -------------------------------------------------------------
# Load Model EXACTLY like src/predict.py logic
# -------------------------------------------------------------
def load_model_correctly(cfg, device):
    grayscale = cfg["data"]["grayscale"]

    # Determine channels exactly the same way predict.py does
    if grayscale:
        input_channels = 1
    else:
        input_channels = 4    # RGB + Sobel channel

    print(f"[Model loader] Expecting input channels = {input_channels}")

    model = CRNN(
        num_classes=len(ITOCH),
        input_channels=input_channels,
        img_height=cfg["data"]["img_height"],
        cnn_out=cfg["model"]["cnn_out_channels"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    ckpt_path = os.path.join(cfg["log"]["ckpt_dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[!] Missing checkpoint: {ckpt_path}")

    print(f"[OK] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    model.eval()
    return model


# -------------------------------------------------------------
# Run inference on dataset and collect samples
# -------------------------------------------------------------
def gather_predictions(model, loader, device, max_samples):
    collected = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            gt_strs = batch["label_strs"]

            logits, _ = model(imgs)
            preds = greedy_decode(logits)

            for img_tensor, gt, pred in zip(batch["images"], gt_strs, preds):
                if len(collected) >= max_samples:
                    return collected

                # (C,H,W) → (H,W,C)
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)

                # Undo normalization (only needed for RGB part)
                # First 3 channels are normalized with mean=0.5 std=0.5
                if img_np.shape[2] >= 3:
                    img_np[:, :, :3] = (img_np[:, :, :3] * 0.5 + 0.5).clip(0, 1)

                collected.append((img_np, gt, pred))

    return collected


# -------------------------------------------------------------
# Build 16×16 grid
# -------------------------------------------------------------
def plot_grid(samples, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    fig.suptitle("CAPTCHA Predictions Grid", fontsize=16)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")

            if idx >= len(samples):
                continue

            img, gt, pred = samples[idx]
            correct = (gt == pred)

            ax.imshow(img[:, :, :3])   # only show RGB

            color = "green" if correct else "red"
            mark = "✓" if correct else "✗"

            ax.set_title(
                f"{mark} GT:{gt}\nPred:{pred}",
                fontsize=6,
                color=color,
                pad=1
            )

            idx += 1

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Dataset
    test_dir = "data/test"
    print("Loading test dataset from:", test_dir)

    ds = CaptchaDataset(
        root_dir=test_dir,
        img_height=cfg["data"]["img_height"],
        max_width=cfg["data"]["max_width"],
        grayscale=cfg["data"]["grayscale"],
        is_train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Load model correctly (with correct channel count)
    model = load_model_correctly(cfg, device)

    # Collect predictions
    samples = gather_predictions(model, loader, device, N_SAMPLES)
    print(f"[OK] Collected {len(samples)} samples")

    # Plot
    plot_grid(samples, GRID_ROWS, GRID_COLS)


if __name__ == "__main__":
    main()
