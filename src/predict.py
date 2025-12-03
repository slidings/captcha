# src/predict.py
import os
import time
import yaml
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CaptchaDataset, collate_fn
from src.model import CRNN
from src.vocab import BLANK, ITOCH
from src.decode import greedy_decode
from src.utils import set_seed
from src.transforms import add_edge_channel   # <-- ensure available


# ==================================================
# Levenshtein (same as train)
# ==================================================
def levenshtein(a: str, b: str) -> int:
    dp = [list(range(len(b) + 1))]
    dp += [[i+1] + [0]*len(b) for i in range(len(a))]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[-1][-1]


# ==================================================
# Evaluation Loop
# ==================================================
def evaluate(model, loader, device, show_samples=10):
    model.eval()
    total_chars = total_edits = total_exact = 0
    n_samples = 0

    sample_buffer = []
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=100):
            images = batch["images"].to(device)   # (B,C,H,W)
            gt_texts = batch["label_strs"]

            logits, _ = model(images)
            preds = greedy_decode(logits)

            for pred, gt in zip(preds, gt_texts):
                e = levenshtein(pred, gt)
                total_edits += e
                total_chars += max(len(gt), 1)
                total_exact += int(pred == gt)
                n_samples += 1
                sample_buffer.append((gt, pred))

    cer = total_edits / total_chars
    exact = total_exact / n_samples
    elapsed = time.time() - start_time

    print(f"\n=== Evaluation Complete ===")
    print(f"Time: {elapsed:.1f}s")
    print(f"CER:   {cer:.4f}")
    print(f"Exact: {exact:.4f}  (N={n_samples})\n")

    print("Random sample predictions:")
    for gt, pred in random.sample(sample_buffer, min(show_samples, len(sample_buffer))):
        ok = "✓" if pred == gt else "✗"
        print(f"GT: {gt:<12} | Pred: {pred:<12} {ok}")

    return cer, exact


# ==================================================
# Main
# ==================================================
def main():
    # ---- Load Config ----
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Dataset for Prediction ----
    test_dir = "data/test"
    print(f"Loading test images from {test_dir} ...")

    ds = CaptchaDataset(
        root_dir=test_dir,
        img_height=cfg["data"]["img_height"],
        max_width=cfg["data"]["max_width"],
        grayscale=cfg["data"]["grayscale"],
        is_train=False
    )

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn
    )

    # ---- Determine Correct Input Channels ----
    expected_channels = 1 if cfg["data"]["grayscale"] else 3

    # If using add_edge_channel, expected is 4
    if not cfg["data"]["grayscale"]:
        print("⚠ Detected color mode → checking if edge channel required...")
        expected_channels = 4  # RGB + Sobel

    print(f"Model expects {expected_channels} channels.")

    # ---- Build Model ----
    num_classes = len(ITOCH)
    model = CRNN(
        num_classes=num_classes,
        input_channels=expected_channels,
        img_height=cfg["data"]["img_height"],
        cnn_out=cfg["model"]["cnn_out_channels"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    # ---- Load Checkpoint ----
    ckpt_path = os.path.join(cfg["log"]["ckpt_dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ No best.pt found at {ckpt_path}")

    print(f"Loading weights from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    # ---- Evaluate ----
    evaluate(model, loader, device, show_samples=10)


if __name__ == "__main__":
    main()
