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


# ==================================================
# Simple Levenshtein distance
# ==================================================
def levenshtein(a: str, b: str) -> int:
    dp = [list(range(len(b)+1))]
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


# Random from test set

def evaluate(model, loader, device, show_samples=10):
    model.eval()
    total_chars, total_edits, total_exact = 0, 0, 0
    n_samples = 0
    sample_buffer = []  # store all GT/pred pairs for random sampling
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", ncols=100):
            images = batch["images"].to(device)
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

    cer = total_edits / total_chars if total_chars > 0 else 1.0
    exact = total_exact / max(n_samples, 1)
    elapsed = time.time() - start_time

    print(f"\n✅ Evaluation complete in {elapsed:.1f}s")
    print(f"🔹 CER={cer:.3f}, Exact={exact:.3f} (N={n_samples})\n")

    # Randomly show a few predictions
    print("🔍 Random sample predictions:")
    for gt, pred in random.sample(sample_buffer, min(show_samples, len(sample_buffer))):
        mark = "✓" if pred == gt else "✗"
        print(f"GT: {gt:<10} | Pred: {pred:<10} {mark}")

    return cer, exact


# ==================================================
# Main
# ==================================================
def main():
    # ---- Load config ----
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Dataset (test) ----
    test_dir = os.path.join(cfg["data"]["train_dir"].replace("train", "test"))
    ds = CaptchaDataset(
        root_dir=test_dir,
        img_height=cfg["data"]["img_height"],
        max_width=cfg["data"]["max_width"],
        grayscale=cfg["data"]["grayscale"]
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=cfg["data"]["num_workers"],
                        pin_memory=True, collate_fn=collate_fn)

    # ---- Model ----
    num_classes = len(ITOCH)
    model = CRNN(
        num_classes=num_classes,
        cnn_out=cfg["model"]["cnn_out_channels"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    # ---- Load best checkpoint ----
    ckpt_path = os.path.join(cfg["log"]["ckpt_dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best.pt found in {cfg['log']['ckpt_dir']}")
    print(f"Loading model from {ckpt_path} ...")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    # ---- Evaluate ----
    evaluate(model, loader, device, show_samples=10)


if __name__ == "__main__":
    main()
