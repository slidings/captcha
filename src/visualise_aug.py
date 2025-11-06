# visualise_aug.py
# Run with python -m src.visualise_aug

# This script visualises the effect of data augmentation on sample images

import os
import random
import yaml
import cv2
import torch
import matplotlib.pyplot as plt

from src.dataset import CaptchaDataset


def show_image_grid(original, processed, title=""):
    """Show original and processed image side-by-side."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[0].axis("off")

    # processed: torch tensor (C,H,W)
    img_np = processed.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    axs[1].imshow(img_np)
    axs[1].set_title("Augmented + Padded")
    axs[1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    # --- Load YAML config ---
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # --- Build dataset using config ---
    ds = CaptchaDataset(
        root_dir=data_cfg["train_dir"],
        img_height=data_cfg["img_height"],
        max_width=data_cfg["max_width"],
        grayscale=data_cfg["grayscale"],
        is_train=True,  # enable augmentation
    )

    print(f"Loaded dataset with {len(ds)} samples from {data_cfg['train_dir']}")

    num_samples = 6
    for _ in range(num_samples):
        idx = random.randint(0, len(ds) - 1)
        path = ds.paths[idx]

        # Original image (for reference)
        orig = cv2.imread(path)
        if orig is None:
            print(f"Failed to read {path}")
            continue

        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        # Process through dataset pipeline
        sample = ds[idx]
        processed = sample["image"]
        label = sample["label_str"]

        show_image_grid(original=orig_rgb, processed=processed, title=f"{label} | idx={idx}")


if __name__ == "__main__":
    main()
