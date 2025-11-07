# src/train.py
# Run with python -m src.train
import os
import yaml
import math
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import CaptchaDataset, collate_fn
from src.model import CRNN
from src.vocab import BLANK, ITOCH
from src.decode import greedy_decode
from src.utils import set_seed


# ==========================
# Levenshtein distance
# ==========================
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

def find_latest_checkpoint(dir_path):
    ckpts = [f for f in os.listdir(dir_path) if f.startswith("epoch") and f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.replace("epoch", "").replace(".pt", "")))
    return os.path.join(dir_path, ckpts[-1])

def find_best_checkpoint(dir_path):
    best_ckpt = os.path.join(dir_path, "best.pt")
    if os.path.isfile(best_ckpt):
        return best_ckpt
    return None

# ==========================
# Evaluate (with samples)
# ==========================
def evaluate(model, loader, device, show_samples=5):
    model.eval()
    total_chars, total_edits, total_exact = 0, 0, 0
    n_samples = 0
    shown = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            gt_texts = batch["label_strs"]

            logits, _ = model(images)
            preds = greedy_decode(logits)  # list[str]

            for pred, gt in zip(preds, gt_texts):
                e = levenshtein(pred, gt)
                total_edits += e
                total_chars += max(len(gt), 1)
                total_exact += int(pred == gt)
                n_samples += 1

                # show a few predictions
                if shown < show_samples:
                    mark = "✓" if pred == gt else "✗"
                    print(f"GT: {gt:<8} | Pred: {pred:<8} {mark}")
                    shown += 1

    cer = total_edits / total_chars if total_chars > 0 else 1.0
    exact = total_exact / max(n_samples, 1)
    elapsed = time.time() - start_time
    print(f"Validation done in {elapsed:.1f}s — CER={cer:.3f}, Exact={exact:.3f} (N={n_samples})")
    return cer, exact, n_samples


# ==========================
# Main training
# ==========================
def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        cfg["train"]["lr"] = float(cfg["train"]["lr"])
        cfg["train"]["weight_decay"] = float(cfg["train"]["weight_decay"])
        cfg["train"]["grad_clip"] = float(cfg["train"]["grad_clip"])

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Create separate datasets for train/val ---
    full_ds = CaptchaDataset(
        root_dir=cfg["data"]["train_dir"],
        img_height=cfg["data"]["img_height"],
        max_width=cfg["data"]["max_width"],
        grayscale=cfg["data"]["grayscale"],
        is_train=True
    )

    n = len(full_ds)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val

    # Split indices manually
    train_indices, val_indices = torch.utils.data.random_split(
        range(n),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )

    # Build two *independent* dataset instances
    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds = torch.utils.data.Subset(
        CaptchaDataset(
            root_dir=cfg["data"]["train_dir"],
            img_height=cfg["data"]["img_height"],
            max_width=cfg["data"]["max_width"],
            grayscale=cfg["data"]["grayscale"],
            is_train=False   # crucial: no augmentation
        ),
        val_indices
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn)

    is_grayscale = cfg["data"]["grayscale"]

    # --- Model
    num_classes = len(ITOCH)
    model = CRNN(
        num_classes=num_classes,
        input_channels=1 if is_grayscale else 3,
        img_height=cfg["data"]["img_height"],
        cnn_out=cfg["model"]["cnn_out_channels"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    ckpt_path = False

    # Continue from the latest one
    #ckpt_path = find_latest_checkpoint(cfg["log"]["ckpt_dir"])

    #Continue from the hand picked best one
    #ckpt_path = find_best_checkpoint(cfg["log"]["ckpt_dir"])
    
    if ckpt_path:
        print(f"Loading latest checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print("No checkpoint found — training from scratch.")


    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    
    # Scheduler based on time
    #scheduler = ExponentialLR(optimizer, gamma=cfg["train"].get("lr_decay", 1.0))

    # Scheduler based on training
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",      # because lower CER is better
        factor=0.5,      # reduce LR by half when plateauing
        patience=2,      # wait 2 epochs before reducing
    )

    ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    os.makedirs(cfg["log"]["ckpt_dir"], exist_ok=True)

    # --- Train
    global_step = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        start_epoch = time.time()

        progress = tqdm(enumerate(train_loader, 1),
                        total=len(train_loader),
                        desc=f"Epoch {epoch}/{cfg['train']['epochs']}",
                        ncols=100)

        for i, batch in progress:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            logits, input_lengths = model(images)

            # logits shape: [Time, Batch, Classes]
            T = logits.size(0)  # Time steps (sequence length)
            B = logits.size(1)  # Batch size
            
            # Safety check: target_lengths should not exceed input_lengths
            valid_mask = target_lengths <= input_lengths
            if not valid_mask.all():
                print(f"Skipping batch: target_lengths > input_lengths")
                continue
            
            loss = ctc_loss(
                logits.log_softmax(dim=-1), 
                targets, 
                input_lengths, 
                target_lengths
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            running += loss.item()
            global_step += 1

            if i % cfg["log"]["print_every"] == 0:
                avg_loss = running / cfg["log"]["print_every"]
                progress.set_postfix(loss=f"{avg_loss:.4f}")
                running = 0.0

        elapsed_epoch = time.time() - start_epoch
        print(f"Epoch {epoch} completed in {elapsed_epoch:.1f}s")

        # --- Validate after each epoch ---
        cer, exact, count = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: CER={cer:.3f}, Exact={exact:.3f} (N={count})")

        # Decay learning rate, remove cer if using by time interval.
        scheduler.step(cer)

        # --- Save checkpoint ---
        ckpt_path = os.path.join(cfg["log"]["ckpt_dir"], f"epoch{epoch:02d}.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}\n")
        
        # --- LOG TO FILE ---
        log_file = os.path.join(cfg["log"]["ckpt_dir"], "logs.txt")
        with open(log_file, "a") as f:
            f.write(
                f"Epoch {epoch:02d} | "
                f"Loss(avg): {running / max(1, i):.4f} | "
                f"CER: {cer:.4f} | "
                f"Exact: {exact:.4f} | "
                f"ValSamples: {count} | "
                f"Time: {elapsed_epoch:.1f}s\n"
            )

if __name__ == "__main__":
    main()
