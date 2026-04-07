"""
Inflect — Para Module Training Script
Phase 2a: Supervised Fine-Tuning (SFT) on paralinguistic Mimi latents.

Usage (run in PowerShell from repo root):
    python inflect/para_module/train.py

Checkpoints saved to: inflect/para_module/checkpoints/
Logs printed to stdout every 10 steps.
"""

import sys
import math
import time
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inflect.para_module.model import ParaModule, count_params

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = PROJECT_ROOT / "inflect/data/paralinguistic_dataset.pt"
CKPT_DIR      = PROJECT_ROOT / "inflect/para_module/checkpoints"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS        = 150
BATCH_SIZE    = 32
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.1       # 10% validation
WARMUP_EPOCHS = 5
SEED          = 42

# Loss weights
W_LATENT      = 1.0       # L1 loss on predicted Mimi latents
W_DURATION    = 0.1       # MSE loss on predicted duration

LOG_EVERY     = 10        # steps
SAVE_EVERY    = 10        # epochs
# ──────────────────────────────────────────────────────────────────────────────


class ParaDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "tag_id":         torch.tensor(s["tag_id"], dtype=torch.long),
            "speaker_latents": s["speaker_latents"].float(),  # [N, 512]
            "para_latents":    s["para_latents"].float(),     # [T, 512]
            "para_T":         torch.tensor(s["para_latents"].shape[0], dtype=torch.long),
        }


def collate_fn(batch):
    """Pad variable-length sequences in a batch."""
    tag_ids  = torch.stack([b["tag_id"] for b in batch])
    para_Ts  = torch.stack([b["para_T"] for b in batch])

    # Pad speaker latents to max N in batch
    max_N = max(b["speaker_latents"].shape[0] for b in batch)
    speaker_padded = torch.zeros(len(batch), max_N, 512)
    for i, b in enumerate(batch):
        N = b["speaker_latents"].shape[0]
        speaker_padded[i, :N] = b["speaker_latents"]

    # Pad para latents to max T in batch
    max_T = max(b["para_latents"].shape[0] for b in batch)
    para_padded = torch.zeros(len(batch), max_T, 512)
    para_mask   = torch.zeros(len(batch), max_T, dtype=torch.bool)  # True = valid
    for i, b in enumerate(batch):
        T = b["para_latents"].shape[0]
        para_padded[i, :T] = b["para_latents"]
        para_mask[i, :T]   = True

    return {
        "tag_ids":         tag_ids,           # [B]
        "speaker_latents": speaker_padded,    # [B, max_N, 512]
        "para_latents":    para_padded,       # [B, max_T, 512]
        "para_mask":       para_mask,         # [B, max_T]
        "para_Ts":         para_Ts,           # [B]
    }


def compute_loss(model_out: dict, batch: dict) -> dict:
    pred_latents  = model_out["pred_latents"]   # [B, T, 512]
    pred_duration = model_out["pred_duration"]  # [B, 1]
    target        = batch["para_latents"]       # [B, T, 512]
    mask          = batch["para_mask"]          # [B, T]
    target_T      = batch["para_Ts"].float()    # [B]

    # L1 loss on latents — only over valid (non-padded) frames
    mask_exp  = mask.unsqueeze(-1).expand_as(pred_latents)  # [B, T, 512]
    latent_loss = F.l1_loss(pred_latents[mask_exp], target[mask_exp])

    # Duration loss — predict frame count correctly
    dur_loss = F.mse_loss(pred_duration.squeeze(-1), target_T)

    total = W_LATENT * latent_loss + W_DURATION * dur_loss
    return {
        "total":   total,
        "latent":  latent_loss,
        "duration": dur_loss,
    }


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train():
    torch.manual_seed(SEED)
    random.seed(SEED)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──
    print(f"Loading dataset from {DATASET_PATH}...")
    raw = torch.load(DATASET_PATH, weights_only=False)
    print(f"  {len(raw)} samples loaded")

    dataset = ParaDataset(raw)
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ── Model ──
    model = ParaModule().to(DEVICE)
    print(f"ParaModule: {count_params(model):,} parameters")
    print(f"Device: {DEVICE}")
    print(f"Train: {n_train} | Val: {n_val}")
    print()

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))  # cosine annealing

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ──
    best_val_loss = float("inf")
    step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = defaultdict(float)
        epoch_steps  = 0

        t0 = time.time()
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            B = batch["tag_ids"].shape[0]
            T = batch["para_latents"].shape[1]

            optimizer.zero_grad()
            out   = model(batch["speaker_latents"], batch["tag_ids"], target_T=T)
            losses = compute_loss(out, batch)
            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()
            epoch_steps += 1
            step += 1

            if step % LOG_EVERY == 0:
                print(f"  step {step:5d}  "
                      f"loss={losses['total'].item():.4f}  "
                      f"latent={losses['latent'].item():.4f}  "
                      f"dur={losses['duration'].item():.4f}  "
                      f"lr={get_lr(optimizer):.2e}")

        scheduler.step()

        # ── Validation ──
        model.eval()
        val_losses = defaultdict(float)
        val_steps  = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                T = batch["para_latents"].shape[1]
                out    = model(batch["speaker_latents"], batch["tag_ids"], target_T=T)
                losses = compute_loss(out, batch)
                for k, v in losses.items():
                    val_losses[k] += v.item()
                val_steps += 1

        avg_train = epoch_losses["total"] / max(1, epoch_steps)
        avg_val   = val_losses["total"]   / max(1, val_steps)
        elapsed   = time.time() - t0

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={get_lr(optimizer):.2e}  t={elapsed:.1f}s")

        # ── Checkpointing ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   avg_val,
            }, CKPT_DIR / "para_best.pt")
            print(f"  ✓ New best checkpoint saved (val={avg_val:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "val_loss": avg_val,
            }, CKPT_DIR / f"para_epoch_{epoch:03d}.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {CKPT_DIR / 'para_best.pt'}")


if __name__ == "__main__":
    train()
