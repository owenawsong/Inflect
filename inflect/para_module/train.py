"""
Inflect — Para Module Training Script (v2, Mel Spectrogram)

Usage (run from repo root in PowerShell):
    python inflect/para_module/train.py

Checkpoints saved to: inflect/para_module/checkpoints/
"""

import sys
import math
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inflect.para_module.model import ParaModule, count_params

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DATASET_PATH = PROJECT_ROOT / "inflect/data/paralinguistic_dataset_mel.pt"
DEFAULT_CKPT_DIR     = PROJECT_ROOT / "inflect/para_module/checkpoints"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS       = 150
BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.1
WARMUP_EPOCHS = 5
SEED         = 42

W_MEL        = 1.0     # L1 loss on mel frames
W_DURATION   = 0.0     # duration loss disabled (scale mismatch with mel L1)

LOG_EVERY    = 10
SAVE_EVERY   = 10
# ──────────────────────────────────────────────────────────────────────────────


class ParaDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "tag_id":      torch.tensor(s["tag_id"], dtype=torch.long),
            "speaker_emb": s["speaker_emb"].float(),   # [80]
            "para_mel":    s["para_mel"].float(),       # [T, 80]
            "para_T":      torch.tensor(s["para_mel"].shape[0], dtype=torch.long),
        }


def collate_fn(batch):
    tag_ids  = torch.stack([b["tag_id"]      for b in batch])
    spk_embs = torch.stack([b["speaker_emb"] for b in batch])
    para_Ts  = torch.stack([b["para_T"]      for b in batch])

    max_T = max(b["para_mel"].shape[0] for b in batch)
    para_padded = torch.zeros(len(batch), max_T, 80)
    para_mask   = torch.zeros(len(batch), max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        T = b["para_mel"].shape[0]
        para_padded[i, :T] = b["para_mel"]
        para_mask[i, :T]   = True

    return {
        "tag_ids":     tag_ids,       # [B]
        "speaker_emb": spk_embs,      # [B, 80]
        "para_mel":    para_padded,   # [B, max_T, 80]
        "para_mask":   para_mask,     # [B, max_T]
        "para_Ts":     para_Ts,       # [B]
    }


def compute_loss(model_out, batch):
    pred_mel  = model_out["pred_mel"]       # [B, T, 80]
    pred_dur  = model_out["pred_duration"]  # [B, 1]
    target    = batch["para_mel"]           # [B, T, 80]
    mask      = batch["para_mask"]          # [B, T]
    target_T  = batch["para_Ts"].float()

    mask_exp  = mask.unsqueeze(-1).expand_as(pred_mel)
    mel_loss  = F.l1_loss(pred_mel[mask_exp], target[mask_exp])
    dur_loss  = F.mse_loss(pred_dur.squeeze(-1), target_T)

    total = W_MEL * mel_loss + W_DURATION * dur_loss
    return {"total": total, "mel": mel_loss, "duration": dur_loss}


def train(dataset_path: Path, ckpt_dir: Path):
    torch.manual_seed(SEED)
    random.seed(SEED)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    raw = torch.load(dataset_path, weights_only=False)
    print(f"  {len(raw)} samples loaded")

    dataset = ParaDataset(raw)
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model = ParaModule().to(DEVICE)
    print(f"ParaModule: {count_params(model):,} parameters")
    print(f"Device: {DEVICE}  |  Train: {n_train}  |  Val: {n_val}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = defaultdict(float)
        epoch_steps  = 0
        t0 = time.time()

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            T = batch["para_mel"].shape[1]
            optimizer.zero_grad()
            out    = model(batch["speaker_emb"], batch["tag_ids"], target_T=T)
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
                      f"mel={losses['mel'].item():.4f}  "
                      f"dur={losses['duration'].item():.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        model.eval()
        val_losses = defaultdict(float)
        val_steps  = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                T = batch["para_mel"].shape[1]
                out    = model(batch["speaker_emb"], batch["tag_ids"], target_T=T)
                losses = compute_loss(out, batch)
                for k, v in losses.items():
                    val_losses[k] += v.item()
                val_steps += 1

        avg_train = epoch_losses["total"] / max(1, epoch_steps)
        avg_val   = val_losses["total"]   / max(1, val_steps)
        elapsed   = time.time() - t0

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "val_loss": avg_val,
            }, ckpt_dir / "para_best.pt")
            print(f"  ✓ Best checkpoint saved (val={avg_val:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(), "val_loss": avg_val,
            }, ckpt_dir / f"para_epoch_{epoch:03d}.pt")

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'para_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    args = parser.parse_args()
    train(args.dataset, args.ckpt_dir)
