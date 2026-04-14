"""
Inflect Phase 1C Augmented — Robust Alignment Fine-Tune

Continues from 1c_best.pt. Augmentation runs on GPU (batch ops),
not CPU per-item, so the GPU stays saturated.

Augmentations (GPU batch):
  - Frequency masking
  - Time masking
  - Gaussian noise
  - Random amplitude scale

Runtime: ~60-90 min on RTX 3060 for 2000 epochs.
Saves every 50 epochs. Best by val_sim -> 1c_aug_best.pt

Usage:
    python train_1c_augmented.py
"""

import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE        = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
CKPT_DIR    = BASE / "checkpoints"
VOICE_DIR   = BASE / "data" / "kokoro_voices"
MANIFEST    = BASE / "data" / "1c_manifest.csv"
RESUME_CKPT = CKPT_DIR / "1c_best.pt"

LR_ENCODER = 2e-5
LR_ADAPTER = 8e-5
EPOCHS     = 2000
BATCH_SIZE = 256
SAVE_EVERY = 50
TEMP       = 0.07
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FREQ_MASK_MAX = 15
TIME_MASK_MAX = 20
N_FREQ_MASKS  = 2
N_TIME_MASKS  = 2
NOISE_STD     = 0.02

print(f"Device: {DEVICE}")


# ── GPU batch augmentation (fast) ─────────────────────────────────────────────

def augment_batch(mels: torch.Tensor) -> torch.Tensor:
    """mels: [B, 80, 160] on GPU — all ops stay on GPU"""
    B, F_, T_ = mels.shape
    mels = mels.clone()

    for _ in range(N_FREQ_MASKS):
        f  = random.randint(1, FREQ_MASK_MAX)
        f0 = random.randint(0, F_ - f)
        mels[:, f0:f0 + f, :] = 0.0

    for _ in range(N_TIME_MASKS):
        t  = random.randint(1, TIME_MASK_MAX)
        t0 = random.randint(0, T_ - t)
        mels[:, :, t0:t0 + t] = 0.0

    mels = mels + torch.randn_like(mels) * NOISE_STD

    scale = torch.empty(B, 1, 1, device=mels.device).uniform_(0.85, 1.15)
    mels  = mels * scale

    return mels


# ── Dataset — no augmentation, just fast RAM loading ─────────────────────────

class MelDataset(Dataset):
    def __init__(self, records, pack_map):
        self.pack_map    = pack_map
        print(f"  Pre-loading {len(records)} mels into RAM...")
        self.mels        = [torch.load(r["mel_path"], weights_only=True) for r in records]
        self.voice_names = [r["voice_name"] for r in records]
        print(f"  Done.")

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return self.mels[idx], self.pack_map[self.voice_names[idx]]


def collate_fn(batch):
    mels  = torch.stack([b[0] for b in batch])
    packs = torch.stack([b[1] for b in batch])
    return mels, packs


# ── InfoNCE ───────────────────────────────────────────────────────────────────

def infonce_loss(embeddings, centroids, temperature=TEMP):
    logits = torch.mm(embeddings, centroids.T) / temperature
    labels = torch.arange(len(embeddings), device=embeddings.device)
    return F.cross_entropy(logits, labels)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    df = pd.read_csv(MANIFEST)
    print(f"Manifest: {len(df)} clips, {df['voice_name'].nunique()} voices")

    pack_map = {}
    for vf in sorted(VOICE_DIR.glob("**/*.pt")):
        pack = torch.load(vf, map_location="cpu", weights_only=True)
        pack_map[vf.stem] = pack[:, 0, :]   # [511,256]
    print(f"Loaded {len(pack_map)} voice packs")

    records = df.to_dict("records")
    random.shuffle(records)
    split    = int(0.9 * len(records))

    train_ds = MelDataset(records[:split], pack_map)
    val_ds   = MelDataset(records[split:],  pack_map)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    from train_ve import VoiceEncoder
    from train_1b_adapter_v2 import StyleAdapterV2

    if not RESUME_CKPT.exists():
        print(f"ERROR: {RESUME_CKPT} not found. Run train_1c_alignment.py first.")
        sys.exit(1)

    ckpt    = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
    encoder = VoiceEncoder().to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    adapter = StyleAdapterV2().to(DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    print(f"Resumed from 1c_best.pt (epoch {ckpt.get('epoch','?')}, val_sim={ckpt.get('val_sim',0):.4f})")

    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": LR_ENCODER},
        {"params": adapter.parameters(), "lr": LR_ADAPTER},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )

    # Each epoch runs the full dataset REPEATS times with different augmentation
    REPEATS = 8
    print(f"Training {EPOCHS} epochs x {REPEATS} passes | batch={BATCH_SIZE} | "
          f"train={len(train_ds)} val={len(val_ds)}\n")

    best_val_sim = ckpt.get("val_sim", 0.0)

    for epoch in range(EPOCHS):
        encoder.train()
        adapter.train()

        t_loss = t_sim = 0.0
        n_batches = 0

        # Multiple passes per epoch with different augmentation each time
        for _ in range(REPEATS):
            for mels, target_packs in train_loader:
                mels         = mels.to(DEVICE, non_blocking=True)
                target_packs = target_packs.to(DEVICE, non_blocking=True)

                # Augment on GPU
                mels = augment_batch(mels)

                enc_emb    = encoder.embed(mels)
                adapted    = adapter.forward_embed(enc_emb)
                tgt_cent   = F.normalize(target_packs.mean(dim=1), dim=-1)
                T_steps    = target_packs.shape[1]
                pred_packs = adapted.unsqueeze(1).expand(-1, T_steps, -1)

                mse_loss = F.mse_loss(pred_packs, target_packs)
                cos_loss = (1.0 - (adapted * tgt_cent).sum(dim=-1)).mean()
                nce_loss = infonce_loss(adapted, tgt_cent)
                gram     = torch.mm(adapted, adapted.T)
                eye      = torch.eye(adapted.shape[0], device=DEVICE)
                div_loss = ((gram - eye) ** 2).mean()

                loss = mse_loss + 0.5 * cos_loss + 0.3 * nce_loss + 0.02 * div_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(adapter.parameters()), 1.0
                )
                optimizer.step()

                with torch.no_grad():
                    sim = (adapted * tgt_cent).sum(dim=-1).mean().item()
                t_loss    += loss.item()
                t_sim     += sim
                n_batches += 1

        scheduler.step()

        # Validate (no augmentation)
        encoder.eval()
        adapter.eval()
        v_loss = v_sim = 0.0
        n_val  = 0

        with torch.no_grad():
            for mels, target_packs in val_loader:
                mels         = mels.to(DEVICE, non_blocking=True)
                target_packs = target_packs.to(DEVICE, non_blocking=True)
                enc_emb      = encoder.embed(mels)
                adapted      = adapter.forward_embed(enc_emb)
                tgt_cent     = F.normalize(target_packs.mean(dim=1), dim=-1)
                T_steps      = target_packs.shape[1]
                pred_packs   = adapted.unsqueeze(1).expand(-1, T_steps, -1)
                mse          = F.mse_loss(pred_packs, target_packs)
                cos          = (1.0 - (adapted * tgt_cent).sum(dim=-1)).mean()
                nce          = infonce_loss(adapted, tgt_cent)
                v_loss      += (mse + 0.5 * cos + 0.3 * nce).item()
                v_sim       += (adapted * tgt_cent).sum(dim=-1).mean().item()
                n_val       += 1

        avg_t = t_loss / max(n_batches, 1)
        avg_v = v_loss / max(n_val, 1)
        avg_s = v_sim  / max(n_val, 1)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            lr_enc = optimizer.param_groups[0]["lr"]
            lr_adp = optimizer.param_groups[1]["lr"]
            print(f"Epoch {epoch+1:04d} — train={avg_t:.4f}  val={avg_v:.4f}  "
                  f"val_sim={avg_s:.4f}  lr_enc={lr_enc:.1e}  lr_adp={lr_adp:.1e}")

        if avg_s > best_val_sim:
            best_val_sim = avg_s
            torch.save({
                "epoch":    epoch + 1,
                "encoder":  encoder.state_dict(),
                "adapter":  adapter.state_dict(),
                "val_loss": avg_v,
                "val_sim":  best_val_sim,
                "version":  "1c_aug",
            }, CKPT_DIR / "1c_aug_best.pt")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                "epoch":    epoch + 1,
                "encoder":  encoder.state_dict(),
                "adapter":  adapter.state_dict(),
                "val_loss": avg_v,
                "val_sim":  avg_s,
                "version":  "1c_aug",
            }, CKPT_DIR / f"1c_aug_epoch_{epoch+1:04d}.pt")

    torch.save({
        "epoch":    EPOCHS,
        "encoder":  encoder.state_dict(),
        "adapter":  adapter.state_dict(),
        "val_sim":  best_val_sim,
        "version":  "1c_aug",
    }, CKPT_DIR / "1c_aug_latest.pt")

    print(f"\nDone. Best val_sim: {best_val_sim:.4f}")
    print(f"Next: python test_cloning.py --use-dataset")


if __name__ == "__main__":
    train()
