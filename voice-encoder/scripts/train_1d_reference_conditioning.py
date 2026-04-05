"""
Inflect Phase 1D — Reference Speaker Conditioning

Instead of snapping to the nearest of 28 voices, this learns SOFT INTERPOLATION.
The adapter outputs a weighted blend of ALL voice packs, producing a continuous
style vector that can represent voices BETWEEN the 28 built-in ones.

Architecture:
  - Encoder produces [B, 256] embedding (frozen from 1c)
  - ReferenceConditioner outputs [B, N_voices] attention weights (softmax)
  - Style = sum(weights[i] * voice_pack[i] for i in range(N_voices))
  - Trained with MSE + cosine similarity against target voice packs

Key difference from 1c:
  - 1c: argmax → nearest voice (discrete, snaps to one of 28)
  - 1d: softmax → weighted blend (continuous, can interpolate)

Runtime: ~30-45 min on RTX 3060 for 500 epochs.

Usage:
    python train_1d_reference_conditioning.py
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
RESUME_CKPT = CKPT_DIR / "1c_aug_best.pt"  # Use best augmented checkpoint

LR           = 5e-5
EPOCHS       = 500
BATCH_SIZE   = 128
SAVE_EVERY   = 50
TEMP         = 0.07
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# ── Reference Conditioner ─────────────────────────────────────────────────────

class ReferenceConditioner(nn.Module):
    """
    Takes encoder embedding [B, 256] → outputs soft attention weights [B, N_voices].
    The final style is a weighted sum of all voice packs.
    """
    def __init__(self, dim=256, n_voices=28):
        super().__init__()
        self.n_voices = n_voices
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, n_voices),
        )

    def forward(self, x):
        """x: [B, 256] → weights: [B, N_voices] (softmax normalized)"""
        return F.softmax(self.net(x), dim=-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MelDataset(Dataset):
    def __init__(self, records, pack_map, voice_names_list):
        self.pack_map = pack_map
        self.voice_names_list = voice_names_list
        self.name_to_idx = {n: i for i, n in enumerate(voice_names_list)}
        print(f"  Pre-loading {len(records)} mels into RAM...")
        self.mels = [torch.load(r["mel_path"], weights_only=True) for r in records]
        self.voice_names = [r["voice_name"] for r in records]
        print(f"  Done.")

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return self.mels[idx], self.name_to_idx[self.voice_names[idx]]


def collate_fn(batch):
    mels = torch.stack([b[0] for b in batch])
    voice_indices = torch.tensor([b[1] for b in batch])
    return mels, voice_indices


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    df = pd.read_csv(MANIFEST)
    print(f"Manifest: {len(df)} clips, {df['voice_name'].nunique()} voices")

    # Load all voice packs
    voice_files = sorted(VOICE_DIR.glob("**/*.pt"))
    voice_names_list = [f.stem for f in voice_files]
    N_voices = len(voice_names_list)
    print(f"Loading {N_voices} voice packs...")

    pack_map = {}
    voice_packs_2d = []  # [N, 511, 256]
    voice_centroids = []  # [N, 256]
    for vf in voice_files:
        pack = torch.load(vf, map_location="cpu", weights_only=True)  # [511,1,256]
        pack_2d = pack[:, 0, :]  # [511, 256]
        centroid = F.normalize(pack_2d.mean(dim=0), dim=0)  # [256]
        pack_map[vf.stem] = pack_2d
        voice_packs_2d.append(pack_2d)
        voice_centroids.append(centroid)

    voice_packs_t = torch.stack(voice_packs_2d).to(DEVICE)  # [N, 511, 256]
    voice_centroids_t = torch.stack(voice_centroids).to(DEVICE)  # [N, 256]
    print(f"Loaded {N_voices} voice packs")

    # Split data
    records = df.to_dict("records")
    random.shuffle(records)
    split = int(0.9 * len(records))
    train_ds = MelDataset(records[:split], pack_map, voice_names_list)
    val_ds = MelDataset(records[split:], pack_map, voice_names_list)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True,
    )

    # Load frozen encoder+adapter from 1c
    from train_ve import VoiceEncoder
    from train_1b_adapter_v2 import StyleAdapterV2

    if not RESUME_CKPT.exists():
        print(f"ERROR: {RESUME_CKPT} not found. Run train_1c_augmented.py first.")
        sys.exit(1)

    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
    encoder = VoiceEncoder().to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    adapter = StyleAdapterV2().to(DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    encoder.eval()
    adapter.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in adapter.parameters():
        p.requires_grad_(False)
    print(f"Resumed from {RESUME_CKPT.name} (epoch {ckpt.get('epoch','?')}, val_sim={ckpt.get('val_sim',0):.4f})")

    # Create reference conditioner
    conditioner = ReferenceConditioner(dim=256, n_voices=N_voices).to(DEVICE)
    n_params = sum(p.numel() for p in conditioner.parameters())
    print(f"Conditioner params: {n_params/1e3:.1f}K")

    optimizer = torch.optim.AdamW(conditioner.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    best_val_sim = 0.0

    print(f"Training {EPOCHS} epochs | batch={BATCH_SIZE} | "
          f"train={len(train_ds)} val={len(val_ds)}\n")

    for epoch in range(EPOCHS):
        conditioner.train()
        t_loss = t_sim = 0.0
        n_batches = 0

        for mels, voice_indices in train_loader:
            mels = mels.to(DEVICE, non_blocking=True)
            voice_indices = voice_indices.to(DEVICE)

            with torch.no_grad():
                enc_emb = encoder.embed(mels)  # [B, 256]

            # Get attention weights from conditioner
            weights = conditioner(enc_emb)  # [B, N_voices]

            # Compute blended style: weighted sum of all voice packs
            # weights: [B, N], voice_packs_t: [N, 511, 256]
            # Result: [B, 511, 256]
            blended = torch.einsum("bn,ntd->btd", weights, voice_packs_t)

            # Target: the actual voice pack for each sample
            target_packs = voice_packs_t[voice_indices]  # [B, 511, 256]

            # Loss 1: MSE against full voice pack
            mse_loss = F.mse_loss(blended, target_packs)

            # Loss 2: Cosine similarity of centroids
            blended_centroid = F.normalize(blended.mean(dim=1), dim=-1)  # [B, 256]
            target_centroid = F.normalize(target_packs.mean(dim=1), dim=-1)  # [B, 256]
            cos_loss = (1.0 - (blended_centroid * target_centroid).sum(dim=-1)).mean()

            # Loss 3: Cross-entropy on weights vs one-hot (encourage correct voice)
            # This pushes the conditioner to put weight on the correct voice
            ce_loss = F.cross_entropy(weights, voice_indices)

            loss = mse_loss + 0.5 * cos_loss + 0.3 * ce_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(conditioner.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                sim = (blended_centroid * target_centroid).sum(dim=-1).mean().item()
            t_loss += loss.item()
            t_sim += sim
            n_batches += 1

        scheduler.step()

        # Validate
        conditioner.eval()
        v_loss = v_sim = 0.0
        n_val = 0

        with torch.no_grad():
            for mels, voice_indices in val_loader:
                mels = mels.to(DEVICE, non_blocking=True)
                voice_indices = voice_indices.to(DEVICE)

                enc_emb = encoder.embed(mels)
                weights = conditioner(enc_emb)
                blended = torch.einsum("bn,ntd->btd", weights, voice_packs_t)
                target_packs = voice_packs_t[voice_indices]

                blended_centroid = F.normalize(blended.mean(dim=1), dim=-1)
                target_centroid = F.normalize(target_packs.mean(dim=1), dim=-1)

                mse = F.mse_loss(blended, target_packs)
                cos = (1.0 - (blended_centroid * target_centroid).sum(dim=-1)).mean()
                ce = F.cross_entropy(weights, voice_indices)
                v_loss += (mse + 0.5 * cos + 0.3 * ce).item()
                v_sim += (blended_centroid * target_centroid).sum(dim=-1).mean().item()
                n_val += 1

        avg_t = t_loss / max(n_batches, 1)
        avg_v = v_loss / max(n_val, 1)
        avg_s = v_sim / max(n_val, 1)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            # Also measure weight entropy (lower = more confident)
            with torch.no_grad():
                # Sample a batch and check weight distribution
                sample_mels, _ = next(iter(train_loader))
                sample_mels = sample_mels[:8].to(DEVICE)
                sample_emb = encoder.embed(sample_mels)
                sample_weights = conditioner(sample_emb)
                # Entropy of weights (normalized)
                entropy = -(sample_weights * (sample_weights + 1e-8).log()).sum(dim=-1).mean().item()
                max_weight = sample_weights.max(dim=-1).values.mean().item()
            print(f"Epoch {epoch+1:04d} — train={avg_t:.4f}  val={avg_v:.4f}  "
                  f"val_sim={avg_s:.4f}  lr={lr:.1e}  entropy={entropy:.3f}  "
                  f"max_weight={max_weight:.3f}")

        if avg_s > best_val_sim:
            best_val_sim = avg_s
            torch.save({
                "epoch":       epoch + 1,
                "encoder":     encoder.state_dict(),
                "adapter":     adapter.state_dict(),
                "conditioner": conditioner.state_dict(),
                "voice_names": voice_names_list,
                "val_loss":    avg_v,
                "val_sim":     best_val_sim,
                "version":     "1d_ref",
            }, CKPT_DIR / "1d_ref_best.pt")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                "epoch":       epoch + 1,
                "encoder":     encoder.state_dict(),
                "adapter":     adapter.state_dict(),
                "conditioner": conditioner.state_dict(),
                "voice_names": voice_names_list,
                "val_loss":    avg_v,
                "val_sim":     avg_s,
                "version":     "1d_ref",
            }, CKPT_DIR / f"1d_ref_epoch_{epoch+1:04d}.pt")

    torch.save({
        "epoch":       EPOCHS,
        "encoder":     encoder.state_dict(),
        "adapter":     adapter.state_dict(),
        "conditioner": conditioner.state_dict(),
        "voice_names": voice_names_list,
        "val_sim":     best_val_sim,
        "version":     "1d_ref",
    }, CKPT_DIR / "1d_ref_latest.pt")

    print(f"\nDone. Best val_sim: {best_val_sim:.4f}")
    print(f"Saved: {CKPT_DIR / '1d_ref_latest.pt'}")
    print(f"Next: update inflect_tts.py to use 1d_ref checkpoint for reference cloning")


if __name__ == "__main__":
    train()
