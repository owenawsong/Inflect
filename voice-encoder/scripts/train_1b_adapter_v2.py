"""
Inflect Phase 1B v2 - Improved Style Adapter

Fixes the centroid-only problem from v1. Now trains against full
[511, 1, 256] voice pack tensors, not just their means.

Key improvements:
  1. MSE loss against full voice pack (all 511 timesteps)
  2. Triplet loss: push encoder output toward nearest voice, away from others
  3. No time_modulation noise — outputs a clean broadcast like real voice packs
  4. Trains longer with lower LR

Runtime: ~45-60 min on RTX 3060.

Usage:
    python train_1b_adapter_v2.py
"""

import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE       = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
MANIFEST   = BASE / "data" / "manifest.csv"
CKPT_DIR   = BASE / "checkpoints"
VOICE_DIR  = BASE / "data" / "kokoro_voices"

ENCODER_CKPT = CKPT_DIR / "ve_latest.pt"

EMBED_DIM      = 256
PROJ_TIMESTEPS = 511
LR             = 1e-4
EPOCHS         = 500
BATCH_SIZE     = 64
SAVE_EVERY     = 50
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


def load_voice_packs():
    """Load all voice packs. Returns (names, centroids [N,256], full_packs [N,511,256])."""
    files = sorted(VOICE_DIR.glob("**/*.pt"))
    names, centroids, packs = [], [], []
    for f in files:
        pack = torch.load(f, map_location="cpu", weights_only=True)  # [511,1,256]
        pack_2d = pack[:, 0, :]                                       # [511,256]
        centroid = F.normalize(pack_2d.mean(dim=0), dim=0)           # [256]
        names.append(f.stem)
        centroids.append(centroid)
        packs.append(pack_2d)
    centroids_t = torch.stack(centroids).to(DEVICE)  # [N,256]
    packs_t     = torch.stack(packs).to(DEVICE)      # [N,511,256]
    print(f"Loaded {len(names)} voice packs")
    return names, centroids_t, packs_t


def load_random_mels(df, n):
    rows = df.sample(n=min(n, len(df)), replace=True)
    mels = []
    T = 160
    for _, row in rows.iterrows():
        try:
            mel = torch.load(row["mel_path"], weights_only=True)
            t = mel.shape[1]
            if t >= T:
                start = random.randint(0, t - T)
                mel = mel[:, start:start+T]
            else:
                mel = F.pad(mel, (0, T - t))
            mels.append(mel)
        except Exception:
            mels.append(torch.zeros(80, T))
    return torch.stack(mels).to(DEVICE)  # [n,80,T]


class StyleAdapterV2(nn.Module):
    """
    Clean adapter — no per-timestep noise, just a well-trained MLP.
    Output: broadcasts a single 256-dim vector to [511,1,256].
    This matches how real Kokoro voice packs actually work (all 511
    timesteps are nearly identical).
    """
    def __init__(self, dim=EMBED_DIM, timesteps=PROJ_TIMESTEPS):
        super().__init__()
        self.timesteps = timesteps
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward_embed(self, x):
        """x: [B,256] -> refined [B,256] L2-normalized"""
        return F.normalize(self.norm(x + self.net(x)), dim=-1)

    def forward(self, x):
        """x: [B,256] -> [B,511,1,256] Kokoro format"""
        emb = self.forward_embed(x)                          # [B,256]
        emb = emb.unsqueeze(1).unsqueeze(2)                  # [B,1,1,256]
        return emb.expand(-1, self.timesteps, 1, -1).contiguous()


def train():
    df = pd.read_csv(MANIFEST)
    voice_names, voice_centroids, voice_packs = load_voice_packs()
    N_voices = len(voice_names)

    from train_ve import VoiceEncoder
    encoder = VoiceEncoder().to(DEVICE)
    ckpt = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Encoder loaded (epoch {ckpt.get('epoch','?')})")

    adapter = StyleAdapterV2().to(DEVICE)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter v2 params: {n_params/1e3:.1f}K")
    print(f"Training for {EPOCHS} epochs...")

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        adapter.train()

        mels = load_random_mels(df, BATCH_SIZE)

        with torch.no_grad():
            enc_emb = encoder.embed(mels)  # [B,256]

        adapted = adapter.forward_embed(enc_emb)  # [B,256]

        # ── Loss 1: Nearest voice pack MSE (full 511 timesteps) ──────────
        # Find nearest voice pack centroid for each item in batch
        sim = torch.mm(adapted, voice_centroids.T)   # [B,N]
        nearest_idx = sim.argmax(dim=1)              # [B]
        # Target: full pack of nearest voice [B,511,256]
        target_packs = voice_packs[nearest_idx]      # [B,511,256]
        # Predicted full pack
        T_steps = target_packs.shape[1]
        pred_packs = adapted.unsqueeze(1).expand(-1, T_steps, -1)  # [B,T,256]
        mse_loss = F.mse_loss(pred_packs, target_packs)

        # ── Loss 2: Contrastive — push toward nearest, away from random ──
        # For each item, nearest voice = positive, random other = negative
        rand_idx = torch.randint(0, N_voices, (BATCH_SIZE,), device=DEVICE)
        # Make sure random != nearest
        rand_idx = torch.where(rand_idx == nearest_idx,
                               (rand_idx + 1) % N_voices, rand_idx)

        pos_sim = (adapted * voice_centroids[nearest_idx]).sum(dim=-1)  # [B]
        neg_sim = (adapted * voice_centroids[rand_idx]).sum(dim=-1)     # [B]
        margin = 0.3
        triplet_loss = F.relu(neg_sim - pos_sim + margin).mean()

        # ── Loss 3: Don't collapse — keep batch diverse ───────────────────
        gram = torch.mm(adapted, adapted.T)  # [B,B]
        eye  = torch.eye(BATCH_SIZE, device=DEVICE)
        diversity_loss = ((gram - eye) ** 2).mean()

        loss = mse_loss + 0.5 * triplet_loss + 0.02 * diversity_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            # Measure alignment: avg cosine sim to nearest voice
            with torch.no_grad():
                avg_sim = pos_sim.mean().item()
            print(f"Epoch {epoch+1:03d} — loss={loss.item():.4f} "
                  f"mse={mse_loss.item():.4f} "
                  f"triplet={triplet_loss.item():.4f} "
                  f"avg_sim={avg_sim:.4f} lr={lr:.2e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                "epoch":   epoch + 1,
                "adapter": adapter.state_dict(),
                "loss":    best_loss,
                "version": "v2",
            }, CKPT_DIR / "adapter_v2_best.pt")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                "epoch":   epoch + 1,
                "adapter": adapter.state_dict(),
                "loss":    loss.item(),
                "version": "v2",
            }, CKPT_DIR / f"adapter_v2_epoch_{epoch+1:03d}.pt")

    torch.save({
        "epoch":   EPOCHS,
        "adapter": adapter.state_dict(),
        "loss":    best_loss,
        "version": "v2",
    }, CKPT_DIR / "adapter_v2_latest.pt")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Saved: {CKPT_DIR / 'adapter_v2_latest.pt'}")
    print("\nWatch avg_sim — if it reaches 0.4+ the adapter is aligning well.")


if __name__ == "__main__":
    train()
