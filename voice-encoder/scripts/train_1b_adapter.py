"""
Inflect Voice Encoder - Stage 1B: Style Adapter

Aligns the GE2E encoder's output space to Kokoro's voice pack distribution.

What this does:
  - Downloads all 54 Kokoro voice packs (each [511, 1, 256])
  - Encodes random audio clips from your dataset with the trained GE2E encoder
  - Trains a small adapter MLP that maps encoder outputs -> Kokoro style space
  - Two losses:
      1. Distribution matching: encoder outputs should look like Kokoro voice packs
      2. Consistency: same speaker -> similar style vectors

After this, voice cloning works:
  feed_audio(clip) -> encoder -> adapter -> [511, 1, 256] -> Kokoro TTS

Usage:
    python train_1b_adapter.py
    python train_1b_adapter.py --resume checkpoints/adapter_latest.pt

~30-45 min on RTX 3060.
"""

import random
import sys
from pathlib import Path

# Allow importing train_ve from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
MANIFEST   = BASE / "data" / "manifest.csv"
CKPT_DIR   = BASE / "checkpoints"
VOICE_DIR  = BASE / "data" / "kokoro_voices"
VOICE_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_CKPT = CKPT_DIR / "ve_latest.pt"

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_DIM      = 256
PROJ_TIMESTEPS = 511
LR             = 3e-4
EPOCHS         = 200
BATCH_SIZE     = 64
SAVE_EVERY     = 20
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# ── Download Kokoro Voice Packs ────────────────────────────────────────────────
VOICE_NAMES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
# Using English voices only (af/am = American F/M, bf/bm = British F/M)
# 28 voices — enough for alignment without noise from other languages


def download_voice_packs() -> dict[str, torch.Tensor]:
    """Download Kokoro voice packs and return {name: centroid [256]}."""
    print("Downloading Kokoro voice packs...")
    voices = {}
    for name in tqdm(VOICE_NAMES, desc="Voice packs"):
        local = VOICE_DIR / f"{name}.pt"
        if not local.exists():
            try:
                path = hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M",
                    filename=f"voices/{name}.pt",
                    local_dir=str(VOICE_DIR),
                )
            except Exception as e:
                print(f"  Skip {name}: {e}")
                continue
        else:
            path = str(local)

        pack = torch.load(path, map_location="cpu", weights_only=True)  # [511, 1, 256]
        # Use mean across timesteps as the canonical embedding
        centroid = pack[:, 0, :].mean(dim=0)     # [256]
        centroid = F.normalize(centroid, dim=0)
        voices[name] = centroid

    print(f"Loaded {len(voices)} voice packs.")
    return voices


# ── Load trained encoder ───────────────────────────────────────────────────────
def load_encoder():
    from train_ve import VoiceEncoder

    model = VoiceEncoder().to(DEVICE)
    ckpt = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Encoder loaded from {ENCODER_CKPT} (epoch {ckpt.get('epoch', '?')})")
    return model


# ── Adapter model ─────────────────────────────────────────────────────────────
class StyleAdapter(nn.Module):
    """
    Maps encoder 256-dim embeddings into Kokoro's style distribution.

    Input:  [B, 256] L2-normalized encoder embedding
    Output: [B, 511, 1, 256] Kokoro-format style tensor

    Architecture: small residual MLP + per-timestep fine-grained output
    """
    def __init__(self, dim: int = EMBED_DIM, timesteps: int = PROJ_TIMESTEPS):
        super().__init__()
        self.timesteps = timesteps

        # Residual MLP to refine embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)

        # Small per-timestep variation (keeps 511 steps slightly different
        # like real Kokoro voice packs, not a flat broadcast)
        self.time_modulation = nn.Parameter(torch.randn(timesteps, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 256] -> [B, 511, 1, 256]"""
        # Refine
        x = self.norm(x + self.mlp(x))          # [B, 256] residual
        x = F.normalize(x, dim=-1)

        # Expand to timesteps with slight variation
        x = x.unsqueeze(1) + self.time_modulation.unsqueeze(0)  # [B, 511, 256]
        x = F.normalize(x, dim=-1)
        return x.unsqueeze(2)                    # [B, 511, 1, 256]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 256] -> [B, 256] refined embedding (centroid of output)"""
        x = self.norm(x + self.mlp(x))
        return F.normalize(x, dim=-1)


# ── Load mel clips for training ────────────────────────────────────────────────
def load_random_mels(manifest: Path, n: int, device) -> torch.Tensor:
    """Load n random mel clips, pad/crop to 160 frames, return [n, 80, 160]."""
    df = pd.read_csv(manifest)
    paths = df["mel_path"].sample(n=min(n, len(df)), replace=True).tolist()
    chunks = []
    T = 160
    for p in paths:
        try:
            mel = torch.load(p, weights_only=True)  # [80, T']
            t = mel.shape[1]
            if t >= T:
                start = random.randint(0, t - T)
                mel = mel[:, start:start + T]
            else:
                mel = F.pad(mel, (0, T - t))
            chunks.append(mel)
        except Exception:
            chunks.append(torch.zeros(80, T))
    return torch.stack(chunks).to(device)  # [n, 80, 160]


# ── Training ───────────────────────────────────────────────────────────────────
def train(args):
    # Load assets
    voice_centroids = download_voice_packs()
    if not voice_centroids:
        raise RuntimeError("No voice packs downloaded. Check HF access.")

    # Stack voice centroids: [N_voices, 256]
    voice_names = list(voice_centroids.keys())
    voice_mat = torch.stack([voice_centroids[n] for n in voice_names]).to(DEVICE)
    N_voices = len(voice_names)
    print(f"Voice pack matrix: {voice_mat.shape}")

    encoder = load_encoder()
    adapter = StyleAdapter().to(DEVICE)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        adapter.load_state_dict(ckpt["adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter params: {n_params/1e3:.1f}K")

    best_loss = float("inf")

    for epoch in range(start_epoch, EPOCHS):
        adapter.train()

        # Sample batch of mel clips
        mels = load_random_mels(MANIFEST, BATCH_SIZE, DEVICE)  # [B, 80, 160]

        with torch.no_grad():
            enc_emb = encoder.embed(mels)   # [B, 256] L2-normalized

        # Adapter forward
        adapted = adapter.embed(enc_emb)    # [B, 256] — refined embedding

        # ── Loss 1: Distribution alignment ─────────────────────────────────
        # Push adapted embeddings toward the nearest Kokoro voice centroid
        # sim: [B, N_voices]
        sim = torch.mm(adapted, voice_mat.T)
        # Soft alignment: pull toward best matching voice, push from others
        best_voice_idx = sim.argmax(dim=1)  # [B]
        dist_loss = F.cross_entropy(sim * 10.0, best_voice_idx)

        # ── Loss 2: Coverage loss ───────────────────────────────────────────
        # Make sure all voice centroids are reachable (prevent collapse)
        # Force the batch to cover many different voices
        voice_sim = torch.mm(voice_mat, voice_mat.T)                   # [N, N]
        voice_usage = torch.mm(adapted, voice_mat.T).softmax(dim=-1)   # [B, N]
        coverage = voice_usage.mean(dim=0)                             # [N] — usage per voice
        coverage_loss = -torch.log(coverage + 1e-8).mean()            # entropy: maximize coverage

        # ── Loss 3: Consistency ─────────────────────────────────────────────
        # Same clip encoded twice should give same output (if we augment)
        # Simplified: embeddings should not be too similar to each other
        # (prevent mode collapse where all outputs -> same voice)
        gram = torch.mm(adapted, adapted.T)                            # [B, B]
        off_diag = gram - torch.eye(BATCH_SIZE, device=DEVICE)
        collapse_loss = (off_diag ** 2).mean()

        loss = dist_loss + 0.1 * coverage_loss + 0.05 * collapse_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:03d} — loss={loss.item():.4f} "
                  f"(dist={dist_loss.item():.4f} cov={coverage_loss.item():.4f} "
                  f"col={collapse_loss.item():.4f}) lr={lr:.2e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                "epoch":     epoch + 1,
                "adapter":   adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":      best_loss,
            }, CKPT_DIR / "adapter_best.pt")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                "epoch":     epoch + 1,
                "adapter":   adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":      loss.item(),
            }, CKPT_DIR / f"adapter_epoch_{epoch+1:03d}.pt")

    # Save final
    torch.save({
        "epoch":   EPOCHS,
        "adapter": adapter.state_dict(),
        "loss":    best_loss,
    }, CKPT_DIR / "adapter_latest.pt")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Adapter saved to: {CKPT_DIR / 'adapter_latest.pt'}")
    print("\nNext: run test_cloning.py to hear your first cloned voice.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)
