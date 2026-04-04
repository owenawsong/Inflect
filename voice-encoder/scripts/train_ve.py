"""
Inflect Voice Encoder - Stage 1A: GE2E Speaker Verification Training

Architecture:
    MelEncoder: Conv1D stack → BiGRU → Attention Pool → [B, 512]
    StyleProjector: Linear → reshape → [B, 511, 1, 256] (Kokoro format)
    GE2E loss: N speakers × M utterances per batch

Usage:
    python train_ve.py
    python train_ve.py --resume checkpoints/ve_epoch_010.pt
    python train_ve.py --epochs 200 --batch-speakers 32

~30h on RTX 3060. Train overnight.
"""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
MANIFEST  = BASE / "data" / "manifest.csv"
CKPT_DIR  = BASE / "checkpoints"
LOG_DIR   = BASE / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_MELS         = 80
EMBED_DIM      = 256    # final embedding dim (matches Kokoro style dim)
PROJ_TIMESTEPS = 511    # Kokoro style tensor timesteps
N_SPEAKERS     = 64     # speakers per GE2E batch
M_UTTERANCES   = 10     # utterances per speaker
LR             = 1e-3
LR_MIN         = 1e-5
WARMUP_STEPS   = 2_000
TOTAL_EPOCHS   = 300
SAVE_EVERY     = 10
LOG_EVERY      = 50     # steps
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Model ─────────────────────────────────────────────────────────────────────
class MelEncoder(nn.Module):
    """
    log-mel [n_mels, T] → speaker embedding [512]

    Conv stack downsamples time, BiGRU captures long-range,
    attention pooling gives fixed-size vector.
    """
    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        # Conv stack: 80 → 256 → 512 → 512, stride=2 each (8x time reduction)
        self.conv = nn.Sequential(
            self._conv_block(n_mels, 256),
            self._conv_block(256,    512),
            self._conv_block(512,    512),
            self._conv_block(512,    512),
        )
        # BiGRU
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=384,   # 384*2 = 768 bidirectional
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        gru_out = 768
        # Attention pooling
        self.attn_w = nn.Linear(gru_out, 1)
        # Project to 512
        self.proj = nn.Linear(gru_out, 512)

    @staticmethod
    def _conv_block(in_c: int, out_c: int, kernel: int = 5, stride: int = 2):
        pad = kernel // 2
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, stride=stride, padding=pad),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_mels, T]
        returns: [B, 512]
        """
        x = self.conv(x)          # [B, 512, T']
        x = x.permute(0, 2, 1)   # [B, T', 512]
        x, _ = self.gru(x)        # [B, T', 768]
        # Attention pool
        a = self.attn_w(x).squeeze(-1)   # [B, T']
        a = torch.softmax(a, dim=-1)
        x = (x * a.unsqueeze(-1)).sum(dim=1)  # [B, 768]
        return self.proj(x)       # [B, 512]


class StyleProjector(nn.Module):
    """
    [B, 512] → [B, 511, 1, 256]  (Kokoro voice pack format)

    Kokoro's 511 time steps encode overall speaking style — they're nearly
    identical across the sequence. We project to a single 256-dim vector and
    broadcast across all 511 steps. This keeps the model at ~8M params.
    """
    def __init__(self, in_dim: int = 512, timesteps: int = PROJ_TIMESTEPS, style_dim: int = EMBED_DIM):
        super().__init__()
        self.timesteps = timesteps
        self.style_dim = style_dim
        self.linear = nn.Linear(in_dim, style_dim)   # 512 → 256 only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 512] → [B, 511, 1, 256]"""
        x = self.linear(x)                          # [B, 256]
        x = F.normalize(x, dim=-1)                  # L2-normalize
        x = x.unsqueeze(1).unsqueeze(2)             # [B, 1, 1, 256]
        x = x.expand(-1, self.timesteps, 1, -1)     # [B, 511, 1, 256]
        return x.contiguous()


class VoiceEncoder(nn.Module):
    """Full encoder: mel → Kokoro-format style tensor"""
    def __init__(self):
        super().__init__()
        self.mel_encoder  = MelEncoder()
        self.style_proj   = StyleProjector()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, 80, T] → style: [B, 511, 1, 256]"""
        emb = self.mel_encoder(mel)   # [B, 512]
        return self.style_proj(emb)   # [B, 511, 1, 256]

    def embed(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, 80, T] → L2-normalized 256-dim embedding (from first timestep)"""
        style = self.forward(mel)             # [B, 511, 1, 256]
        return F.normalize(style[:, 0, 0, :], dim=-1)  # [B, 256]


# ── GE2E Loss ─────────────────────────────────────────────────────────────────
class GE2ELoss(nn.Module):
    """
    Generalized End-to-End Speaker Verification Loss (Wan et al. 2018)

    Batch: [N*M, D] where N=speakers, M=utterances per speaker
    Each utterance is compared to centroids of all speakers.
    """
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeds: torch.Tensor, n_speakers: int, m_utterances: int) -> torch.Tensor:
        """
        embeds: [N*M, D] L2-normalized
        Returns: scalar loss
        """
        N, M, D = n_speakers, m_utterances, embeds.shape[-1]
        embeds = embeds.view(N, M, D)  # [N, M, D]

        # Centroids: [N, D]
        centroids = embeds.mean(dim=1)
        centroids = F.normalize(centroids, dim=-1)

        # Exclusive centroids (leave-one-out per utterance)
        # c_ji = mean of speaker i's utterances EXCLUDING utterance j
        # Faster: sum - utt, divide by (M-1)
        sums = embeds.sum(dim=1, keepdim=True)      # [N, 1, D]
        excl = (sums - embeds) / (M - 1)            # [N, M, D]
        excl = F.normalize(excl, dim=-1)

        # Similarity matrix: for each (speaker i, utterance j), sim to all centroids
        # Use exclusive centroid for (i, j) → centroid of same speaker
        # Use regular centroid for other speakers
        sim = torch.zeros(N, M, N, device=embeds.device)
        for i in range(N):
            for j in range(M):
                # Use exclusive centroid for same speaker (i,i)
                # Use regular centroid for other speakers
                utt = embeds[i, j]  # [D]
                for k in range(N):
                    if k == i:
                        c = excl[i, j]  # exclusive centroid
                    else:
                        c = centroids[k]
                    sim[i, j, k] = torch.dot(utt, c)

        # Scale by learnable w, b
        sim = self.w.abs() * sim + self.b

        # Loss: for each utterance (i,j), correct class is i
        # Reshape: [N*M, N]
        sim_flat = sim.view(N * M, N)
        labels = torch.arange(N, device=embeds.device).unsqueeze(1).expand(N, M).reshape(-1)
        return F.cross_entropy(sim_flat, labels)


class GE2ELossFast(nn.Module):
    """
    Vectorized GE2E loss — no Python loops, runs fast on GPU.
    Equivalent to GE2ELoss but ~10x faster.
    """
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeds: torch.Tensor, n_speakers: int, m_utterances: int) -> torch.Tensor:
        N, M, D = n_speakers, m_utterances, embeds.shape[-1]
        embeds = embeds.view(N, M, D)

        centroids = F.normalize(embeds.mean(dim=1), dim=-1)  # [N, D]

        # Exclusive centroids: [N, M, D]
        sums = embeds.sum(dim=1, keepdim=True)
        excl = F.normalize((sums - embeds) / max(M - 1, 1), dim=-1)  # [N, M, D]

        # Similarity to all centroids: [N, M, N]
        # embeds: [N, M, D], centroids: [N, D]
        sim = torch.einsum("nmd,kd->nmk", embeds, centroids)  # [N, M, N]

        # Replace diagonal with exclusive centroid similarity
        # diag: for speaker i, utterance j → sim to excl[i,j] vs centroid[i]
        excl_sim = (embeds * excl).sum(dim=-1)   # [N, M] — sim to own exclusive centroid
        # Replace sim[i, j, i] with excl_sim[i, j]
        idx = torch.arange(N, device=embeds.device)
        sim[idx, :, idx] = excl_sim

        sim = self.w.abs() * sim + self.b

        # [N*M, N] — target for each utterance is its speaker index
        sim_flat = sim.view(N * M, N)
        labels = idx.unsqueeze(1).expand(N, M).reshape(-1)
        return F.cross_entropy(sim_flat, labels)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SpeakerDataset(Dataset):
    """
    For GE2E: returns M random mel clips for a given speaker.
    __getitem__ returns a list of M mel tensors (variable length along T).
    """
    def __init__(self, manifest_path: Path, min_clips: int = 5):
        df = pd.read_csv(manifest_path)

        # Group by speaker, keep speakers with enough clips
        self.speakers: list[list[str]] = []
        self.speaker_ids: list[str] = []
        for spk_id, group in df.groupby("speaker_id"):
            paths = group["mel_path"].tolist()
            if len(paths) >= min_clips:
                self.speakers.append(paths)
                self.speaker_ids.append(spk_id)

        print(f"Dataset: {len(self.speakers)} speakers with >={min_clips} clips each")

    def __len__(self) -> int:
        return len(self.speakers)

    def get_clips(self, idx: int, m: int) -> list[torch.Tensor]:
        """Return m random mel clips for speaker idx."""
        paths = self.speakers[idx]
        chosen = random.choices(paths, k=m)
        mels = []
        for p in chosen:
            try:
                mel = torch.load(p, weights_only=True)
                mels.append(mel)
            except Exception:
                # Replace with zeros if file is corrupt
                mels.append(torch.zeros(N_MELS, 200))
        return mels


def pad_and_chunk(mels: list[torch.Tensor], chunk_frames: int = 160) -> torch.Tensor:
    """
    Take a list of mel tensors, randomly crop/pad each to chunk_frames.
    Returns [len(mels), n_mels, chunk_frames].
    """
    out = []
    for mel in mels:
        T = mel.shape[1]
        if T >= chunk_frames:
            start = random.randint(0, T - chunk_frames)
            mel = mel[:, start:start + chunk_frames]
        else:
            pad_amt = chunk_frames - T
            mel = F.pad(mel, (0, pad_amt))
        out.append(mel)
    return torch.stack(out)  # [M, n_mels, chunk_frames]


def ge2e_collate(batch_speakers: list[list[torch.Tensor]]) -> torch.Tensor:
    """
    batch_speakers: list of N lists, each with M mels
    Returns: [N*M, n_mels, T] — ready for GE2E
    """
    all_chunks = []
    for speaker_mels in batch_speakers:
        chunks = pad_and_chunk(speaker_mels)  # [M, n_mels, T]
        all_chunks.append(chunks)
    return torch.cat(all_chunks, dim=0)  # [N*M, n_mels, T]


# ── Training ──────────────────────────────────────────────────────────────────
def cosine_lr(step: int, total_steps: int, warmup: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    t = (step - warmup) / max(total_steps - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def save_ckpt(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "loss":       loss,
    }, path)


def load_ckpt(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("loss", float("inf"))


def train(args):
    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = SpeakerDataset(MANIFEST, min_clips=args.m_utterances)
    n_speakers_total = len(dataset)
    print(f"Total speakers available: {n_speakers_total}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = VoiceEncoder().to(DEVICE)
    criterion = GE2ELossFast().to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LR, weight_decay=1e-5
    )

    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_ckpt(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")

    # ── Training loop ─────────────────────────────────────────────────────────
    N = args.n_speakers
    M = args.m_utterances
    steps_per_epoch = max(1, n_speakers_total // N)
    total_steps = TOTAL_EPOCHS * steps_per_epoch

    log_path = LOG_DIR / "ge2e_train.log"
    log_f = open(log_path, "a")

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        # Shuffle speakers each epoch
        spk_indices = list(range(n_speakers_total))
        random.shuffle(spk_indices)

        # Build batches of N speakers
        for batch_start in range(0, n_speakers_total - N, N):
            batch_spk_indices = spk_indices[batch_start:batch_start + N]

            # Load M mels per speaker
            batch_speaker_mels = [dataset.get_clips(i, M) for i in batch_spk_indices]

            # Stack: [N*M, n_mels, T]
            batch = ge2e_collate(batch_speaker_mels).to(DEVICE)

            # LR schedule
            lr = cosine_lr(global_step, total_steps, WARMUP_STEPS, LR, LR_MIN)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            optimizer.zero_grad()
            embeds = model.embed(batch)             # [N*M, 256]
            loss = criterion(embeds, N, M)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                avg = epoch_loss / epoch_steps
                msg = f"epoch={epoch:03d} step={global_step:06d} loss={loss.item():.4f} avg={avg:.4f} lr={lr:.2e}"
                print(f"\r{msg}", end="", flush=True)
                log_f.write(msg + "\n")
                log_f.flush()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch:03d} done — avg_loss={avg_loss:.4f}")

        if (epoch + 1) % SAVE_EVERY == 0 or epoch == TOTAL_EPOCHS - 1:
            ckpt_path = CKPT_DIR / f"ve_epoch_{epoch+1:03d}.pt"
            save_ckpt(model, optimizer, epoch + 1, avg_loss, ckpt_path)
            print(f"  Saved: {ckpt_path}")

            # Also save as "latest" for easy resume
            save_ckpt(model, optimizer, epoch + 1, avg_loss, CKPT_DIR / "ve_latest.pt")

    log_f.close()
    print(f"\nTraining complete. Final checkpoint: {CKPT_DIR / 've_latest.pt'}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",        type=str,  default=None)
    parser.add_argument("--epochs",        type=int,  default=TOTAL_EPOCHS)
    parser.add_argument("--n-speakers",    type=int,  default=N_SPEAKERS,   dest="n_speakers")
    parser.add_argument("--m-utterances",  type=int,  default=M_UTTERANCES, dest="m_utterances")
    args = parser.parse_args()

    TOTAL_EPOCHS = args.epochs
    train(args)
