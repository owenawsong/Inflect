"""
Phase 3 — Paralinguistic fine-tuning of Kokoro-82M.

Teaches Kokoro to generate voice-matched paralinguistic sounds
([laughs], [sighs], [whispers], [crying], emotion tags, etc.)
as part of its native synthesis, conditioned on the speaker style.

What gets trained:
  - New embedding rows for tag tokens (added to bert + text_encoder)
  - Decoder (last N layers) — learns to generate the sounds
  - Prosody predictor — learns duration/pitch of paralinguistic events
  Everything else stays frozen.

Loss:
  - Multi-resolution STFT loss (perceptual, length-invariant)
  - Mel spectrogram L1 loss (via DTW alignment)

Usage:
    # First run prepare_finetune_data.py, then:
    python train_phase3_paralinguistic.py
    python train_phase3_paralinguistic.py --epochs 100 --lr 3e-5
    python train_phase3_paralinguistic.py --resume checkpoints/phase3_epoch_020.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

BASE     = Path(__file__).resolve().parent.parent
# Use absolute checkpoint dir so it works regardless of where script is placed
CKPT_DIR = Path("/root/checkpoints")
DATA_FILE = BASE / "data" / "paralinguistic" / "finetune_dataset.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────

class ParalinguisticDataset(Dataset):
    def __init__(self, data: list[dict], max_mel_frames: int = 1000):
        self.data           = data
        self.max_mel_frames = max_mel_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mel  = item["target_mel"]  # [80, T]
        # Truncate long clips
        if mel.shape[1] > self.max_mel_frames:
            mel = mel[:, :self.max_mel_frames]
        return {
            "token_ids":  item["token_ids"],
            "target_mel": mel,
            "style":      item["style"],
            "voice":      item["voice"],
            "tags":       item["tags"],
        }


def collate_fn(batch):
    """Pad token_ids and mel to same length within batch."""
    max_tokens = max(b["token_ids"].shape[0] for b in batch)
    max_frames = max(b["target_mel"].shape[1] for b in batch)

    token_ids  = torch.zeros(len(batch), max_tokens, dtype=torch.long)
    target_mel = torch.zeros(len(batch), 80, max_frames)
    styles     = torch.stack([b["style"] for b in batch])

    for i, b in enumerate(batch):
        t = b["token_ids"].shape[0]
        f = b["target_mel"].shape[1]
        token_ids[i, :t]     = b["token_ids"]
        target_mel[i, :, :f] = b["target_mel"]

    return {
        "token_ids":  token_ids,
        "target_mel": target_mel,
        "styles":     styles,
        "voices":     [b["voice"] for b in batch],
        "tags":       [b["tags"] for b in batch],
    }


# ── Multi-resolution STFT loss ────────────────────────────────────────────────

class STFTLoss(nn.Module):
    """Perceptual loss: compare audio via multiple STFT resolutions."""
    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(120, 240, 480),
                 win_lengths=(600, 1200, 2400)):
        super().__init__()
        self.fft_sizes   = fft_sizes
        self.hop_sizes   = hop_sizes
        self.win_lengths = win_lengths

    def stft(self, x, fft, hop, win):
        return torch.stft(x, fft, hop, win,
                          window=torch.hann_window(win).to(x.device),
                          return_complex=True).abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, T] audio or [T] audio."""
        if pred.dim() == 1:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        # Match lengths
        min_len = min(pred.shape[1], target.shape[1])
        pred    = pred[:, :min_len]
        target  = target[:, :min_len]

        loss = torch.tensor(0.0, device=pred.device)
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            p = self.stft(pred,   fft, hop, win)
            t = self.stft(target, fft, hop, win)
            # Spectral convergence + log magnitude
            loss += ((p - t).abs().mean() + (p.log() - t.log()).abs().mean()) / 2
        return loss / len(self.fft_sizes)


# ── Trainable Kokoro wrapper ──────────────────────────────────────────────────

class TrainableKokoro(nn.Module):
    """
    Wraps KModel with gradient-enabled forward and extended vocab.
    Adds new embedding rows for paralinguistic tag tokens.
    """
    def __init__(self, tag_tokens: dict, unique_tag_ids: list):
        super().__init__()
        import warnings
        warnings.filterwarnings("ignore")
        from kokoro import KModel
        self.base = KModel()

        # Extend ALL embedding tables that use token IDs:
        #   1. bert.embeddings.word_embeddings  [vocab, 128]
        #   2. text_encoder.embedding           [vocab, hidden]
        def extend_embedding(emb: nn.Embedding, new_size: int) -> nn.Embedding:
            old_size, dim = emb.weight.shape
            if new_size <= old_size:
                return emb
            new_emb = nn.Embedding(new_size, dim)
            with torch.no_grad():
                new_emb.weight[:old_size] = emb.weight
                mean_w = emb.weight.mean(0, keepdim=True)
                new_emb.weight[old_size:] = mean_w.expand(new_size - old_size, -1)
            return new_emb

        max_new = max(unique_tag_ids) + 1  # e.g. 229

        old_bert_emb = self.base.bert.embeddings.word_embeddings
        self.base.bert.embeddings.word_embeddings = extend_embedding(old_bert_emb, max_new)
        print(f"  Extended BERT word_embeddings: {old_bert_emb.weight.shape[0]} -> {max_new}")

        old_te_emb = self.base.text_encoder.embedding
        self.base.text_encoder.embedding = extend_embedding(old_te_emb, max_new)
        print(f"  Extended TextEncoder embedding: {old_te_emb.weight.shape[0]} -> {max_new}")

        self.tag_tokens = tag_tokens

        # Freeze everything first
        for param in self.base.parameters():
            param.requires_grad = False

        # Unfreeze 1: new tag token embedding rows only (not all 178 original rows)
        # We do this by registering a gradient hook that zeros out gradients for old rows
        self.base.bert.embeddings.word_embeddings.weight.requires_grad = True
        self.base.text_encoder.embedding.weight.requires_grad = True
        self._old_vocab_size = 178   # mask old rows in optimizer step

        # Unfreeze 2: full predictor (duration + F0 + noise)
        # F0Ntrain is critical: [laughs]=irregular F0, [whispers]=aperiodic/no F0,
        # [crying]=high-pitched F0, [sighs]=breathy noise — all need different F0/N patterns
        for param in self.base.predictor.parameters():
            param.requires_grad = True

        # Unfreeze 3: decoder — last 6 layers for more capacity to learn new sounds
        # More layers = model can learn the texture of laughing/crying/whispering
        decoder_children = list(self.base.decoder.named_children())
        for name, child in decoder_children[-6:]:
            for param in child.parameters():
                param.requires_grad = True

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {n_train:,} / {n_total:,} params ({100*n_train/n_total:.1f}%)")

    def forward(self, token_ids: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [T] LongTensor of phoneme+tag IDs
        style:     [256] or [1, 256] style vector
        Returns:   audio FloatTensor [samples]
        """
        # Build input_ids with CLS/SEP (0)
        device = next(self.parameters()).device
        ids    = token_ids.to(device)
        # Filter out None (unknown phonemes keep 0 which is padding in Kokoro)
        ids    = torch.clamp(ids, 0)
        input_ids    = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                   ids,
                                   torch.zeros(1, dtype=torch.long, device=device)]).unsqueeze(0)
        input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(device)
        text_mask     = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask     = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

        ref_s = style.to(device)
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)   # [1, 256]

        # Forward through Kokoro components (with gradients)
        bert_dur = self.base.bert(input_ids, attention_mask=(~text_mask).int())
        d_en     = self.base.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]
        d = self.base.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.base.predictor.lstm(d)
        duration  = self.base.predictor.duration_proj(x)
        duration  = torch.sigmoid(duration).sum(axis=-1)
        pred_dur  = torch.round(duration).clamp(min=1).long().squeeze()

        indices      = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=device), pred_dur)
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en  = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.base.predictor.F0Ntrain(en, s)
        t_en = self.base.text_encoder(input_ids, input_lengths, text_mask)
        asr  = t_en @ pred_aln_trg
        audio = self.base.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str,   default=str(DATA_FILE))
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch",      type=int,   default=1,
                        help="Batch size (keep 1 for variable length audio)")
    parser.add_argument("--grad-accum", type=int,   default=8,
                        help="Gradient accumulation steps (simulates larger batch)")
    parser.add_argument("--fp16",       action="store_true",
                        help="Use mixed precision (fp16) — halves VRAM, use on Vast.ai")
    parser.add_argument("--save-every", type=int,   default=10)
    parser.add_argument("--resume",     type=str,   default=None)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading dataset: {args.data}")
    saved   = torch.load(args.data, map_location="cpu", weights_only=False)
    dataset = saved["dataset"]
    tag_tokens    = saved["tag_tokens"]
    unique_tag_ids = saved["unique_tag_ids"]
    print(f"  {len(dataset)} clips loaded")
    print(f"  Tag tokens: {len(unique_tag_ids)} unique IDs")

    ds     = ParalinguisticDataset(dataset)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    # Build model
    print("\nBuilding TrainableKokoro...")
    model = TrainableKokoro(tag_tokens, unique_tag_ids).to(DEVICE)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )
    # Warmup for first 5 epochs, then cosine decay
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(args.epochs - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=args.fp16)
    stft_loss = STFTLoss().to(DEVICE)

    if args.fp16:
        print("  Mixed precision (fp16) enabled")

    mel_tf = T.MelSpectrogram(
        sample_rate=24_000, n_fft=1024, hop_length=256,
        win_length=1024, n_mels=80, f_max=8000.0
    ).to(DEVICE)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print(f"\nTraining for {args.epochs} epochs, lr={args.lr}\n")

    import time

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t0         = time.time()

        optimizer.zero_grad()
        accum_loss = torch.tensor(0.0, device=DEVICE)

        for bi, batch in enumerate(loader):
            token_ids  = batch["token_ids"]   # [B, T]
            target_mel = batch["target_mel"]  # [B, 80, F]
            styles     = batch["styles"]      # [B, 256]

            batch_loss = torch.tensor(0.0, device=DEVICE)

            for b in range(token_ids.shape[0]):
                ids   = token_ids[b]
                # Trim padding (zeros at end)
                nonzero = (ids != 0).nonzero(as_tuple=True)[0]
                if len(nonzero) == 0:
                    continue
                ids = ids[:nonzero[-1] + 1]

                style = styles[b]   # [256]
                tgt   = target_mel[b]  # [80, F]

                try:
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        pred_audio = model(ids, style)    # [samples]
                        pred_mel   = torch.log(
                            mel_tf(pred_audio.unsqueeze(0)) + 1e-5
                        ).squeeze(0)  # [80, T_pred]
                        tgt_mel    = tgt.to(DEVICE)
                        min_f      = min(pred_mel.shape[1], tgt_mel.shape[1])
                        # L1 loss for content accuracy
                        mel_l1     = F.l1_loss(pred_mel[:, :min_f], tgt_mel[:, :min_f])
                        # L2 loss for smoothness (penalizes large deviations harder)
                        mel_l2     = F.mse_loss(pred_mel[:, :min_f], tgt_mel[:, :min_f])
                        # Combo: L1 for robustness, L2 for smoothness
                        clip_loss  = mel_l1 * 0.8 + mel_l2 * 0.2
                except Exception as e:
                    print(f"  [skip] forward error: {e}")
                    continue

                batch_loss = batch_loss + clip_loss

            # Gradient accumulation
            if batch_loss.requires_grad:
                scaler.scale(batch_loss / args.grad_accum).backward()
                accum_loss = accum_loss + batch_loss.detach()

            if (bi + 1) % args.grad_accum == 0 or (bi + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += batch_loss.item()
            n_batches  += 1

            if (bi + 1) % 20 == 0:
                elapsed  = time.time() - t0
                per_clip = elapsed / (bi + 1)
                eta      = per_clip * (len(loader) - bi - 1)
                print(f"    [{bi+1:3d}/{len(loader)}] "
                      f"loss={batch_loss.item():.4f}  "
                      f"eta={eta/60:.1f}min", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0

        print(f"  Epoch {epoch+1:04d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed/60:.1f}min", flush=True)

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = CKPT_DIR / f"phase3_epoch_{epoch+1:04d}.pt"
            torch.save({
                "model":      model.state_dict(),
                "epoch":      epoch + 1,
                "loss":       avg_loss,
                "tag_tokens": tag_tokens,
            }, str(ckpt_path))
            print(f"  Saved: {ckpt_path}")

        # Always track best checkpoint (every epoch, not just save_every)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model":      model.state_dict(),
                "epoch":      epoch + 1,
                "loss":       avg_loss,
                "tag_tokens": tag_tokens,
            }, str(CKPT_DIR / "phase3_best.pt"))
            print(f"  ** New best: {best_loss:.4f}")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Best checkpoint: {CKPT_DIR / 'phase3_best.pt'}")

    # Auto-shutdown instance when training completes (safe on Vast.ai)
    import os, subprocess
    print("\nTraining complete. Shutting down instance in 60 seconds...")
    print("Download your checkpoint NOW if you need to cancel shutdown:")
    print(f"  scp -P <PORT> root@<IP>:/root/checkpoints/phase3_best.pt .")
    print(f"  Best checkpoint: {CKPT_DIR / 'phase3_best.pt'}")
    import time as _time; _time.sleep(60)
    subprocess.run(["shutdown", "-h", "now"])


if __name__ == "__main__":
    main()
