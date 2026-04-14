"""
Inflect Enhancer — Training Script

Three staged training phases:
  Stage 1: IRMAE pretraining (mel autoencoder on clean audio)
  Stage 2: CFM training (noisy → clean mel, frozen IRMAE encoder)
  Stage 3: UnivNet vocoder training (clean mel → waveform, adversarial)

Usage (run from repo root):
  python inflect/enhancer/train.py --stage 1
  python inflect/enhancer/train.py --stage 2
  python inflect/enhancer/train.py --stage 3

Checkpoints: inflect/enhancer/checkpoints/
"""

import sys
import time
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inflect.enhancer.configs.base import EnhancerConfig
from inflect.enhancer.model import (
    EnhancerModel, IRMAEEncoder, IRMAEDecoder,
    CFMEnhancer, UnivNetGenerator, UnivNetDiscriminator,
    MelExtractor, count_params,
)
from inflect.enhancer.losses import (
    MelReconstructionLoss, MultiResolutionSTFTLoss,
    HingeAdversarialLoss, FeatureMatchingLoss,
)
from inflect.enhancer.dataset import EnhancerDataset, collate_fn

# ── Config ────────────────────────────────────────────────────────────────────
cfg = EnhancerConfig()

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR  = PROJECT_ROOT / "inflect/enhancer/checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Training data paths — update these before running
MANIFEST_PATH = PROJECT_ROOT / "outputs/enhancer_pairs/manifest.csv"

LOG_EVERY  = cfg.log_every
SAVE_EVERY = cfg.save_every
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    return base_lr


def _make_loaders(manifest: Path, val_frac: float = 0.05):
    ds = EnhancerDataset(
        manifest,
        sample_rate=cfg.sample_rate,
        clip_seconds=cfg.clip_seconds,
        augment=True,
    )
    n_val  = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ── Stage 1: IRMAE pretraining ────────────────────────────────────────────────

def train_stage1(manifest: Path):
    print(f"\n[Stage 1] IRMAE pretraining — {cfg.stage1_steps} steps")
    set_seed(cfg.seed)

    mel_fn  = MelExtractor(cfg).to(DEVICE)
    encoder = IRMAEEncoder(cfg).to(DEVICE)
    decoder = IRMAEDecoder(cfg).to(DEVICE)
    mel_loss = MelReconstructionLoss()

    print(f"  encoder: {count_params(encoder):,}  decoder: {count_params(decoder):,}")

    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler()

    train_loader, val_loader = _make_loaders(manifest)

    step = 0
    best_val = float("inf")
    losses   = defaultdict(float)
    t0       = time.time()

    while step < cfg.stage1_steps:
        encoder.train()
        decoder.train()
        for batch in train_loader:
            if step >= cfg.stage1_steps:
                break

            # LR warmup
            lr = get_lr(step, cfg.warmup_steps, cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            wav_clean = batch["wav_clean"].to(DEVICE)  # [B, T]
            with torch.amp.autocast(device_type="cuda"):
                mel = mel_fn(wav_clean)                # [B, n_mels, T_frames]
                z   = encoder(mel)                     # [B, Z, T_frames]
                mel_recon = decoder(z)                 # [B, n_mels, T_frames]
                loss = mel_loss(mel_recon, mel)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses["mel"] += loss.item()
            step += 1

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                avg = {k: v / LOG_EVERY for k, v in losses.items()}
                print(f"  step {step:>6}  mel={avg['mel']:.4f}  "
                      f"lr={lr:.2e}  {elapsed/60:.1f}min")
                losses = defaultdict(float)

            if step % SAVE_EVERY == 0:
                # Validation
                encoder.eval()
                decoder.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vb in val_loader:
                        wc  = vb["wav_clean"].to(DEVICE)
                        mel = mel_fn(wc)
                        z   = encoder(mel)
                        mr  = decoder(z)
                        val_loss += mel_loss(mr, mel).item()
                val_loss /= max(len(val_loader), 1)
                print(f"  [val] step {step}  mel_loss={val_loss:.4f}")

                ckpt = {
                    "step": step, "cfg": cfg,
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                }
                path = CKPT_DIR / "irmae_latest.pt"
                torch.save(ckpt, path)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(ckpt, CKPT_DIR / "irmae_best.pt")
                    print(f"  ** new best: {best_val:.4f}")

    print(f"\nStage 1 done. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {CKPT_DIR / 'irmae_best.pt'}")


# ── Stage 2: CFM training ─────────────────────────────────────────────────────

def train_stage2(manifest: Path):
    print(f"\n[Stage 2] CFM training — {cfg.stage2_steps} steps")
    set_seed(cfg.seed)

    mel_fn  = MelExtractor(cfg).to(DEVICE)
    encoder = IRMAEEncoder(cfg).to(DEVICE)
    cfm     = CFMEnhancer(cfg).to(DEVICE)

    # Load frozen IRMAE encoder from Stage 1
    irmae_ckpt = CKPT_DIR / "irmae_best.pt"
    if not irmae_ckpt.exists():
        raise FileNotFoundError(f"Run Stage 1 first. Not found: {irmae_ckpt}")
    ckpt = torch.load(irmae_ckpt, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    print(f"  Loaded IRMAE encoder from step {ckpt['step']} (frozen)")
    print(f"  cfm: {count_params(cfm):,}")

    optimizer = torch.optim.AdamW(cfm.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler()

    train_loader, val_loader = _make_loaders(manifest)

    step = 0
    best_val = float("inf")
    losses   = defaultdict(float)
    t0       = time.time()

    while step < cfg.stage2_steps:
        cfm.train()
        for batch in train_loader:
            if step >= cfg.stage2_steps:
                break

            lr = get_lr(step, cfg.warmup_steps, cfg.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            wav_deg   = batch["wav_degraded"].to(DEVICE)
            wav_clean = batch["wav_clean"].to(DEVICE)

            with torch.amp.autocast(device_type="cuda"):
                mel_noisy = mel_fn(wav_deg)
                mel_clean = mel_fn(wav_clean)
                with torch.no_grad():
                    latent = encoder(mel_clean)   # condition on clean latent
                loss = cfm.forward_train(mel_clean, mel_noisy, latent)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses["cfm"] += loss.item()
            step += 1

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                avg = {k: v / LOG_EVERY for k, v in losses.items()}
                print(f"  step {step:>6}  cfm={avg['cfm']:.4f}  "
                      f"lr={lr:.2e}  {elapsed/60:.1f}min")
                losses = defaultdict(float)

            if step % SAVE_EVERY == 0:
                cfm.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vb in val_loader:
                        wd = vb["wav_degraded"].to(DEVICE)
                        wc = vb["wav_clean"].to(DEVICE)
                        mn = mel_fn(wd)
                        mc = mel_fn(wc)
                        lat = encoder(mc)
                        val_loss += cfm.forward_train(mc, mn, lat).item()
                val_loss /= max(len(val_loader), 1)
                print(f"  [val] step {step}  cfm_loss={val_loss:.4f}")

                ckpt = {
                    "step": step, "cfg": cfg,
                    "cfm": cfm.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                }
                torch.save(ckpt, CKPT_DIR / "cfm_latest.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(ckpt, CKPT_DIR / "cfm_best.pt")
                    print(f"  ** new best: {best_val:.4f}")

    print(f"\nStage 2 done. Best val loss: {best_val:.4f}")


# ── Stage 3: UnivNet vocoder ──────────────────────────────────────────────────

def train_stage3(manifest: Path):
    print(f"\n[Stage 3] UnivNet vocoder — {cfg.stage3_steps} steps")
    set_seed(cfg.seed)

    mel_fn    = MelExtractor(cfg).to(DEVICE)
    generator = UnivNetGenerator(cfg).to(DEVICE)
    disc      = UnivNetDiscriminator().to(DEVICE)

    print(f"  generator: {count_params(generator):,}")
    print(f"  disc:      {count_params(disc):,}")

    mrstft_loss = MultiResolutionSTFTLoss()
    adv_loss    = HingeAdversarialLoss()
    fm_loss     = FeatureMatchingLoss()

    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr,   betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(),      lr=cfg.lr,   betas=(0.8, 0.99))
    scaler = torch.amp.GradScaler()

    train_loader, val_loader = _make_loaders(manifest)

    step = 0
    best_val = float("inf")
    losses   = defaultdict(float)
    t0       = time.time()
    USE_ADV  = False  # enable adversarial after 5000 steps

    while step < cfg.stage3_steps:
        generator.train()
        disc.train()

        for batch in train_loader:
            if step >= cfg.stage3_steps:
                break

            if step >= 5000:
                USE_ADV = True

            lr = get_lr(step, cfg.warmup_steps, cfg.lr)
            for g in opt_g.param_groups + opt_d.param_groups:
                g["lr"] = lr

            wav_clean = batch["wav_clean"].to(DEVICE)  # [B, T]

            with torch.amp.autocast(device_type="cuda"):
                mel = mel_fn(wav_clean)           # [B, n_mels, T_mel]
                wav_pred = generator(mel)         # [B, 1, T_audio]
                wav_pred_sq = wav_pred.squeeze(1) # [B, T_audio]

                # Trim to same length
                T = min(wav_pred_sq.shape[-1], wav_clean.shape[-1])
                wp = wav_pred_sq[..., :T]
                wc = wav_clean[..., :T]

                # Generator loss
                loss_mrstft = mrstft_loss(wp, wc)
                loss_g = loss_mrstft

                if USE_ADV:
                    logits_r, feats_r = disc(wc.unsqueeze(1))
                    logits_f, feats_f = disc(wav_pred)
                    loss_g_adv = adv_loss.generator_loss(logits_f)
                    loss_g_fm  = fm_loss(feats_r, feats_f)
                    loss_g     = loss_mrstft + 0.1 * loss_g_adv + 2.0 * loss_g_fm

            opt_g.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(opt_g)

            # Discriminator update
            if USE_ADV:
                with torch.amp.autocast(device_type="cuda"):
                    logits_r, _ = disc(wc.unsqueeze(1))
                    logits_f, _ = disc(wav_pred.detach())
                    loss_d = adv_loss.discriminator_loss(logits_r, logits_f)
                opt_d.zero_grad()
                scaler.scale(loss_d).backward()
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                scaler.step(opt_d)
                losses["disc"] += loss_d.item()

            scaler.update()

            losses["mrstft"] += loss_mrstft.item()
            losses["gen"]    += loss_g.item()
            step += 1

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                avg = {k: v / LOG_EVERY for k, v in losses.items()}
                adv_str = f"  disc={avg.get('disc', 0):.4f}" if USE_ADV else ""
                print(f"  step {step:>6}  mrstft={avg['mrstft']:.4f}"
                      f"{adv_str}  lr={lr:.2e}  {elapsed/60:.1f}min")
                losses = defaultdict(float)

            if step % SAVE_EVERY == 0:
                generator.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vb in val_loader:
                        wc  = vb["wav_clean"].to(DEVICE)
                        mel = mel_fn(wc)
                        wp  = generator(mel).squeeze(1)
                        T   = min(wp.shape[-1], wc.shape[-1])
                        val_loss += mrstft_loss(wp[..., :T], wc[..., :T]).item()
                val_loss /= max(len(val_loader), 1)
                print(f"  [val] step {step}  mrstft={val_loss:.4f}")

                ckpt = {
                    "step": step, "cfg": cfg,
                    "generator": generator.state_dict(),
                    "disc":      disc.state_dict(),
                    "opt_g":     opt_g.state_dict(),
                    "opt_d":     opt_d.state_dict(),
                    "val_loss":  val_loss,
                }
                torch.save(ckpt, CKPT_DIR / "univnet_latest.pt")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(ckpt, CKPT_DIR / "univnet_best.pt")
                    print(f"  ** new best: {best_val:.4f}")

    print(f"\nStage 3 done. Best val loss: {best_val:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",    type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    if not args.manifest.exists() and args.stage in (2, 3):
        print(f"\nManifest not found: {args.manifest}")
        print("Run scripts/generate_enhancer_pairs.py first.")
        return

    if args.stage == 1:
        train_stage1(args.manifest)
    elif args.stage == 2:
        train_stage2(args.manifest)
    elif args.stage == 3:
        train_stage3(args.manifest)


if __name__ == "__main__":
    main()
