"""
Inflect Enhancer — Loss Functions

MultiResolutionSTFTLoss  : spectral convergence + log-mag L1 at multiple resolutions
MelReconstructionLoss    : L1 on log-mel spectrograms
HingeAdversarialLoss     : standard GAN hinge loss
FeatureMatchingLoss      : discriminator feature matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── STFT helpers ──────────────────────────────────────────────────────────────

def _stft(wav: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """Return magnitude spectrogram [B, F, T]."""
    window = torch.hann_window(win, device=wav.device)
    S = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        return_complex=True,
    )
    return S.abs()  # [B, F, T]


def _spectral_convergence(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """||target - pred||_F / ||target||_F"""
    return torch.norm(target - pred, p="fro") / (torch.norm(target, p="fro") + 1e-8)


def _log_mag_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(torch.log(pred + 1e-7), torch.log(target + 1e-7))


# ── Multi-Resolution STFT Loss ────────────────────────────────────────────────

class MultiResolutionSTFTLoss(nn.Module):
    """
    Spectral convergence + log-magnitude L1 at multiple STFT resolutions.
    Standard loss for training neural vocoders (UnivNet, HiFiGAN).
    """

    RESOLUTIONS = [
        # (n_fft, hop, win)
        (2048, 512,  2048),
        (1024, 256,  1024),
        (512,  128,  512),
    ]

    def __init__(self, resolutions=None):
        super().__init__()
        self.resolutions = resolutions or self.RESOLUTIONS

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        """
        pred_wav, target_wav: [B, T] waveforms
        Returns: scalar loss
        """
        sc_loss  = 0.0
        mag_loss = 0.0
        for n_fft, hop, win in self.resolutions:
            pred_mag   = _stft(pred_wav,   n_fft, hop, win)
            target_mag = _stft(target_wav, n_fft, hop, win)
            sc_loss  += _spectral_convergence(pred_mag, target_mag)
            mag_loss += _log_mag_l1(pred_mag, target_mag)

        n = len(self.resolutions)
        return sc_loss / n + mag_loss / n


# ── Mel Reconstruction Loss ───────────────────────────────────────────────────

class MelReconstructionLoss(nn.Module):
    """L1 on log-mel spectrograms. Used in Stage 1 (IRMAE) and as auxiliary loss."""

    def forward(self, pred_mel: torch.Tensor, target_mel: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred_mel, target_mel: [B, n_mels, T] or [B, T, n_mels]
        mask: [B, T] bool, True = valid frames (optional)
        """
        loss = F.l1_loss(pred_mel, target_mel, reduction="none")
        if mask is not None:
            # loss shape [B, n_mels, T] or [B, T, n_mels]
            if loss.dim() == 3 and loss.shape[-1] != mask.shape[-1]:
                mask = mask.unsqueeze(1)  # [B, 1, T]
            else:
                mask = mask.unsqueeze(-1)  # [B, T, 1]
            loss = loss * mask.float()
            return loss.sum() / (mask.float().sum() * loss.shape[-2] + 1e-8)
        return loss.mean()


# ── Adversarial Losses ────────────────────────────────────────────────────────

class HingeAdversarialLoss(nn.Module):
    """Hinge GAN loss for discriminator and generator."""

    def discriminator_loss(self, real_logits: list, fake_logits: list) -> torch.Tensor:
        loss = 0.0
        for r, f in zip(real_logits, fake_logits):
            loss += F.relu(1.0 - r).mean()
            loss += F.relu(1.0 + f).mean()
        return loss / len(real_logits)

    def generator_loss(self, fake_logits: list) -> torch.Tensor:
        loss = 0.0
        for f in fake_logits:
            loss += -f.mean()
        return loss / len(fake_logits)

    def forward(self, real_logits, fake_logits, mode="discriminator"):
        if mode == "discriminator":
            return self.discriminator_loss(real_logits, fake_logits)
        return self.generator_loss(fake_logits)


class FeatureMatchingLoss(nn.Module):
    """
    L1 loss on intermediate discriminator features.
    Encourages generator to produce features matching real audio.
    """

    def forward(self, real_features: list[list], fake_features: list[list]) -> torch.Tensor:
        loss = 0.0
        n = 0
        for real_feats, fake_feats in zip(real_features, fake_features):
            for r, f in zip(real_feats, fake_feats):
                loss += F.l1_loss(f, r.detach())
                n += 1
        return loss / max(n, 1)
