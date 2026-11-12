"""Frozen original0.1.1 loss arithmetic, used only for equivalence tests."""
import torch
from torch.nn import functional as F

def reference_spectral_loss(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError('Waveform and target clocks must match')
    total = predicted.new_zeros(())
    for size in (256, 512, 1024):
        window = torch.hann_window(size, device=predicted.device)
        a = torch.stft(predicted.float(), size, hop_length=size//4, window=window, return_complex=True).abs()
        b = torch.stft(target.float(), size, hop_length=size//4, window=window, return_complex=True).abs()
        total = total + F.l1_loss(torch.log(a.clamp_min(1e-5)), torch.log(b.clamp_min(1e-5)))
        total = total + torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-5)
    return total / 3
