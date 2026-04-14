"""
Inflect Enhancer — Model (~24M parameters)

Four components:
  STFTDenoiser    ~2M  optional, off by default for TTS input
  IRMAEEncoder    }
  IRMAEDecoder    } ~8M  mel-domain autoencoder
  CFMEnhancer     ~5M  conditional flow matching mel → mel
  UnivNetGenerator ~9M  mel → waveform at 48kHz

Pipeline:
  wav_in
   └─[optional STFTDenoiser]
   └─ mel extraction
   └─ IRMAEEncoder → latent (48-dim)
   └─ CFMEnhancer(mel_noisy, latent) → mel_clean  (NFE=8, midpoint, temp=0.5)
   └─ UnivNetGenerator(mel_clean) → wav_out (48kHz)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from inflect.enhancer.configs.base import EnhancerConfig


# ── Utility ───────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _mel_filterbank(n_fft: int, n_mels: int, sr: int, f_min: float, f_max: float,
                    device: torch.device) -> torch.Tensor:
    """Build a mel filterbank matrix [n_mels, n_fft//2+1] on device."""
    f_max = f_max or sr / 2.0
    n_freqs = n_fft // 2 + 1
    # Hz → mel
    def hz_to_mel(f): return 2595.0 * math.log10(1.0 + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_lo = hz_to_mel(f_min)
    mel_hi = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_points  = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
    bin_points = torch.floor((n_fft + 1) * hz_points / sr).long()

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        lo, center, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, center):
            if lo != center:
                fb[m - 1, k] = (k - lo) / (center - lo)
        for k in range(center, hi):
            if center != hi:
                fb[m - 1, k] = (hi - k) / (hi - center)
    return fb.to(device)


class MelExtractor(nn.Module):
    """Differentiable log-mel spectrogram (no torchaudio needed)."""

    def __init__(self, cfg: EnhancerConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "window",
            torch.hann_window(cfg.n_fft),
        )
        # filterbank built lazily on first forward (needs device)
        self._fb: torch.Tensor | None = None

    def _get_fb(self, device):
        if self._fb is None or self._fb.device != device:
            self._fb = _mel_filterbank(
                self.cfg.n_fft, self.cfg.n_mels,
                self.cfg.sample_rate,
                self.cfg.f_min, self.cfg.f_max, device,
            )
        return self._fb

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [B, T] → log-mel [B, n_mels, T_frames]"""
        cfg = self.cfg
        S = torch.stft(
            wav,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window=self.window.to(wav.device),
            return_complex=True,
        )  # [B, F, T]
        mag = S.abs() ** 2  # power spectrum
        fb  = self._get_fb(wav.device)  # [n_mels, F]
        mel = torch.matmul(fb, mag)     # [B, n_mels, T]
        log_mel = torch.log(mel.clamp(min=1e-7))
        return log_mel  # [B, n_mels, T]


# ── 1. STFT Denoiser (~2M, optional) ─────────────────────────────────────────

class _UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class STFTDenoiser(nn.Module):
    """
    UNet operating on STFT magnitude frames.
    Projects 1025 freq bins → 256 for compactness, then UNet along time.
    ~2M params.
    """

    def __init__(self, cfg: EnhancerConfig):
        super().__init__()
        F_bins = cfg.n_fft // 2 + 1   # 1025
        P = cfg.denoiser_freq_proj     # 256
        H = cfg.denoiser_hidden        # 32
        L = cfg.denoiser_levels        # 3

        self.freq_in  = nn.Linear(F_bins, P)
        self.freq_out = nn.Linear(P, F_bins)

        # Encoder: P → H*2^0 → H*2^1 → ... → H*2^L
        enc_chs = [P] + [H * (2 ** i) for i in range(L)]
        self.encoders = nn.ModuleList([
            _UNetBlock(enc_chs[i], enc_chs[i + 1]) for i in range(L)
        ])
        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = _UNetBlock(enc_chs[-1], enc_chs[-1])

        # Decoder
        dec_chs = list(reversed(enc_chs))  # H*2^L → ... → P
        self.decoders = nn.ModuleList([
            _UNetBlock(dec_chs[i] * 2, dec_chs[i + 1]) for i in range(L)
        ])

        self.mask_head = nn.Sequential(
            nn.Conv1d(P, P, 1),
            nn.Sigmoid(),   # soft mask ∈ [0, 1]
        )

        self.register_buffer("window", torch.hann_window(cfg.n_fft))
        self.cfg = cfg

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: [B, T] → denoised wav [B, T]"""
        cfg = self.cfg
        window = self.window.to(wav.device)

        S = torch.stft(wav, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                       win_length=cfg.n_fft, window=window, return_complex=True)
        mag   = S.abs()           # [B, F, T_s]
        phase = S / (mag + 1e-8)  # unit-phase complex

        # Freq projection: [B, F, T] → [B, T, F] → Linear → [B, T, P] → [B, P, T]
        x = mag.permute(0, 2, 1)     # [B, T, F]
        x = self.freq_in(x)          # [B, T, P]
        x = x.permute(0, 2, 1)       # [B, P, T]

        # UNet encode
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # UNet decode
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        mask = self.mask_head(x)  # [B, P, T]

        # Map mask back to freq domain
        mask = mask.permute(0, 2, 1)   # [B, T, P]
        mask = self.freq_out(mask)     # [B, T, F]
        mask = mask.permute(0, 2, 1).sigmoid()  # [B, F, T]

        # Apply mask and reconstruct
        S_clean = mag * mask * phase
        wav_out = torch.istft(S_clean, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                              win_length=cfg.n_fft, window=window,
                              length=wav.shape[-1])
        return wav_out


# ── 2. IRMAE (Information-Restricting Mel Autoencoder, ~8M) ──────────────────

class _ResBlock1d(nn.Module):
    """Dilated residual block, 4 convolutions with increasing dilation."""

    DILATIONS = (1, 2, 4, 8)

    def __init__(self, ch: int):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ch, ch, 3, dilation=d, padding=d),
                nn.GroupNorm(min(16, ch), ch),
                nn.GELU(),
            )
            for d in self.DILATIONS
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class IRMAEEncoder(nn.Module):
    """mel [B, 80, T] → latent [B, latent, T]"""

    def __init__(self, cfg: EnhancerConfig):
        super().__init__()
        H, Z, n_mels = cfg.irmae_hidden, cfg.irmae_latent, cfg.n_mels
        self.input_proj = nn.Conv1d(n_mels, H, 3, padding=1)
        self.blocks = nn.ModuleList([_ResBlock1d(H) for _ in range(cfg.irmae_n_blocks)])
        self.output_proj = nn.Conv1d(H, Z, 1)
        self.noise_std = cfg.irmae_noise_std

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mels, T] → latent [B, Z, T]"""
        x = self.input_proj(mel)
        for block in self.blocks:
            x = block(x)
        z = self.output_proj(x)
        z = torch.tanh(z)  # bounded latent
        # Additive noise during training (information restriction)
        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std
        return z


class IRMAEDecoder(nn.Module):
    """latent [B, latent, T] → mel [B, 80, T]"""

    def __init__(self, cfg: EnhancerConfig):
        super().__init__()
        H, Z, n_mels = cfg.irmae_hidden, cfg.irmae_latent, cfg.n_mels
        self.input_proj  = nn.Conv1d(Z, H, 3, padding=1)
        self.blocks      = nn.ModuleList([_ResBlock1d(H) for _ in range(cfg.irmae_n_blocks)])
        self.output_proj = nn.Conv1d(H, n_mels, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, Z, T] → mel [B, n_mels, T]"""
        x = self.input_proj(z)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


# ── 3. CFM Enhancer (~5M) ─────────────────────────────────────────────────────

class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] scalar time → [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb


class _TransformerLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = nn.Sequential(
            nn.Linear(d, d * ffn_mult),
            nn.GELU(),
            nn.Linear(d * ffn_mult, d),
        )
        # FiLM conditioning from time + IRMAE latent
        self.film_scale = nn.Linear(d, d)
        self.film_shift = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, T, d]
        cond: [B, d]  — time + IRMAE conditioning
        """
        scale = self.film_scale(cond).unsqueeze(1)  # [B, 1, d]
        shift = self.film_shift(cond).unsqueeze(1)

        x = x * (1 + scale) + shift          # FiLM modulation
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class CFMEnhancer(nn.Module):
    """
    Conditional Flow Matching for mel-to-mel enhancement.
    Learns vector field v(mel_t, t | latent) → mel_clean.
    ~5M params.
    """

    def __init__(self, cfg: EnhancerConfig):
        super().__init__()
        d = cfg.cfm_hidden
        n_mels = cfg.n_mels
        Z = cfg.irmae_latent

        self.input_proj = nn.Linear(n_mels, d)
        self.time_emb   = _SinusoidalPosEmb(d)
        self.time_proj  = nn.Linear(d, d)
        self.cond_proj  = nn.Linear(Z, d)   # IRMAE latent → conditioning

        self.layers = nn.ModuleList([
            _TransformerLayer(d, cfg.cfm_n_heads, cfg.cfm_ffn_mult)
            for _ in range(cfg.cfm_n_layers)
        ])
        self.output_proj = nn.Linear(d, n_mels)

        self.cfg = cfg

    def _velocity(self, mel_t: torch.Tensor, t: torch.Tensor,
                  latent: torch.Tensor) -> torch.Tensor:
        """
        mel_t:  [B, n_mels, T_frames]
        t:      [B] ∈ [0, 1]
        latent: [B, Z, T_frames] from IRMAEEncoder
        Returns: velocity [B, n_mels, T_frames]
        """
        # mel → [B, T, d]
        x = mel_t.permute(0, 2, 1)   # [B, T, n_mels]
        x = self.input_proj(x)        # [B, T, d]

        # conditioning = time emb + mean-pooled IRMAE latent
        t_emb    = self.time_proj(self.time_emb(t))          # [B, d]
        lat_mean = latent.mean(dim=-1)                        # [B, Z]
        cond     = t_emb + self.cond_proj(lat_mean)           # [B, d]

        for layer in self.layers:
            x = layer(x, cond)

        v = self.output_proj(x)        # [B, T, n_mels]
        return v.permute(0, 2, 1)      # [B, n_mels, T]

    def forward_train(self, mel_clean: torch.Tensor, mel_noisy: torch.Tensor,
                      latent: torch.Tensor) -> torch.Tensor:
        """
        Optimal-transport CFM training loss (L1 on velocity field).
        mel_clean, mel_noisy: [B, n_mels, T]
        latent: [B, Z, T]
        Returns: scalar loss
        """
        B = mel_clean.shape[0]
        t = torch.rand(B, device=mel_clean.device)  # t ~ U[0,1]

        # Interpolate: mel_t = (1-t)*mel_noisy + t*mel_clean  (OT path)
        t_ = t[:, None, None]
        mel_t = (1.0 - t_) * mel_noisy + t_ * mel_clean

        # Target velocity: mel_clean - mel_noisy (constant vector field)
        v_target = mel_clean - mel_noisy

        v_pred = self._velocity(mel_t, t, latent)
        return F.l1_loss(v_pred, v_target)

    @torch.no_grad()
    def forward_infer(self, mel_noisy: torch.Tensor, latent: torch.Tensor,
                      nfe: int = None, solver: str = None,
                      temperature: float = None) -> torch.Tensor:
        """
        ODE solve: mel_noisy → mel_clean.
        mel_noisy: [B, n_mels, T]
        Returns: mel_clean [B, n_mels, T]
        """
        cfg = self.cfg
        nfe         = nfe or cfg.cfm_nfe
        solver      = solver or cfg.cfm_solver
        temperature = temperature if temperature is not None else cfg.cfm_temperature

        # Add temperature noise to starting point
        x = mel_noisy + temperature * torch.randn_like(mel_noisy)

        dt = 1.0 / nfe
        for i in range(nfe):
            t_val = i * dt
            t = torch.full((x.shape[0],), t_val, device=x.device)

            if solver == "midpoint":
                v1 = self._velocity(x, t, latent)
                x_mid = x + 0.5 * dt * v1
                t_mid = t + 0.5 * dt
                v2 = self._velocity(x_mid, t_mid, latent)
                x = x + dt * v2
            else:  # euler
                v = self._velocity(x, t, latent)
                x = x + dt * v

        return x


# ── 4. UnivNet Generator (~9M) ────────────────────────────────────────────────

class _Snake(nn.Module):
    """Snake activation: x + (1/a) * sin^2(a*x)"""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-8)) * (self.alpha * x).sin() ** 2


class _KernelPredictor(nn.Module):
    """
    Per-frame kernel predictor for Location-Variable Convolutions.
    Maps mel conditioning → conv kernels + biases.
    """

    def __init__(self, cond_ch: int, conv_ch: int, conv_layers: int, ksize: int = 3):
        super().__init__()
        self.conv_ch     = conv_ch
        self.conv_layers = conv_layers
        self.ksize       = ksize

        kernel_ch = conv_ch * conv_ch * ksize * conv_layers
        bias_ch   = conv_ch * conv_layers

        self.net = nn.Sequential(
            nn.Conv1d(cond_ch, 64, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, kernel_ch + bias_ch, 1),
        )
        self.kc = kernel_ch
        self.bc = bias_ch

    def forward(self, cond: torch.Tensor):
        """cond: [B, cond_ch, T] → kernels list, biases list"""
        out = self.net(cond)               # [B, kc+bc, T]
        kernels_flat = out[:, :self.kc]    # [B, kc, T]
        biases_flat  = out[:, self.kc:]    # [B, bc, T]
        return kernels_flat, biases_flat


class _LVCBlock(nn.Module):
    """
    Location-Variable Convolution block: apply position-specific kernels.
    Massively reduces params vs full Conv1d over position.
    """

    def __init__(self, ch: int, cond_ch: int, n_layers: int, ksize: int = 3):
        super().__init__()
        self.ch       = ch
        self.n_layers = n_layers
        self.ksize    = ksize
        self.predictor = _KernelPredictor(cond_ch, ch, n_layers, ksize)
        self.activation = _Snake(ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, ch, T_audio]
        cond: [B, cond_ch, T_mel]  — upsampled to match T_audio before calling
        Returns: [B, ch, T_audio]
        """
        B, C, T = x.shape
        kernels_flat, biases_flat = self.predictor(cond)  # [B, kc, T], [B, bc, T]

        ch, ksize, n_layers = self.ch, self.ksize, self.n_layers
        k_per_layer = ch * ch * ksize

        for i in range(n_layers):
            # Extract kernels and biases for this layer [B, ch*ch*ksize, T]
            k_flat = kernels_flat[:, i * k_per_layer:(i + 1) * k_per_layer]
            b_flat = biases_flat[:, i * ch:(i + 1) * ch]

            # Apply position-specific conv frame by frame (einsum trick)
            # k_flat: [B, ch_out*ch_in*k, T] → [B*T, ch_out, ch_in, k]
            # This is an approximation: treat each time step independently
            # Full LVC is expensive; we use grouped + depthwise as a practical simplification
            dilation = 2 ** i
            pad = dilation * (ksize - 1) // 2
            # Use standard conv with mean-pooled kernel (keeps params low)
            k_mean = k_flat.mean(dim=-1).view(B, ch, ch, ksize)
            b_mean = b_flat.mean(dim=-1).view(B, ch)

            # Apply per-sample (loop over batch) — acceptable for inference
            outs = []
            for b in range(B):
                o = F.conv1d(x[b:b+1], k_mean[b], b_mean[b],
                             dilation=dilation, padding=pad)
                outs.append(o)
            x = torch.cat(outs, dim=0) + x  # residual

        return self.activation(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, cond_ch: int, n_lvc: int):
        super().__init__()
        self.upsample  = nn.ConvTranspose1d(in_ch, out_ch, stride * 2,
                                            stride=stride, padding=stride // 2)
        self.snake     = _Snake(out_ch)
        self.lvc       = _LVCBlock(out_ch, cond_ch, n_lvc)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x    = self.snake(self.upsample(x))
        cond = F.interpolate(cond, size=x.shape[-1], mode="nearest")
        x    = self.lvc(x, cond)
        return x


class UnivNetGenerator(nn.Module):
    """
    Mel → 48kHz waveform via 3× ×8 upsampling with LVC blocks.
    ~9M params.
    """

    def __init__(self, cfg: EnhancerConfig):
        super().__init__()
        nc        = cfg.univnet_nc
        n_mels    = cfg.n_mels
        strides   = cfg.univnet_upsample_strides
        n_lvc     = cfg.univnet_lvc_layers

        self.input_conv = nn.Conv1d(n_mels, nc * 4, 7, padding=3)

        ch = nc * 4
        self.blocks = nn.ModuleList()
        for stride in strides:
            out_ch = ch // 2
            self.blocks.append(_UpsampleBlock(ch, out_ch, stride, n_mels, n_lvc))
            ch = out_ch

        self.output_conv = nn.Sequential(
            _Snake(ch),
            nn.Conv1d(ch, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mels, T] → wav [B, 1, T_audio]"""
        x = self.input_conv(mel)
        for block in self.blocks:
            x = block(x, mel)
        return self.output_conv(x)


# ── Discriminators (for Stage 3 vocoder training) ────────────────────────────

class _PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        ch = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList([
            nn.utils.spectral_norm(nn.Conv2d(ch[i], ch[i+1], (5, 1), (3, 1), (2, 0)))
            for i in range(len(ch) - 1)
        ])
        self.final = nn.utils.spectral_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))
        self.act   = nn.LeakyReLU(0.1)

    def forward(self, wav: torch.Tensor):
        """wav: [B, 1, T] or [B, T]"""
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        B, C, T = wav.shape
        # Pad to multiple of period
        pad = (self.period - T % self.period) % self.period
        wav = F.pad(wav, (0, pad))
        wav = wav.view(B, C, -1, self.period)  # [B, 1, T//p, p]
        feats = []
        x = wav
        for conv in self.convs:
            x = self.act(conv(x))
            feats.append(x)
        x = self.final(x)
        feats.append(x)
        return x.flatten(1, -1).mean(-1), feats


class UnivNetDiscriminator(nn.Module):
    """Multi-Period Discriminator for vocoder training."""

    PERIODS = [2, 3, 5, 7, 11]

    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([_PeriodDiscriminator(p) for p in self.PERIODS])

    def forward(self, wav: torch.Tensor):
        logits, feats = [], []
        for d in self.discs:
            l, f = d(wav)
            logits.append(l)
            feats.append(f)
        return logits, feats


# ── EnhancerModel (full pipeline wrapper) ─────────────────────────────────────

class EnhancerModel(nn.Module):
    """
    Full Inflect Enhancer pipeline.

    Usage:
        model = EnhancerModel(cfg)
        wav_enhanced = model(wav_in)           # TTS pipeline (denoiser off)
        wav_enhanced = model(wav_in, denoise=True)  # noisy mic input
    """

    def __init__(self, cfg: EnhancerConfig = None):
        super().__init__()
        cfg = cfg or EnhancerConfig()
        self.cfg      = cfg
        self.mel      = MelExtractor(cfg)
        self.denoiser = STFTDenoiser(cfg)
        self.encoder  = IRMAEEncoder(cfg)
        self.decoder  = IRMAEDecoder(cfg)
        self.cfm      = CFMEnhancer(cfg)
        self.vocoder  = UnivNetGenerator(cfg)

    def forward(self, wav: torch.Tensor, denoise: bool = False,
                nfe: int = None, temperature: float = None) -> torch.Tensor:
        """
        wav: [B, T] at cfg.sample_rate
        Returns: enhanced wav [B, T]
        """
        if denoise:
            wav = self.denoiser(wav)

        mel_noisy = self.mel(wav)                           # [B, n_mels, T_mel]
        latent    = self.encoder(mel_noisy)                 # [B, Z, T_mel]
        mel_clean = self.cfm.forward_infer(mel_noisy, latent, nfe=nfe,
                                           temperature=temperature)
        wav_out   = self.vocoder(mel_clean)                 # [B, 1, T_audio]
        return wav_out.squeeze(1)

    def count_params(self) -> dict:
        return {
            "denoiser":  count_params(self.denoiser),
            "irmae_enc": count_params(self.encoder),
            "irmae_dec": count_params(self.decoder),
            "cfm":       count_params(self.cfm),
            "vocoder":   count_params(self.vocoder),
            "total":     count_params(self),
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = EnhancerConfig()
    model = EnhancerModel(cfg)
    params = model.count_params()

    print("\nInflect Enhancer — Parameter Counts")
    print(f"  denoiser:  {params['denoiser']:>10,}")
    print(f"  irmae_enc: {params['irmae_enc']:>10,}")
    print(f"  irmae_dec: {params['irmae_dec']:>10,}")
    print(f"  cfm:       {params['cfm']:>10,}")
    print(f"  vocoder:   {params['vocoder']:>10,}")
    print(f"  total:     {params['total']:>10,}")

    # Forward pass smoke test
    wav = torch.randn(2, 48_000)  # 2 clips, 1 second at 48kHz
    with torch.no_grad():
        out = model(wav, denoise=False)
    print(f"\nSmoke test: in {list(wav.shape)} → out {list(out.shape)}")
    print("OK")
