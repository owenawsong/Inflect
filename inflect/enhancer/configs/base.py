"""
Inflect Enhancer — Config

All architecture and training hyperparameters in one place.
"""

from dataclasses import dataclass, field


@dataclass
class EnhancerConfig:
    # ── Audio ─────────────────────────────────────────────────────────────────
    sample_rate: int   = 48_000   # VoxCPM outputs 48kHz — we match it
    n_fft: int         = 2048     # same freq resolution per Hz as 24kHz/1024
    hop_length: int    = 512      # 48000/512 = 93.75 Hz frame rate
    n_mels: int        = 80
    f_min: float       = 0.0
    f_max: float       = 22_050.0  # Nyquist for 48kHz

    # ── Denoiser (optional, off by default for TTS pipeline) ─────────────────
    # Note: hidden=32 after projecting 1025 freq bins → 256, keeps params ~2M
    denoiser_hidden: int    = 32
    denoiser_freq_proj: int = 256   # project 1025-dim freq axis here
    denoiser_levels: int    = 3     # UNet encoder/decoder levels

    # ── IRMAE ─────────────────────────────────────────────────────────────────
    irmae_hidden: int   = 512
    irmae_latent: int   = 48
    irmae_n_blocks: int = 2      # residual blocks; each block = 4 dilated convs
    irmae_noise_std: float = 0.1  # additive noise on latent during training only

    # ── UnivNet Vocoder ───────────────────────────────────────────────────────
    univnet_nc: int      = 48    # channel width for LVC kernel predictor
    # 3 × ×8 upsampling = ×512 = 48000 / 93.75 Hz frame rate
    univnet_upsample_strides: tuple = (8, 8, 8)
    univnet_lvc_layers: int = 4  # dilated LVC layers per upsample block

    # ── CFM Enhancer ──────────────────────────────────────────────────────────
    cfm_hidden: int     = 256
    cfm_n_layers: int   = 8
    cfm_n_heads: int    = 4
    cfm_ffn_mult: int   = 2
    # Inference defaults
    cfm_nfe: int        = 8          # number of ODE function evaluations
    cfm_solver: str     = "midpoint" # "euler" or "midpoint"
    cfm_temperature: float = 0.5     # noise temperature at inference

    # ── Training ──────────────────────────────────────────────────────────────
    lr: float          = 2e-4
    batch_size: int    = 16
    clip_seconds: float = 4.0    # fixed clip length for batching
    warmup_steps: int  = 1_000
    # Stage step budgets
    stage1_steps: int  = 10_000  # IRMAE pretraining
    stage2_steps: int  = 50_000  # CFM training
    stage3_steps: int  = 30_000  # UnivNet vocoder
    # Logging
    log_every: int     = 100
    save_every: int    = 1_000
    seed: int          = 42
