"""
Inflect Enhancer — Inference Pipeline

Usage:
  from inflect.enhancer.infer import EnhancerPipeline
  pipe = EnhancerPipeline("inflect/enhancer/checkpoints")
  enhanced_wav = pipe.enhance(wav_tensor, sr=48000)

CLI:
  python inflect/enhancer/infer.py --input in.wav --output out.wav
  python inflect/enhancer/infer.py --input in.wav --output out.wav --denoise --nfe 16
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inflect.enhancer.configs.base import EnhancerConfig
from inflect.enhancer.model import (
    EnhancerModel, IRMAEEncoder, IRMAEDecoder,
    CFMEnhancer, UnivNetGenerator, MelExtractor,
)

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    from scipy.io import wavfile as _sciwav
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _load_wav(path: str | Path, target_sr: int) -> torch.Tensor:
    path = str(path)
    if _HAS_SF:
        audio, sr = sf.read(path, dtype="float32")
    elif _HAS_SCIPY:
        sr, audio = _sciwav.read(path)
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0
    else:
        raise RuntimeError("pip install soundfile")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    wav = torch.from_numpy(audio)
    if sr != target_sr and _HAS_SCIPY:
        import math
        g = math.gcd(sr, target_sr)
        out = resample_poly(wav.numpy(), target_sr // g, sr // g)
        wav = torch.from_numpy(out.astype(np.float32))
    return wav


def _save_wav(path: str | Path, wav: torch.Tensor, sr: int):
    path = str(path)
    audio = wav.cpu().numpy()
    if _HAS_SF:
        sf.write(path, audio, sr, format="WAV", subtype="PCM_16")
    elif _HAS_SCIPY:
        audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        _sciwav.write(path, sr, audio_int)
    else:
        raise RuntimeError("pip install soundfile")


class EnhancerPipeline:
    """
    Full Inflect Enhancer inference pipeline.

    Loads component checkpoints individually (they're trained in separate stages),
    assembles the full pipeline, and exposes a simple `enhance()` method.
    """

    def __init__(
        self,
        ckpt_dir:     str | Path = None,
        device:       str = None,
        cfg:          EnhancerConfig = None,
        nfe:          int = None,
        temperature:  float = None,
    ):
        self.cfg         = cfg or EnhancerConfig()
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nfe         = nfe or self.cfg.cfm_nfe
        self.temperature = temperature if temperature is not None else self.cfg.cfm_temperature

        ckpt_dir = Path(ckpt_dir or PROJECT_ROOT / "inflect/enhancer/checkpoints")

        self.mel_fn  = MelExtractor(self.cfg).to(self.device)
        self.encoder = IRMAEEncoder(self.cfg).to(self.device)
        self.decoder = IRMAEDecoder(self.cfg).to(self.device)
        self.cfm     = CFMEnhancer(self.cfg).to(self.device)
        self.vocoder = UnivNetGenerator(self.cfg).to(self.device)

        # Denoiser only loaded if checkpoint exists
        self._denoiser = None

        self._load_checkpoints(ckpt_dir)

        self.mel_fn.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.cfm.eval()
        self.vocoder.eval()

    def _load_checkpoints(self, ckpt_dir: Path):
        def _maybe_load(module, path, key):
            if path.exists():
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                module.load_state_dict(ckpt[key])
                print(f"  Loaded {path.name}")
            else:
                print(f"  [skip] {path.name} not found")

        _maybe_load(self.encoder, ckpt_dir / "irmae_best.pt",  "encoder")
        _maybe_load(self.decoder, ckpt_dir / "irmae_best.pt",  "decoder")
        _maybe_load(self.cfm,     ckpt_dir / "cfm_best.pt",    "cfm")
        _maybe_load(self.vocoder, ckpt_dir / "univnet_best.pt", "generator")

    @torch.no_grad()
    def enhance(
        self,
        wav:       torch.Tensor,
        sr:        int = None,
        denoise:   bool = False,
        nfe:       int = None,
        temperature: float = None,
    ) -> torch.Tensor:
        """
        wav: [T] or [B, T] mono float32 at cfg.sample_rate (or sr)
        Returns: enhanced wav [T] or [B, T]
        """
        sr          = sr or self.cfg.sample_rate
        nfe         = nfe or self.nfe
        temperature = temperature if temperature is not None else self.temperature

        squeeze = wav.dim() == 1
        if squeeze:
            wav = wav.unsqueeze(0)  # [1, T]

        wav = wav.to(self.device)

        # Optional denoising
        if denoise and self._denoiser is not None:
            wav = self._denoiser(wav)

        mel_noisy = self.mel_fn(wav)                             # [B, n_mels, T_mel]
        latent    = self.encoder(mel_noisy)                      # [B, Z, T_mel]
        mel_clean = self.cfm.forward_infer(mel_noisy, latent,
                                           nfe=nfe, temperature=temperature)
        wav_out   = self.vocoder(mel_clean).squeeze(1)           # [B, T_audio]

        if squeeze:
            wav_out = wav_out.squeeze(0)

        return wav_out.cpu()

    def enhance_file(
        self,
        in_path:   str | Path,
        out_path:  str | Path,
        denoise:   bool = False,
        nfe:       int = None,
    ):
        """Load wav file, enhance, save."""
        wav = _load_wav(in_path, self.cfg.sample_rate)
        out = self.enhance(wav, denoise=denoise, nfe=nfe)
        _save_wav(out_path, out, self.cfg.sample_rate)
        print(f"{in_path} -> {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Inflect Enhancer inference")
    ap.add_argument("--input",    required=True,  type=Path)
    ap.add_argument("--output",   required=True,  type=Path)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--denoise",  action="store_true",
                    help="Enable STFT denoiser (for noisy mic input, not needed for TTS)")
    ap.add_argument("--nfe",      type=int, default=None,
                    help=f"CFM ODE steps (default: {EnhancerConfig().cfm_nfe})")
    ap.add_argument("--device",   type=str, default=None)
    args = ap.parse_args()

    print("Loading Inflect Enhancer...")
    pipe = EnhancerPipeline(
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        nfe=args.nfe,
    )
    print(f"Enhancing: {args.input}")
    pipe.enhance_file(args.input, args.output, denoise=args.denoise, nfe=args.nfe)
    print("Done.")


if __name__ == "__main__":
    main()
