"""
Inflect - Para Module Audio Test (v2, Mel Spectrogram)

Generates audio from predicted mel spectrograms.
Uses Griffin-Lim by default (no extra deps). Install vocos for better quality.

Usage:
    python inflect/para_module/test.py
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import soundfile as sf

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inflect.para_module.model import ParaModule
from inflect.data.tags import TAGS, TAG_TO_ID

OUT_DIR   = PROJECT_ROOT / "inflect/para_module/test_outputs"
DEFAULT_CKPT_PATH = PROJECT_ROOT / "inflect/para_module/checkpoints/para_best.pt"
DEFAULT_DATA_PATH = PROJECT_ROOT / "inflect/data/paralinguistic_dataset_mel.pt"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE  = 24_000
N_FFT        = 1024
HOP_LENGTH   = 256
N_MELS       = 80
VOCOS_MELS   = 100
VOCOS_REPO   = "charactr/vocos-mel-24khz"

TEST_TAGS = ["laughs", "gasps", "sighs", "crying", "clears_throat", "whispers", "excited", "laughs_hard"]


def mel_to_audio_griffinlim(log_mel: torch.Tensor) -> torch.Tensor:
    """Convert log-mel [T, 80] to audio using Griffin-Lim (no neural vocoder needed)."""
    import torchaudio.transforms as T
    import torchaudio.functional as AF

    mel = log_mel.T.float().exp()  # [80, T] amplitude mel

    # Approximate inverse mel filterbank using torchaudio
    inv_mel = T.InverseMelScale(
        n_stft=N_FFT // 2 + 1,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        f_min=0,
        f_max=8000,
    )(mel)  # [n_stft, T]

    gl = T.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=64,
        power=1.0,
    )(inv_mel)  # [T_audio]

    return gl


def mel_to_audio_vocos(log_mel: torch.Tensor) -> torch.Tensor:
    """High-quality mel-to-audio using Vocos (pip install vocos)."""
    vocos = load_cached_vocos()
    with torch.no_grad():
        mel_input = prepare_vocos_features(log_mel)
        audio = vocos.decode(mel_input)
    return audio.squeeze(0)


def prepare_vocos_features(log_mel: torch.Tensor) -> torch.Tensor:
    """
    Map our 80-bin log-mel features into the 100-bin format expected by
    `charactr/vocos-mel-24khz`.
    """
    mel = log_mel.T.unsqueeze(0).float().exp()  # [1, 80, T]
    mel = F.interpolate(
        mel.unsqueeze(1),
        size=(VOCOS_MELS, mel.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)  # [1, 100, T]
    return mel.clamp_min(1e-5).log()


def load_cached_vocos():
    """Load Vocos from the local Hugging Face cache without network access."""
    from vocos import Vocos

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{VOCOS_REPO.replace('/', '--')}"
    snapshot_dir = repo_dir / "snapshots"
    snapshots = sorted(snapshot_dir.glob("*")) if snapshot_dir.exists() else []
    if not snapshots:
        raise FileNotFoundError(
            f"Vocos cache not found at {snapshot_dir}. Run once with network access to cache {VOCOS_REPO}."
        )

    latest = snapshots[-1]
    config_path = latest / "config.yaml"
    model_path = latest / "pytorch_model.bin"

    vocos = Vocos.from_hparams(str(config_path))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    vocos.load_state_dict(state_dict)
    vocos.eval()
    return vocos


def save_wav(audio: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = audio.cpu().float().numpy()
    if arr.ndim > 1:
        arr = arr.squeeze()
    sf.write(str(path), arr, SAMPLE_RATE)


def main(ckpt_path: Path, data_path: Path):
    # Choose vocoder
    try:
        import vocos
        vocoder_fn = mel_to_audio_vocos
        print("Using Vocos vocoder (high quality)")
    except ImportError:
        vocoder_fn = mel_to_audio_griffinlim
        print("Using Griffin-Lim vocoder (install vocos for better quality: pip install vocos)")

    print(f"\nLoading Para Module from {ckpt_path}...")
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ParaModule()
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    print(f"  Loaded epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    print(f"\nLoading dataset...")
    data = torch.load(data_path, weights_only=False)

    tag_samples = {}
    for s in data:
        tag_samples.setdefault(s["tag"], []).append(s)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating -> {OUT_DIR}\n")

    for tag in TEST_TAGS:
        if tag not in tag_samples:
            print(f"  [{tag}] - no samples, skipping")
            continue

        tag_id  = TAG_TO_ID[tag]
        samples = tag_samples[tag]

        # Pick 2 different voices
        seen, picked = set(), []
        for s in samples:
            if s["voice"] not in seen:
                picked.append(s); seen.add(s["voice"])
            if len(picked) >= 2:
                break

        for s in picked:
            voice   = s["voice"]
            spk_emb = s["speaker_emb"].float().unsqueeze(0).to(DEVICE)  # [1, 80]
            target_T = int(s["para_mel"].shape[0])

            with torch.no_grad():
                out = model(spk_emb, torch.tensor([tag_id], device=DEVICE), target_T=target_T)
            pred_mel = out["pred_mel"].squeeze(0).cpu()   # [T, 80]
            gt_mel   = s["para_mel"].float()              # [T, 80]

            try:
                pred_audio = vocoder_fn(pred_mel)
                gt_audio   = vocoder_fn(gt_mel)

                save_wav(pred_audio, OUT_DIR / f"pred_{tag}_{voice}.wav")
                save_wav(gt_audio,   OUT_DIR / f"gt_{tag}_{voice}.wav")

                print(f"  [{tag:15s}] {voice:12s}  pred_T={pred_mel.shape[0]}  gt_T={gt_mel.shape[0]}  OK")
            except Exception as e:
                print(f"  [{tag}] {voice} ERROR: {e}")

    print(f"\nDone. Listen at: {OUT_DIR}")
    print("Compare pred_*.wav (Para Module) vs gt_*.wav (original ElevenLabs)")
    print("\nFor higher quality, run:  pip install vocos  then re-run this script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    main(args.ckpt, args.dataset)
