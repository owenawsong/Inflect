"""
Inflect — Phase 1 (v2): Mel Spectrogram Dataset Extraction

Extracts 80-dim mel spectrograms from paralinguistic audio clips.
This replaces the Mimi latent approach which required Pocket TTS's
internal 32-dim FlowLM space (not accessible for standalone decoding).

For each clip:
  1. Load audio → mel spectrogram [80, N]
  2. Detect paralinguistic timestamp from tag position in text
  3. Slice mel → paralinguistic segment [80, T]
  4. Slice clean speech mel → mean-pool → speaker embedding [80]
  5. Save: (tag_id, tag, voice, speaker_emb [80], para_mel [T, 80])

Mel settings (24kHz, matches standard HiFiGAN/BigVGAN):
  n_fft=1024, hop_length=256, n_mels=80, fmin=0, fmax=8000
  Frame rate: 24000/256 = 93.75 Hz

Output: inflect/data/paralinguistic_dataset_mel.pt
"""

import csv
import re
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inflect.data.tags import tag_to_id, normalize_tag
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio

try:
    import torchaudio.transforms as T
    USE_TORCHAUDIO = True
except Exception:
    USE_TORCHAUDIO = False

MANIFEST      = Path("C:/Users/Owen/Inflect-New/voice-encoder/data/paralinguistic/manifest.csv")
OUT_PATH      = PROJECT_ROOT / "inflect/data/paralinguistic_dataset_mel.pt"

SAMPLE_RATE   = 24_000
N_FFT         = 1024
HOP_LENGTH    = 256
N_MELS        = 80
FMIN          = 0
FMAX          = 8_000
MEL_FRAME_RATE = SAMPLE_RATE / HOP_LENGTH   # 93.75 Hz

PARA_WINDOW_SEC = 2.5
PARA_MIN_FRAMES = 6    # ~0.064s
PARA_MAX_FRAMES = 250  # ~2.67s
SPEAKER_MAX_FRAMES = 280  # ~3s of speaker context


def build_mel_transform():
    if not USE_TORCHAUDIO:
        raise RuntimeError("torchaudio not available")
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        power=1.0,          # amplitude (not power) mel — standard for HiFiGAN
    )


def log_mel(mel: torch.Tensor) -> torch.Tensor:
    """Convert amplitude mel to log mel (clamp to avoid -inf)."""
    return torch.log(mel.clamp(min=1e-5))


def load_audio(path: str) -> torch.Tensor:
    """Load audio → [1, T] mono float32 at SAMPLE_RATE."""
    wav, sr = audio_read(path)
    wav = convert_audio(wav, sr, SAMPLE_RATE, to_channels=1)  # [1, T]
    return wav


def find_tag_ratio(text: str) -> float:
    """Estimate where [tag] falls in text as a 0–1 ratio."""
    match = re.search(r'\[([^\]]+)\]', text)
    if not match:
        return 0.5
    before = text[:match.start()].split()
    after  = text[match.end():].split()
    total  = len(before) + len(after)
    return len(before) / total if total > 0 else 0.0


def slice_mel(mel_db: torch.Tensor, start_sec: float, end_sec: float) -> torch.Tensor:
    """Slice log-mel [80, N] at time range → [T, 80]."""
    sf = int(start_sec * MEL_FRAME_RATE)
    ef = int(end_sec   * MEL_FRAME_RATE)
    seg = mel_db[:, sf:ef]           # [80, T]
    return seg.transpose(0, 1)       # [T, 80]


def speaker_embedding(mel_db: torch.Tensor, para_start: float, para_end: float,
                      duration: float) -> torch.Tensor:
    """Mean-pool clean-speech mel frames → [80] speaker embedding."""
    before = para_start
    after  = duration - para_end

    if before >= after and before > 0.3:
        sf = max(0, int((para_start - 3.0) * MEL_FRAME_RATE))
        ef = int(para_start * MEL_FRAME_RATE)
    elif after > 0.3:
        sf = int(para_end * MEL_FRAME_RATE)
        ef = min(mel_db.shape[-1], int((para_end + 3.0) * MEL_FRAME_RATE))
    else:
        sf, ef = 0, mel_db.shape[-1]

    seg = mel_db[:, sf:ef]                        # [80, N]
    if seg.shape[-1] > SPEAKER_MAX_FRAMES:
        seg = seg[:, :SPEAKER_MAX_FRAMES]
    return seg.mean(dim=-1)                       # [80]


def main():
    mel_transform = build_mel_transform()

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Processing {len(rows)} clips...")

    dataset = []
    skipped = errors = 0

    for i, row in enumerate(rows):
        raw_tags = [t.strip() for t in row["tags"].split("|")]
        primary  = next(
            ((rt, tag_to_id(rt)) for rt in raw_tags if tag_to_id(rt) is not None),
            None
        )
        if primary is None:
            skipped += 1
            continue
        raw_tag, tag_id = primary
        canonical = normalize_tag(raw_tag)

        try:
            wav = load_audio(row["audio_path"])           # [1, T]
            duration = wav.shape[-1] / SAMPLE_RATE

            # Mel spectrogram → log scale
            mel    = mel_transform(wav.squeeze(0))        # [80, N] (amplitude)
            mel_db = log_mel(mel)                         # [80, N] (log)

            # Locate paralinguistic segment
            ratio      = find_tag_ratio(row["text"])
            tag_sec    = ratio * duration
            para_start = max(0.0, tag_sec - PARA_WINDOW_SEC * 0.4)
            para_end   = min(duration, tag_sec + PARA_WINDOW_SEC * 0.6)

            para_mel   = slice_mel(mel_db, para_start, para_end)  # [T, 80]
            T          = para_mel.shape[0]

            if T < PARA_MIN_FRAMES:
                skipped += 1
                continue
            if T > PARA_MAX_FRAMES:
                para_mel = para_mel[:PARA_MAX_FRAMES]
                T = PARA_MAX_FRAMES

            spk_emb = speaker_embedding(mel_db, para_start, para_end, duration)  # [80]

            dataset.append({
                "tag_id":      tag_id,
                "tag":         canonical,
                "voice":       row["voice"],
                "speaker_emb": spk_emb.to(torch.float16),     # [80]
                "para_mel":    para_mel.to(torch.float16),     # [T, 80]
                "para_duration": T / MEL_FRAME_RATE,
            })

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(rows)}]  tag={canonical:15s}  T={T:3d}  voice={row['voice']}")

        except Exception as e:
            print(f"  [ERROR] {row['audio_path']}: {e}")
            errors += 1

    print(f"\nDone: {len(dataset)} samples, {skipped} skipped, {errors} errors")

    tag_counts = Counter(s["tag"] for s in dataset)
    print("\nTag distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {count:4d}  {tag}")

    print(f"\nSaving to {OUT_PATH}...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Saved {len(dataset)} samples ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
