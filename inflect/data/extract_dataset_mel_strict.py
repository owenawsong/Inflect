"""
Inflect - Strict mel dataset extraction for the Para Module.

Why this exists:
- the original extractor uses a broad text-ratio heuristic
- many rows contain full spoken sentences, multiple tags, or style labels
- the current Para Module was trained on targets that are not clean event-only audio

This script builds a stricter dataset:
- only one bracket tag in the text
- only canonical SOUND_EVENT_TAGS
- segment boundaries chosen with simple energy-valley snapping
- saves enough metadata to audit the extraction quality

Output:
    inflect/data/paralinguistic_dataset_mel_strict.pt
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inflect.data.tags import SOUND_EVENT_TAGS, normalize_tag, tag_to_id
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio

try:
    import torchaudio.transforms as T
except Exception as exc:
    raise RuntimeError("torchaudio is required for strict mel extraction") from exc

MANIFEST = PROJECT_ROOT / "voice-encoder" / "data" / "paralinguistic" / "manifest.csv"
OUT_PATH = PROJECT_ROOT / "inflect" / "data" / "paralinguistic_dataset_mel_strict.pt"

SAMPLE_RATE = 24_000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 0
FMAX = 8_000
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH

MIN_EVENT_SEC = 0.18
MAX_EVENT_SEC = 1.20
MAX_SPEAKER_SEC = 3.0
ENERGY_SMOOTH = 9

TAG_DURATION_PRIOR = {
    "gasps": 0.45,
    "inhales": 0.55,
    "clears_throat": 0.45,
    "sighs": 0.85,
    "exhales": 0.75,
    "laughs": 0.90,
    "laughs_hard": 0.95,
    "chuckles": 0.65,
    "giggles": 0.85,
    "wheezing": 0.90,
    "snorts": 0.50,
}


def build_mel_transform():
    return T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        power=1.0,
    )


def log_mel(mel: torch.Tensor) -> torch.Tensor:
    return torch.log(mel.clamp(min=1e-5))


def load_audio(path: str) -> torch.Tensor:
    wav, sr = audio_read(path)
    return convert_audio(wav, sr, SAMPLE_RATE, to_channels=1)


def parse_tags(text: str):
    return list(re.finditer(r"\[([^\]]+)\]", text))


def rms_frames(wav: torch.Tensor) -> torch.Tensor:
    sq = wav.square().unsqueeze(0)
    rms = F.avg_pool1d(sq, kernel_size=N_FFT, stride=HOP_LENGTH, ceil_mode=True).sqrt().squeeze(0)
    rms = rms.squeeze(0)
    if rms.numel() == 0:
        return torch.zeros(1)
    rms = F.avg_pool1d(
        rms.view(1, 1, -1),
        kernel_size=ENERGY_SMOOTH,
        stride=1,
        padding=ENERGY_SMOOTH // 2,
    ).view(-1)
    return rms


def char_ratio(text: str, match: re.Match) -> float:
    if not text:
        return 0.5
    return ((match.start() + match.end()) * 0.5) / max(1, len(text))


def valley_index(energy: torch.Tensor, lo: int, hi: int) -> int:
    lo = max(0, lo)
    hi = min(int(energy.numel()), hi)
    if hi <= lo + 1:
        return lo
    local = energy[lo:hi]
    return lo + int(local.argmin().item())


def infer_segment(duration_sec: float, text: str, match: re.Match, canonical_tag: str, energy: torch.Tensor):
    total_frames = int(energy.numel())
    anchor = int(char_ratio(text, match) * total_frames)
    prior = TAG_DURATION_PRIOR.get(canonical_tag, 0.75)
    min_frames = max(8, int(MIN_EVENT_SEC * FRAME_RATE))
    prior_frames = max(min_frames, int(prior * FRAME_RATE))
    max_frames = min(int(MAX_EVENT_SEC * FRAME_RATE), total_frames)

    before = text[:match.start()].strip()
    after = text[match.end():].strip()

    if not before and after:
        start = 0
        end_lo = min(total_frames, start + min_frames)
        end_hi = min(total_frames, start + max_frames)
        end = valley_index(energy, end_lo, end_hi)
    elif before and not after:
        end = total_frames
        start_lo = max(0, end - max_frames)
        start_hi = max(0, end - min_frames)
        start = valley_index(energy, start_lo, start_hi)
    else:
        start_lo = max(0, anchor - max_frames)
        start_hi = max(0, anchor - max(min_frames // 2, 4))
        end_lo = min(total_frames, anchor + max(min_frames // 2, 4))
        end_hi = min(total_frames, anchor + max_frames)
        start = valley_index(energy, start_lo, start_hi)
        end = valley_index(energy, end_lo, end_hi)

    if end <= start:
        end = min(total_frames, start + prior_frames)
    if end - start < min_frames:
        end = min(total_frames, start + max(min_frames, prior_frames))

    return start / FRAME_RATE, end / FRAME_RATE


def speaker_embedding(mel_db: torch.Tensor, start_sec: float, end_sec: float) -> torch.Tensor:
    total_frames = mel_db.shape[-1]
    start_f = int(start_sec * FRAME_RATE)
    end_f = int(end_sec * FRAME_RATE)

    left = mel_db[:, :start_f]
    right = mel_db[:, end_f:]

    chosen = left if left.shape[-1] >= right.shape[-1] else right
    if chosen.shape[-1] < 8:
        chosen = mel_db

    max_frames = int(MAX_SPEAKER_SEC * FRAME_RATE)
    if chosen.shape[-1] > max_frames:
        chosen = chosen[:, :max_frames]
    return chosen.mean(dim=-1)


def main():
    mel_transform = build_mel_transform()
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))

    dataset = []
    stats = Counter()

    for row in rows:
        text = row["text"]
        matches = parse_tags(text)
        if len(matches) != 1:
            stats["skip_multi_or_zero_tag"] += 1
            continue

        raw_tag = matches[0].group(1).strip()
        canonical = normalize_tag(raw_tag)
        if canonical is None:
            stats["skip_unknown_tag"] += 1
            continue
        if canonical not in SOUND_EVENT_TAGS:
            stats["skip_non_sound_tag"] += 1
            continue

        tag_id = tag_to_id(raw_tag)

        try:
            wav = load_audio(row["audio_path"])
            duration = wav.shape[-1] / SAMPLE_RATE
            mel = mel_transform(wav.squeeze(0))
            mel_db = log_mel(mel)
            energy = rms_frames(wav)

            start_sec, end_sec = infer_segment(duration, text, matches[0], canonical, energy)
            start_f = max(0, int(start_sec * FRAME_RATE))
            end_f = min(mel_db.shape[-1], int(end_sec * FRAME_RATE))

            if end_f - start_f < 8:
                stats["skip_too_short"] += 1
                continue

            para_mel = mel_db[:, start_f:end_f].transpose(0, 1)
            spk_emb = speaker_embedding(mel_db, start_sec, end_sec)

            dataset.append({
                "tag_id": tag_id,
                "tag": canonical,
                "voice": row["voice"],
                "speaker_emb": spk_emb.to(torch.float16),
                "para_mel": para_mel.to(torch.float16),
                "para_duration": para_mel.shape[0] / FRAME_RATE,
                "source_text": text,
                "slice_start_sec": start_sec,
                "slice_end_sec": end_sec,
            })
            stats[f"keep_{canonical}"] += 1
        except Exception:
            stats["error"] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, OUT_PATH)

    print(f"saved {len(dataset)} strict samples -> {OUT_PATH}")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
