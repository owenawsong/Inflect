"""
Inflect — Phase 1 Data Pipeline
Extracts Mimi latents from paralinguistic ElevenLabs clips for Para Module training.

For each clip in manifest.csv:
  1. Load audio
  2. Encode full clip → Mimi latents (32-dim, 12.5 Hz)
  3. Detect timestamp of paralinguistic event using word-ratio heuristic
  4. Slice Mimi latents → paralinguistic segment latents
  5. Get speaker state from non-paralinguistic portion
  6. Save: {tag_id, voice, speaker_latents, para_latents, para_duration}

Output: paralinguistic_dataset.pt
  List of dicts, each with keys:
    - tag_id: int
    - tag: str (canonical)
    - voice: str
    - speaker_latents: Tensor [N, 512]  (Mimi latents of clean speech, speaker conditioning)
    - para_latents:   Tensor [T, 512]  (Mimi latents of paralinguistic segment, training target)
    - para_duration:  float (seconds)
"""

import csv
import re
import sys
from pathlib import Path

import torch
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio

# Add project root to path so `inflect` package is importable
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from inflect.data.tags import tag_to_id, normalize_tag

MANIFEST = Path("C:/Users/Owen/Inflect-New/voice-encoder/data/paralinguistic/manifest.csv")
OUT_PATH  = Path("C:/Users/Owen/Inflect-New/inflect/data/paralinguistic_dataset.pt")

MIMI_FRAME_RATE   = 12.5   # frames per second
MIMI_SAMPLE_RATE  = 24_000  # samples per second
MIMI_DIM          = 512    # Mimi latent dimension (confirmed empirically)
PARA_WINDOW_SEC   = 2.5    # max seconds to capture around the tag (laugh = ~1.5s, gasp = ~0.5s)
PARA_MIN_FRAMES   = 4      # minimum frames (0.32s) for a valid paralinguistic segment
PARA_MAX_FRAMES   = 50     # maximum frames (4s) hard cap


def load_audio(path: str, target_sr: int = MIMI_SAMPLE_RATE) -> torch.Tensor:
    """Load audio file → [1, C, T] float32 tensor at target_sr.
    Uses Pocket TTS's own audio_read which works without FFmpeg on Windows."""
    wav, sr = audio_read(path)          # wav: [C, T], sr: int
    wav = convert_audio(wav, sr, target_sr, to_channels=1)  # [1, T]
    return wav.unsqueeze(0)             # [1, 1, T]


def find_tag_position(text: str) -> tuple[float, float]:
    """
    Given text like "That's so funny [laughs] I can't believe it",
    return (start_ratio, end_ratio) of where the tag falls in the text.

    Uses word-count ratio: words before tag / total words.
    Returns (ratio_start, ratio_end) where ratio_end = ratio_start (tag is a point event).
    """
    # Remove tag from text to count surrounding words
    tag_pattern = re.compile(r'\[([^\]]+)\]')

    # Find first tag
    match = tag_pattern.search(text)
    if not match:
        return 0.5, 0.5  # fallback: assume middle

    before_text = text[:match.start()]
    after_text  = text[match.end():]

    words_before = len(before_text.split())
    words_after  = len(after_text.split())
    total_words  = words_before + words_after

    if total_words == 0:
        return 0.0, 1.0

    ratio = words_before / total_words
    return ratio, ratio


def estimate_para_segment(
    audio_duration: float,
    tag_ratio: float,
    window_sec: float = PARA_WINDOW_SEC,
) -> tuple[float, float]:
    """
    Estimate start/end seconds of the paralinguistic event.
    The tag is at tag_ratio * audio_duration.
    We grab window_sec / 2 before and after.
    """
    tag_sec = tag_ratio * audio_duration

    # Tags at the very start: grab from 0
    # Tags at the very end: grab to end
    start_sec = max(0.0, tag_sec - window_sec * 0.4)
    end_sec   = min(audio_duration, tag_sec + window_sec * 0.6)

    return start_sec, end_sec


def extract_latent_segment(
    latents: torch.Tensor,
    start_sec: float,
    end_sec: float,
) -> torch.Tensor:
    """
    Slice Mimi latents [1, 512, N] for the given time range.
    Returns [T, 512] (transposed, batch dim removed).
    """
    start_frame = int(start_sec * MIMI_FRAME_RATE)
    end_frame   = int(end_sec   * MIMI_FRAME_RATE)
    segment = latents[0, :, start_frame:end_frame]  # [512, T]
    return segment.transpose(0, 1)                  # [T, 512]


def get_speaker_latents(
    latents: torch.Tensor,
    para_start_sec: float,
    para_end_sec: float,
    audio_duration: float,
    max_frames: int = 75,  # 6 seconds of speaker context
) -> torch.Tensor:
    """
    Extract raw Mimi latents from the non-paralinguistic speech portion of the clip.
    Returns [N, 512] — the Para Module will learn its own speaker projection.
    """
    before_sec = para_start_sec
    after_sec  = audio_duration - para_end_sec

    if before_sec >= after_sec and before_sec > 0.5:
        start = max(0.0, para_start_sec - 6.0)
        end   = para_start_sec
    elif after_sec > 0.5:
        start = para_end_sec
        end   = min(audio_duration, para_end_sec + 6.0)
    else:
        start = 0.0
        end   = audio_duration

    sf = int(start * MIMI_FRAME_RATE)
    ef = int(end   * MIMI_FRAME_RATE)
    segment = latents[0, :, sf:ef]      # [512, N]
    segment = segment.transpose(0, 1)   # [N, 512]

    # Cap to max_frames
    if segment.shape[0] > max_frames:
        segment = segment[:max_frames]

    return segment


def main():
    print("Loading Pocket TTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    model.eval()
    print(f"  Sample rate: {model.sample_rate} Hz")
    print(f"  Mimi frame rate: {model.mimi.frame_rate} Hz")

    dataset = []
    skipped = 0
    errors  = 0

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} clips...")

    for i, row in enumerate(rows):
        audio_path  = row["audio_path"]
        text        = row["text"]
        voice       = row["voice"]
        raw_tags    = [t.strip() for t in row["tags"].split("|")]

        # Only process clips with exactly one primary tag (simpler training signal)
        primary_tag = None
        for rt in raw_tags:
            tid = tag_to_id(rt)
            if tid is not None:
                primary_tag = (rt, tid)
                break

        if primary_tag is None:
            skipped += 1
            continue

        raw_tag, tag_id = primary_tag
        canonical_tag   = normalize_tag(raw_tag)

        try:
            # Load audio
            audio = load_audio(audio_path)  # [1, 1, T]
            audio_duration = audio.shape[-1] / MIMI_SAMPLE_RATE

            # Encode full clip to Mimi latents [1, 32, N]
            with torch.no_grad():
                latents = model.mimi.encode_to_latent(audio)

            n_frames = latents.shape[-1]

            # Find tag position and estimate segment
            tag_ratio, _ = find_tag_position(text)
            para_start, para_end = estimate_para_segment(audio_duration, tag_ratio)

            # Extract paralinguistic segment
            para_latents = extract_latent_segment(latents, para_start, para_end)  # [T, 512]
            T = para_latents.shape[0]

            if T < PARA_MIN_FRAMES:
                skipped += 1
                continue
            if T > PARA_MAX_FRAMES:
                para_latents = para_latents[:PARA_MAX_FRAMES, :]
                T = PARA_MAX_FRAMES

            # Extract speaker latents from clean speech portion
            speaker_latents = get_speaker_latents(
                latents, para_start, para_end, audio_duration
            )  # [N, 512]

            dataset.append({
                "tag_id":           tag_id,
                "tag":              canonical_tag,
                "voice":            voice,
                "speaker_latents":  speaker_latents.to(torch.float16),  # [N, 512]
                "para_latents":     para_latents.to(torch.float16),      # [T, 512]
                "para_duration":    T / MIMI_FRAME_RATE,
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(rows)}] OK  tag={canonical_tag}  T={T}  voice={voice}")

        except Exception as e:
            print(f"  [ERROR] {audio_path}: {e}")
            errors += 1
            continue

    print(f"\nDone. {len(dataset)} samples extracted, {skipped} skipped, {errors} errors.")

    # Tag distribution summary
    from collections import Counter
    tag_counts = Counter(s["tag"] for s in dataset)
    print("\nTag distribution:")
    for tag, count in tag_counts.most_common():
        print(f"  {count:4d}  {tag}")

    print(f"\nSaving to {OUT_PATH}...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, OUT_PATH)
    print(f"Saved {len(dataset)} samples ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
