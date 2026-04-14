"""
One-time script: transcribe all reference voice audio files and save .txt transcripts.
Uses miniaudio (no FFmpeg needed) + openai-whisper.

Usage:
    python scripts/transcribe_reference_voices.py
"""
import sys
from pathlib import Path
import numpy as np

REFERENCE_VOICES = Path("reference_voices")
WHISPER_MODEL = "small"  # small handles diverse voices/accents much better


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio to float32 mono numpy array at target_sr.
    Handles .mp3 files that are actually RIFF/WAV containers (wrong extension).
    """
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    # Detect actual format by magic bytes
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic[:4] == b"RIFF":
        # Actually a WAV file despite the extension
        audio, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
    else:
        # True MP3 — use miniaudio
        import miniaudio
        result = miniaudio.mp3_read_file_f32(str(path))
        audio = np.frombuffer(result.samples, dtype=np.float32).copy()
        native_sr = result.sample_rate
        if result.nchannels > 1:
            audio = audio.reshape(-1, result.nchannels).mean(axis=1)

    # Mix down to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to target_sr with scipy
    if native_sr != target_sr:
        g = gcd(target_sr, native_sr)
        audio = resample_poly(audio, target_sr // g, native_sr // g).astype(np.float32)

    # Normalize so Whisper VAD doesn't reject quiet recordings
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio


def main():
    import whisper

    # Collect all audio files
    audio_files = []
    for voice_dir in sorted(REFERENCE_VOICES.iterdir()):
        if not voice_dir.is_dir():
            continue
        for ext in ("*.wav", "*.mp3"):
            for f in voice_dir.glob(ext):
                txt = f.with_suffix(".txt")
                if txt.exists():
                    print(f"  skip (already has .txt): {f.name}")
                else:
                    audio_files.append(f)

    if not audio_files:
        print("All reference voices already have transcripts.")
        return

    print(f"Loading Whisper '{WHISPER_MODEL}' model...")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"Transcribing {len(audio_files)} audio files...\n")

    ok = 0
    err = 0
    for i, path in enumerate(audio_files, 1):
        txt_path = path.with_suffix(".txt")
        print(f"[{i}/{len(audio_files)}] {path.parent.name}/{path.name} ... ", end="", flush=True)
        try:
            audio = load_audio(path)
            peak = np.abs(audio).max()
            if peak < 0.01:
                print(f"SKIP (silent, peak={peak:.6f})")
                continue
            result = model.transcribe(
                audio,
                language="en",
                fp16=False,
                no_speech_threshold=0.05,
                temperature=0,
            )
            transcript = result["text"].strip()
            if not transcript:
                print(f'EMPTY (no speech detected, peak={peak:.6f})')
                txt_path.write_text("[NO_SPEECH]", encoding="utf-8")
            else:
                txt_path.write_text(transcript, encoding="utf-8")
                print(f'"{transcript[:60]}"')
                ok += 1
        except Exception as e:
            print(f"FAIL: {e}")
            err += 1

    print(f"\nDone: {ok} transcribed, {err} failed")
    print("Now run: python scripts/generate_inflect_enhance_pairs.py --prompts 5000")


if __name__ == "__main__":
    main()
