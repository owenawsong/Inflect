"""
Inflect TTS — Full Paralinguistic Tag Engine

Synthesizes speech with natural paralinguistic events spliced in.

Sprite tags (real audio):
    [laugh]       — laughter
    [sigh]        — sigh
    [cough]       — cough
    [throat]      — throat clearing
    [sneeze]      — sneeze
    [sniff]       — sniff
    [breath]      — short breath (alias: sniff)

Synthesized filler tags (use current voice):
    [hmm]         — "hmm."
    [um]          — "um,"
    [uh]          — "uh,"
    [er]          — "er,"
    [mhm]         — "mhm."
    [huh]         — "huh?"
    [oh]          — "oh."
    [ah]          — "ah."

Effect tags (applied to next speech segment):
    [stutter]     — repeats first ~200ms of next speech
    [whisper]     — applies whisper post-processing to next speech

Pause tags:
    [pause]       — 0.5s silence
    [pause:N]     — N seconds of silence (e.g. [pause:1.5])

Aliases:
    [breath]  -> sniff sprite
    [exhale]  -> sigh sprite
    [chuckle] -> laugh sprite
    [ha]      -> laugh sprite
    [ahem]    -> throat sprite

Usage:
    python inflect_tts.py "Hello! [laugh] That's so funny."
    python inflect_tts.py "I was like, [um] I don't know. [sigh]"
    python inflect_tts.py "Wait, [stutter] what did you just say?"
    python inflect_tts.py "[whisper] Meet me at midnight."
    python inflect_tts.py "Hold on. [pause:2.0] Okay, I'm back."
    python inflect_tts.py "text here" --voice bm_fable
    python inflect_tts.py "text here" --input ref_clip.wav
    python inflect_tts.py "text here" --blend "af_heart:0.6,bm_fable:0.4"
    python inflect_tts.py "text here" --pitch 1.1
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_preprocessor import TextPreprocessor

BASE       = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
CKPT_DIR   = BASE / "checkpoints"
SPRITE_DIR = BASE / "data" / "sprites"
VOICE_DIR  = BASE / "data" / "kokoro_voices"
OUT_DIR    = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR    = 24_000
N_MELS       = 80
CROSSFADE_MS = 25  # ms for fade-in/out on segment boundaries

# Tags that map to audio sprites
SPRITE_TAGS = {"laugh", "sigh", "cough", "throat", "sneeze", "sniff"}

# Tags that get synthesized as speech with current voice
FILLER_TAGS = {
    "hmm":  "hmm.",
    "um":   "um,",
    "uh":   "uh,",
    "er":   "er,",
    "mhm":  "mhm.",
    "huh":  "huh?",
    "oh":   "oh.",
    "ah":   "ah.",
    "yeah": "yeah.",
}

# Sprite tag aliases
TAG_ALIASES = {
    "breath":  "sniff",
    "exhale":  "sigh",
    "chuckle": "laugh",
    "ha":      "laugh",
    "ahem":    "throat",
}

_mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=1024, hop_length=256,
    win_length=1024, n_mels=N_MELS, f_max=8000.0,
)


# ── Audio effects ─────────────────────────────────────────────────────────────

def apply_fade_out(audio: np.ndarray, ms: int = CROSSFADE_MS) -> np.ndarray:
    n = min(int(ms * TARGET_SR / 1000), len(audio) // 4)
    if n <= 0:
        return audio
    out = audio.copy()
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def apply_fade_in(audio: np.ndarray, ms: int = CROSSFADE_MS) -> np.ndarray:
    n = min(int(ms * TARGET_SR / 1000), len(audio) // 4)
    if n <= 0:
        return audio
    out = audio.copy()
    out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def apply_stutter(audio: np.ndarray, repeat: int = 1, fragment_ms: int = 180) -> np.ndarray:
    """Repeat first fragment_ms of audio to create a stutter."""
    n = min(int(fragment_ms * TARGET_SR / 1000), len(audio) // 3)
    if n <= 0:
        return audio
    fragment = apply_fade_out(audio[:n].copy(), ms=20)
    gap      = np.zeros(int(0.05 * TARGET_SR), dtype=np.float32)
    parts    = []
    for _ in range(repeat):
        parts.extend([fragment, gap])
    parts.append(audio)
    return np.concatenate(parts)


def apply_whisper_effect(audio: np.ndarray) -> np.ndarray:
    """Whisper post-processing: lowpass + breathiness + gain reduction."""
    from scipy.signal import butter, sosfilt
    # Cut high harmonics that voiced speech has (keep 300-3500 Hz range)
    sos = butter(4, 3500.0 / (TARGET_SR / 2), btype="low", output="sos")
    audio = sosfilt(sos, audio).astype(np.float32)
    # Add a tiny breath noise layer for texture
    noise = (np.random.randn(len(audio)) * 0.035).astype(np.float32)
    audio = audio + noise
    # Reduce to whisper amplitude
    audio = audio * 0.40
    return audio


# ── Sprite loading ────────────────────────────────────────────────────────────

def measure_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0


def load_sprite(tag: str, target_rms: float = 0.15) -> np.ndarray | None:
    import random as _random
    tag = TAG_ALIASES.get(tag, tag)

    variants = sorted(SPRITE_DIR.glob(f"{tag}_*.wav"))
    if not variants:
        single = SPRITE_DIR / f"{tag}.wav"
        variants = [single] if single.exists() else []
    if not variants:
        print(f"  [!] No sprite for [{tag}] — run download_sprites.py first")
        return None

    path  = _random.choice(variants)
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        wav   = torch.from_numpy(audio).unsqueeze(0)
        wav   = T.Resample(sr, TARGET_SR)(wav)
        audio = wav.squeeze(0).numpy()

    # Match sprite RMS to speech RMS (scaled down slightly so sprites feel natural)
    rms = measure_rms(audio)
    if rms > 0:
        audio = audio * (target_rms / rms)

    # Fade in/out edges so splice sounds clean
    audio = apply_fade_in(apply_fade_out(audio))
    return audio


# ── Text parser ───────────────────────────────────────────────────────────────

def parse_text(text: str) -> list[dict]:
    """
    Splits text into segment dicts.
    Handles: speech, sprite tags, filler tags, pause[:N], [stutter], [whisper].
    """
    pattern = r'\[(\w+(?::\d+\.?\d*)?)\]'
    parts   = re.split(pattern, text)
    result  = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            part = part.strip()
            if part:
                result.append({"type": "speech", "text": part, "speed": 1.0})
        else:
            raw = part.lower().strip()

            # pause:N — custom duration
            pause_m = re.match(r'^pause:(\d+\.?\d*)$', raw)
            if pause_m:
                result.append({"type": "pause", "duration": float(pause_m.group(1))})
                continue

            if raw == "pause":
                result.append({"type": "pause", "duration": 0.5})
                continue

            if raw == "stutter":
                result.append({"type": "stutter"})
                continue

            if raw == "whisper":
                result.append({"type": "whisper"})
                continue

            # Filler tags -> synthesized as speech with current voice
            canonical = TAG_ALIASES.get(raw, raw)
            filler_text = FILLER_TAGS.get(canonical) or FILLER_TAGS.get(raw)
            if filler_text:
                result.append({"type": "speech", "text": filler_text,
                                "speed": 0.82, "is_filler": True})
                continue

            # Sprite tag (or unknown — load_sprite will warn)
            result.append({"type": "tag", "tag": canonical})

    return result


# ── Voice loading ─────────────────────────────────────────────────────────────

def load_voice_from_clip(audio_path: str) -> torch.Tensor:
    """Encode reference audio -> Kokoro-compatible voice pack [511,1,256].

    Priority:
        1d_ref_best.pt  — reference conditioning (soft blend, best quality)
        1c_aug_best.pt  — augmented alignment
        1c_best.pt      — basic alignment
    """
    from train_ve import VoiceEncoder
    from train_1b_adapter_v2 import StyleAdapterV2

    ckpt_1d  = CKPT_DIR / "1d_ref_best.pt"
    ckpt_aug = CKPT_DIR / "1c_aug_best.pt"
    ckpt_1c  = CKPT_DIR / "1c_best.pt"

    if ckpt_1d.exists():
        ckpt = torch.load(ckpt_1d, map_location=DEVICE, weights_only=False)
        use_ref = True
    elif ckpt_aug.exists():
        ckpt = torch.load(ckpt_aug, map_location=DEVICE, weights_only=False)
        use_ref = False
    elif ckpt_1c.exists():
        ckpt = torch.load(ckpt_1c, map_location=DEVICE, weights_only=False)
        use_ref = False
    else:
        raise FileNotFoundError("No checkpoint found (need 1d_ref, 1c_aug, or 1c).")

    encoder = VoiceEncoder().to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    adapter = StyleAdapterV2().to(DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    voice_files = sorted(VOICE_DIR.glob("**/*.pt"))
    voice_names = [f.stem for f in voice_files]
    voice_packs = [torch.load(vf, map_location="cpu", weights_only=True)[:, 0, :]
                   for vf in voice_files]
    voice_packs_t = torch.stack(voice_packs).to(DEVICE)  # [N, 511, 256]

    conditioner = None
    if use_ref and "conditioner" in ckpt:
        from train_1d_reference_conditioning import ReferenceConditioner
        conditioner = ReferenceConditioner(dim=256, n_voices=len(voice_names)).to(DEVICE)
        conditioner.load_state_dict(ckpt["conditioner"])
        conditioner.eval()
        print(f"  Mode: reference conditioning (soft blend across {len(voice_names)} voices)")
    else:
        norms       = [torch.load(vf, map_location="cpu", weights_only=True)[:, 0, :]
                       .norm(dim=-1).mean().item() for vf in voice_files]
        target_norm = sum(norms) / len(norms)
        print(f"  Mode: nearest-voice mapping (fallback)")

    # Load and preprocess reference audio
    arr, sr = sf.read(audio_path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    wav = torch.from_numpy(arr).unsqueeze(0)
    if sr != TARGET_SR:
        wav = T.Resample(sr, TARGET_SR)(wav)
    wav     = wav[:, :TARGET_SR * 10]
    mel     = _mel_transform(wav)
    log_mel = torch.log(mel + 1e-5).squeeze(0)
    T_      = log_mel.shape[1]
    log_mel = log_mel[:, :160] if T_ >= 160 else F.pad(log_mel, (0, 160 - T_))
    batch   = log_mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enc_emb = encoder.embed(batch)  # [1, 256]
        if conditioner is not None:
            weights = conditioner(enc_emb)                               # [1, N]
            style   = torch.einsum("bn,ntd->btd", weights, voice_packs_t)  # [1, 511, 256]
            style   = style.unsqueeze(2)                                 # [1, 511, 1, 256]
        else:
            style = adapter(enc_emb) * target_norm

    return style.squeeze(0).cpu()  # [511, 1, 256]


# ── Synthesis ─────────────────────────────────────────────────────────────────

def synthesize_segment(pipeline, text: str, voice_key: str, speed: float) -> np.ndarray | None:
    chunks = []
    for chunk in pipeline(text, voice=voice_key, speed=speed):
        if chunk.audio is not None:
            chunks.append(chunk.audio)
    return np.concatenate(chunks) if chunks else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text",               type=str,   help="Text with [tags]")
    parser.add_argument("--voice",            type=str,   default="af_heart")
    parser.add_argument("--input",            type=str,   default=None,
                        help="Reference audio clip for voice cloning")
    parser.add_argument("--out",              type=str,   default="inflect_out")
    parser.add_argument("--speed",            type=float, default=1.0)
    parser.add_argument("--gap-before",       type=float, default=0.06,
                        help="Silence before each sprite (seconds)")
    parser.add_argument("--gap-after",        type=float, default=0.10,
                        help="Silence after each sprite (seconds)")
    parser.add_argument("--blend",            type=str,   default=None,
                        help="'voice_a:0.6,voice_b:0.4'")
    parser.add_argument("--pitch",            type=float, default=1.0,
                        help="Pitch factor (1.0=normal, 1.1=10pct higher)")
    parser.add_argument("--batch",            type=str,   default=None,
                        help="Output subfolder name. Auto-timestamp if omitted.")
    parser.add_argument("--use-preprocessor", action="store_true",
                        help="Use TextPreprocessor for per-segment speed/emotion")
    parser.add_argument("--use-llm",          action="store_true",
                        help="Load Qwen2.5-0.5B for richer understanding (requires --use-preprocessor)")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from kokoro import KPipeline

    pipeline  = KPipeline(lang_code="a")
    voice_key = args.voice

    # ── Load voice ────────────────────────────────────────────────────────────
    if args.blend:
        print(f"Blending voices: {args.blend}")
        parts        = [p.strip() for p in args.blend.split(",")]
        blend_voice  = None
        total_weight = 0.0
        for part in parts:
            name, weight = part.split(":")
            weight = float(weight)
            name   = name.strip()
            pack   = None
            try:
                if name not in pipeline.voices:
                    _ = pipeline.load_voice(name)
                if name in pipeline.voices:
                    pack = pipeline.voices[name]
            except Exception:
                pass
            if pack is None:
                p = VOICE_DIR / f"{name}.pt"
                if p.exists():
                    pack = torch.load(p, map_location="cpu", weights_only=True)
                else:
                    print(f"  WARNING: voice '{name}' not found"); continue
            blend_voice   = (pack.clone().float() * weight if blend_voice is None
                             else blend_voice + pack.float() * weight)
            total_weight += weight
            print(f"  {name}: {weight:.0%}")
        if blend_voice is not None and total_weight > 0:
            pipeline.voices["_inflect_blend"] = blend_voice / total_weight
            voice_key = "_inflect_blend"
        else:
            print("Blend failed, using default voice")

    elif args.input:
        print(f"Loading voice from: {args.input}")
        pipeline.voices["_inflect_clone"] = load_voice_from_clip(args.input)
        voice_key = "_inflect_clone"
        print("Voice loaded.")

    else:
        print(f"Using built-in voice: {voice_key}")

    # ── Parse segments ────────────────────────────────────────────────────────
    if args.use_preprocessor:
        print("\nUsing TextPreprocessor...")
        preprocessor = TextPreprocessor(use_llm=args.use_llm)
        processed    = preprocessor.process(args.text)
        segments     = processed["segments"]
    else:
        segments = parse_text(args.text)
        for seg in segments:
            if seg["type"] == "speech" and not seg.get("is_filler"):
                seg["speed"] = args.speed

    print(f"\nParsed {len(segments)} segments:")
    for seg in segments:
        t = seg["type"]
        if   t == "speech":
            label = " [filler]" if seg.get("is_filler") else ""
            print(f"  [speech{label}] \"{seg['text'][:60]}\" speed={seg.get('speed',1.0):.2f}")
        elif t == "tag":    print(f"  [sprite]  [{seg['tag']}]")
        elif t == "pause":  print(f"  [pause]   {seg.get('duration',0.5):.2f}s")
        elif t == "stutter":print(f"  [stutter] -> next speech")
        elif t == "whisper":print(f"  [whisper] -> next speech")

    # ── Synthesize ────────────────────────────────────────────────────────────
    print("\nSynthesizing...")
    audio_parts     = []
    last_speech_rms = 0.15   # initial sprite volume reference
    next_stutter    = False
    next_whisper    = False
    last_was_sprite = False  # track boundary for crossfade

    for seg in segments:
        seg_type = seg["type"]

        if seg_type == "stutter":
            next_stutter = True
            continue
        if seg_type == "whisper":
            next_whisper = True
            continue

        if seg_type == "speech":
            text  = seg["text"]
            speed = seg.get("speed", args.speed)
            label = "[filler] " if seg.get("is_filler") else ""
            print(f"  TTS {label}\"{text[:60]}\" (speed={speed:.2f})")

            audio = synthesize_segment(pipeline, text, voice_key, speed)
            if audio is None:
                print(f"    WARNING: no audio returned")
                continue

            # Apply effect flags
            if next_stutter:
                audio        = apply_stutter(audio)
                next_stutter = False
                print(f"    -> stutter applied")
            if next_whisper:
                audio        = apply_whisper_effect(audio)
                next_whisper = False
                print(f"    -> whisper effect applied")

            # Measure RMS for dynamic sprite matching
            last_speech_rms = max(measure_rms(audio), 0.05)

            # Crossfades: fade in if previous was a sprite, always fade out end
            if last_was_sprite:
                audio = apply_fade_in(audio)
            audio = apply_fade_out(audio)

            audio_parts.append(audio)
            last_was_sprite = False

        elif seg_type == "tag":
            tag    = seg["tag"]
            sprite = load_sprite(tag, target_rms=last_speech_rms * 0.82)
            if sprite is not None:
                print(f"  Sprite: [{tag}]  ({len(sprite)/TARGET_SR:.2f}s)")
                gap_b = np.zeros(int(args.gap_before * TARGET_SR), dtype=np.float32)
                gap_a = np.zeros(int(args.gap_after  * TARGET_SR), dtype=np.float32)
                audio_parts.extend([gap_b, sprite, gap_a])
                last_was_sprite = True

        elif seg_type == "pause":
            dur = seg.get("duration", 0.5)
            print(f"  Pause: {dur:.2f}s")
            audio_parts.append(np.zeros(int(dur * TARGET_SR), dtype=np.float32))
            last_was_sprite = False

    if not audio_parts:
        print("No audio generated.")
        return

    final = np.concatenate(audio_parts).astype(np.float32)

    # ── Pitch shift ───────────────────────────────────────────────────────────
    if args.pitch != 1.0:
        print(f"Pitch shifting: {args.pitch:.2f}x")
        from scipy.signal import resample
        final = resample(final, int(len(final) / args.pitch)).astype(np.float32)

    # ── Peak normalize ────────────────────────────────────────────────────────
    peak = np.abs(final).max()
    if peak > 0:
        final = final * (0.95 / peak)

    # ── Save ──────────────────────────────────────────────────────────────────
    suffix_parts = []
    if args.blend:      suffix_parts.append("blend")
    elif args.input:    suffix_parts.append("cloned")
    else:               suffix_parts.append(args.voice)
    if args.pitch != 1.0:
        suffix_parts.append(f"pitch{args.pitch:.1f}")

    # Resolve output subfolder
    from datetime import datetime
    batch_name = args.batch if args.batch else datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir  = OUT_DIR / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    suffix   = "_".join(suffix_parts)
    out_path = batch_dir / f"{args.out}_{suffix}.wav"
    sf.write(str(out_path), final, TARGET_SR)
    print(f"\nSaved: {out_path}  ({len(final)/TARGET_SR:.1f}s)")


if __name__ == "__main__":
    main()
