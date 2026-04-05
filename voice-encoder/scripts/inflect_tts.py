"""
Inflect TTS — Full Paralinguistic Tag Engine

Synthesizes speech with natural paralinguistic events spliced in.

Supported tags:
    [laugh]   — laughter
    [sigh]    — sigh
    [cough]   — cough
    [throat]  — throat clearing
    [sneeze]  — sneeze
    [sniff]   — sniff
    [pause]   — 0.5s silence
    [breath]  — short breath (uses sniff sprite)

Usage:
    python inflect_tts.py "Hello! [laugh] That's so funny."
    python inflect_tts.py "I don't know... [sigh] it's complicated." --voice af_heart
    python inflect_tts.py "Excuse me. [cough] Sorry about that." --input ref_clip.wav
    python inflect_tts.py "text here" --out my_output

Args:
    text              positional — text with [tags]
    --voice           Kokoro built-in voice (default: af_heart)
    --input           reference audio clip for voice cloning (overrides --voice)
    --out             output filename stem (default: inflect_out)
    --speed           speech speed (default: 1.0)
    --gap-before      silence before each sprite in seconds (default: 0.08)
    --gap-after       silence after each sprite in seconds (default: 0.12)
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

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 24_000
N_MELS    = 80

# Aliases for tags that share a sprite
TAG_ALIASES = {
    "breath": "sniff",
    "exhale": "sigh",
    "chuckle": "laugh",
    "ha":    "laugh",
    "ahem":  "throat",
}

_mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=1024, hop_length=256,
    win_length=1024, n_mels=N_MELS, f_max=8000.0,
)


# ── Sprite loading ────────────────────────────────────────────────────────────

_sprite_cache: dict[str, np.ndarray] = {}

def load_sprite(tag: str) -> np.ndarray | None:
    import random as _random
    tag = TAG_ALIASES.get(tag, tag)

    # Find all available variants: tag_0.wav, tag_1.wav, ... or tag.wav
    variants = sorted(SPRITE_DIR.glob(f"{tag}_*.wav"))
    if not variants:
        single = SPRITE_DIR / f"{tag}.wav"
        variants = [single] if single.exists() else []
    if not variants:
        print(f"  [!] No sprite for [{tag}] — run download_sprites.py first")
        return None

    # Pick a random variant each time (never same sound twice in a row)
    path = _random.choice(variants)

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        wav = torch.from_numpy(audio).unsqueeze(0)
        wav = T.Resample(sr, TARGET_SR)(wav)
        audio = wav.squeeze(0).numpy()
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio * (0.15 / rms)
    return audio


# ── Text parser ───────────────────────────────────────────────────────────────

def parse_text(text: str) -> list[tuple[str, str]]:
    """
    Splits text into (type, content) segments.
    type = 'speech' | 'tag' | 'pause'

    E.g. "Hello! [laugh] How are you?"
      -> [('speech','Hello!'), ('tag','laugh'), ('speech','How are you?')]
    """
    pattern = r'\[(\w+)\]'
    parts   = re.split(pattern, text)
    result  = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            part = part.strip()
            if part:
                result.append(("speech", part))
        else:
            tag = part.lower().strip()
            if tag == "pause":
                result.append(("pause", ""))
            else:
                result.append(("tag", tag))
    return result


# ── Voice loading ─────────────────────────────────────────────────────────────

def load_voice_from_clip(audio_path: str) -> torch.Tensor:
    """Encode a reference audio clip -> voice pack [511,1,256].

    Tries 1d_ref (reference conditioning with soft voice blending) first,
    falls back to 1c_aug (nearest-voice mapping) if not available.
    """
    from train_ve import VoiceEncoder
    from train_1b_adapter_v2 import StyleAdapterV2

    ckpt_1d  = CKPT_DIR / "1d_ref_best.pt"
    ckpt_aug = CKPT_DIR / "1c_aug_best.pt"
    ckpt_1c  = CKPT_DIR / "1c_best.pt"

    use_ref_conditioning = ckpt_1d.exists()

    if use_ref_conditioning:
        ckpt = torch.load(ckpt_1d, map_location=DEVICE, weights_only=False)
    elif ckpt_aug.exists():
        ckpt = torch.load(ckpt_aug, map_location=DEVICE, weights_only=False)
    elif ckpt_1c.exists():
        ckpt = torch.load(ckpt_1c, map_location=DEVICE, weights_only=False)
    else:
        raise FileNotFoundError("No checkpoint found (need 1d_ref, 1c_aug, or 1c).")

    encoder = VoiceEncoder().to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    adapter = StyleAdapterV2().to(DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    # Load reference voice packs for blending (used by 1d_ref)
    voice_files = sorted(VOICE_DIR.glob("**/*.pt"))
    voice_names = [f.stem for f in voice_files]
    voice_packs = []
    for vf in voice_files:
        pack = torch.load(vf, map_location="cpu", weights_only=True)  # [511,1,256]
        voice_packs.append(pack[:, 0, :])  # [511, 256]
    voice_packs_t = torch.stack(voice_packs).to(DEVICE)  # [N, 511, 256]

    # Load conditioner if available
    conditioner = None
    if use_ref_conditioning and "conditioner" in ckpt:
        from train_1d_reference_conditioning import ReferenceConditioner
        conditioner = ReferenceConditioner(dim=256, n_voices=len(voice_names)).to(DEVICE)
        conditioner.load_state_dict(ckpt["conditioner"])
        conditioner.eval()
        print(f"  Using reference conditioning (soft voice blending across {len(voice_names)} voices)")
    else:
        # Compute target norm for nearest-voice fallback
        norms = []
        for vf in voice_files:
            pack = torch.load(vf, map_location="cpu", weights_only=True)
            norms.append(pack[:, 0, :].norm(dim=-1).mean().item())
        target_norm = sum(norms) / len(norms)

    arr, sr = sf.read(audio_path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    wav = torch.from_numpy(arr).unsqueeze(0)
    if sr != TARGET_SR:
        wav = T.Resample(sr, TARGET_SR)(wav)
    wav = wav[:, :TARGET_SR * 10]
    mel = _mel_transform(wav)
    log_mel = torch.log(mel + 1e-5).squeeze(0)
    T_ = log_mel.shape[1]
    log_mel = log_mel[:, :160] if T_ >= 160 else F.pad(log_mel, (0, 160 - T_))
    batch = log_mel.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enc_emb = encoder.embed(batch)  # [1, 256]
        if conditioner is not None:
            # Soft blending: weighted sum of all voice packs
            weights = conditioner(enc_emb)  # [1, N]
            style = torch.einsum("bn,ntd->btd", weights, voice_packs_t)  # [1, 511, 256]
            style = style.unsqueeze(2)  # [1, 511, 1, 256]
        else:
            # Nearest-voice mapping (fallback)
            style = adapter(enc_emb) * target_norm
    return style.squeeze(0).cpu()  # [511,1,256]


# ── Synthesis ─────────────────────────────────────────────────────────────────

def synthesize_segment(pipeline, text: str, voice_key: str, speed: float) -> np.ndarray | None:
    chunks = []
    for chunk in pipeline(text, voice=voice_key, speed=speed):
        if chunk.audio is not None:
            chunks.append(chunk.audio)
    if not chunks:
        return None
    return np.concatenate(chunks)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text",         type=str,   help="Text with [tags]")
    parser.add_argument("--voice",      type=str,   default="af_heart",
                        help="Built-in Kokoro voice (ignored if --input is set)")
    parser.add_argument("--input",      type=str,   default=None,
                        help="Reference audio clip for voice cloning")
    parser.add_argument("--out",        type=str,   default="inflect_out")
    parser.add_argument("--speed",      type=float, default=1.0)
    parser.add_argument("--gap-before", type=float, default=0.08,
                        help="Silence before each sprite (seconds)")
    parser.add_argument("--gap-after",  type=float, default=0.12,
                        help="Silence after each sprite (seconds)")
    parser.add_argument("--blend",      type=str,   default=None,
                        help="Blend two voices: 'voice_a:0.6,voice_b:0.4' or 'bf_isabella:0.7,am_michael:0.3'")
    parser.add_argument("--pitch",          type=float, default=1.0,
                        help="Pitch shift factor (1.0 normal, 1.1 is 10 percent higher)")
    parser.add_argument("--use-preprocessor", action="store_true",
                        help="Use text_preprocessor for per-segment speed/emotion (default: off)")
    parser.add_argument("--use-llm",         action="store_true",
                        help="Load Qwen2.5-0.5B for richer text understanding (only with --use-preprocessor)")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from kokoro import KPipeline

    pipeline  = KPipeline(lang_code="a")
    voice_key = args.voice

    # Load voice
    if args.blend:
        print(f"Blending voices: {args.blend}")
        parts = [p.strip() for p in args.blend.split(",")]
        blend_voice = None
        total_weight = 0.0
        for part in parts:
            voice_name, weight = part.split(":")
            weight = float(weight)
            voice_name = voice_name.strip()

            # First try loading from HF (pipeline.voices will auto-download)
            # Then try local file
            pack = None
            try:
                if voice_name not in pipeline.voices:
                    _ = pipeline.load_voice(voice_name)
                if voice_name in pipeline.voices:
                    pack = pipeline.voices[voice_name]
            except Exception:
                pass

            if pack is None:
                pack_path = VOICE_DIR / f"{voice_name}.pt"
                if pack_path.exists():
                    pack = torch.load(pack_path, map_location="cpu", weights_only=True)
                else:
                    print(f"  WARNING: voice '{voice_name}' not found")
                    continue

            if blend_voice is None:
                blend_voice = pack.clone().float() * weight
            else:
                blend_voice = blend_voice + pack.float() * weight
            total_weight += weight
            print(f"  {voice_name}: {weight:.0%}")

        if blend_voice is not None and total_weight > 0:
            blend_voice = blend_voice / total_weight
            voice_key = "_inflect_blend"
            pipeline.voices[voice_key] = blend_voice
            print(f"Blended voice created: {voice_key}")
        else:
            print("Blend failed, using default voice")
    elif args.input:
        print(f"Loading voice from: {args.input}")
        voice_pack = load_voice_from_clip(args.input)
        voice_key  = "_inflect_clone"
        pipeline.voices[voice_key] = voice_pack
        print(f"Voice cloned.")
    else:
        print(f"Using built-in voice: {voice_key}")

    # Parse text (optionally via preprocessor for per-segment speed)
    if args.use_preprocessor:
        print("\nUsing text preprocessor for per-segment speed/emotion...")
        preprocessor = TextPreprocessor(use_llm=args.use_llm)
        processed = preprocessor.process(args.text)
        segments = processed["segments"]
        print(f"\nParsed {len(segments)} segments:")
        for i, seg in enumerate(segments):
            if seg["type"] == "speech":
                print(f"  {i}. SPEECH: \"{seg['text'][:60]}\" emotion={seg['emotion']}, speed={seg['speed']:.2f}")
            elif seg["type"] == "tag":
                print(f"  {i}. TAG: [{seg['tag']}] intensity={seg.get('intensity', '?')}")
    else:
        # Use simple parser
        raw_segments = parse_text(args.text)
        # Convert to preprocessor format
        segments = []
        for seg_type, content in raw_segments:
            if seg_type == "speech":
                segments.append({"type": "speech", "text": content, "speed": args.speed})
            elif seg_type == "tag":
                segments.append({"type": "tag", "tag": content})
            elif seg_type == "pause":
                segments.append({"type": "pause"})
        print(f"\nParsed {len(segments)} segments:")
        for i, seg in enumerate(segments):
            print(f"  [{seg['type']}] {seg.get('text', seg.get('tag', ''))[:60] if seg.get('text') or seg.get('tag') else ''}")

    # Silence buffers
    gap_before = np.zeros(int(args.gap_before * TARGET_SR), dtype=np.float32)
    gap_after  = np.zeros(int(args.gap_after  * TARGET_SR), dtype=np.float32)
    pause_clip = np.zeros(int(0.5 * TARGET_SR), dtype=np.float32)

    # Synthesize
    print("\nSynthesizing...")
    audio_parts = []

    for seg in segments:
        seg_type = seg["type"]
        if seg_type == "speech":
            text  = seg["text"]
            speed = seg.get("speed", args.speed)
            print(f"  TTS: \"{text[:60]}\" (speed={speed:.2f})")
            audio = synthesize_segment(pipeline, text, voice_key, speed)
            if audio is not None:
                audio_parts.append(audio)
            else:
                print(f"    WARNING: no audio for segment")

        elif seg_type == "tag":
            tag = seg["tag"]
            sprite = load_sprite(tag)
            if sprite is not None:
                print(f"  Sprite: [{tag}]  ({len(sprite)/TARGET_SR:.2f}s)")
                audio_parts.append(gap_before)
                audio_parts.append(sprite)
                audio_parts.append(gap_after)

        elif seg_type == "pause":
            print(f"  Pause: 0.5s")
            audio_parts.append(pause_clip)

    if not audio_parts:
        print("No audio generated.")
        return

    final = np.concatenate(audio_parts).astype(np.float32)

    # Pitch shift if requested
    if args.pitch != 1.0:
        print(f"Pitch shifting: {args.pitch:.2f}x")
        from scipy.signal import resample
        # Simple pitch shift: resample
        n_samples = len(final)
        n_new = int(n_samples / args.pitch)
        final = resample(final, n_new)
        final = final.astype(np.float32)

    # Normalize output
    peak = np.abs(final).max()
    if peak > 0:
        final = final * (0.95 / peak)

    # Build unique filename from voice + options
    suffix_parts = []
    if args.blend:
        suffix_parts.append("blend")
    elif args.input:
        suffix_parts.append("cloned")
    else:
        suffix_parts.append(args.voice)
    if args.pitch != 1.0:
        suffix_parts.append(f"pitch{args.pitch:.1f}")

    suffix = "_".join(suffix_parts) if suffix_parts else "default"
    out_path = OUT_DIR / f"{args.out}_{suffix}.wav"
    sf.write(str(out_path), final, TARGET_SR)
    print(f"\nSaved: {out_path}  ({len(final)/TARGET_SR:.1f}s)")


if __name__ == "__main__":
    main()
