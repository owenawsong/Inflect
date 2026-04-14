r"""
Run the real LuxTTS model on the same benchmark voices/prompts used for ZipVoice tests.

Examples:
  .\.venv-voxcpm\Scripts\python.exe scripts\test_true_lux.py
  .\.venv-voxcpm\Scripts\python.exe scripts\test_true_lux.py --voices jessica,henry --prompt-ids 06_emotional,08_clone_stress
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LUX_ROOT = PROJECT_ROOT / "third_party" / "luxtts_clean"
LINA_ROOT = PROJECT_ROOT / "third_party" / "LinaCodec" / "src"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "zipvoice_bench"
REFERENCE_ROOT = PROJECT_ROOT / "reference_voices"

# Avoid importing the local project package named `inflect` instead of the PyPI package
sys.path = [p for p in sys.path if Path(p or ".").resolve() != PROJECT_ROOT.resolve()]
sys.path.insert(0, str(LINA_ROOT.resolve()))
sys.path.insert(0, str(LUX_ROOT.resolve()))

import zipvoice.luxvoice as luxvoice_module  # noqa: E402
from zipvoice.luxvoice import LuxTTS  # noqa: E402


BENCHMARK_VOICES = [
    "shannon",
    "clara",
    "jessica",
    "adam_american",
    "michael",
    "henry",
]


BENCHMARK_PROMPTS = [
    ("01_short_neutral", "I need a quick answer right now."),
    ("02_short_conversational", "Hey, can you send that over when you have a minute?"),
    ("03_medium_conversational", "I was thinking about what you said earlier, and I think you're probably right."),
    ("04_long_narrative", "By the time we reached the last station, everyone in the car was quiet, and it felt like the whole day had been moving toward that moment."),
    ("05_punctuation", "Wait, what? No, no, no. That is not what I meant at all."),
    ("06_emotional", "I really wanted this to work, and I'm honestly trying not to sound disappointed right now."),
    ("07_disbelief", "Are you serious? You're telling me this happened again, after all of that?"),
    ("08_clone_stress", "The final version looked simple on the surface, but keeping it stable under pressure turned out to be much harder than anyone expected."),
]


def load_voice(voice_name: str):
    voice_dir = REFERENCE_ROOT / voice_name
    wav = voice_dir / f"{voice_name}.wav"
    mp3 = voice_dir / f"{voice_name}.mp3"
    audio_path = wav if wav.exists() else mp3
    txt_path = voice_dir / f"{voice_name}.txt"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing prompt audio for {voice_name}")
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing transcript for {voice_name}")
    return audio_path, txt_path.read_text(encoding="utf-8").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--voices", type=str, default=None, help="Comma-separated benchmark voice names to run.")
    ap.add_argument("--prompt-ids", type=str, default=None, help="Comma-separated prompt ids to run.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--t-shift", type=float, default=0.9)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--rms", type=float, default=0.01)
    ap.add_argument("--return-smooth", action="store_true")
    ap.add_argument("--ref-duration", type=float, default=10.0)
    ap.add_argument("--keep-speed-hack", action="store_true",
                    help="Use Lux's original internal speed*1.3 behavior.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    selected_voices = BENCHMARK_VOICES
    if args.voices:
        wanted = {x.strip() for x in args.voices.split(",") if x.strip()}
        selected_voices = [v for v in BENCHMARK_VOICES if v in wanted]
        if not selected_voices:
            raise ValueError("No benchmark voices matched --voices")

    selected_prompts = BENCHMARK_PROMPTS
    if args.prompt_ids:
        wanted = {x.strip() for x in args.prompt_ids.split(",") if x.strip()}
        selected_prompts = [item for item in BENCHMARK_PROMPTS if item[0] in wanted]
        if not selected_prompts:
            raise ValueError("No benchmark prompts matched --prompt-ids")

    neutralize_speed_hack = not args.keep_speed_hack
    variant_name = "lux_tts_real_normal_speed" if neutralize_speed_hack else "lux_tts_real"
    out_dir = args.out_root / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Variant: {variant_name}")
    print("Loading LuxTTS...")
    lux = LuxTTS("YatharthS/LuxTTS", device=device, load_transcriber=False)
    if neutralize_speed_hack:
        original_generate = luxvoice_module.generate

        def neutralized_generate(
            prompt_tokens,
            prompt_features_lens,
            prompt_features,
            prompt_rms,
            text,
            model,
            vocoder,
            tokenizer,
            num_step=4,
            guidance_scale=3.0,
            speed=1.0,
            t_shift=0.5,
            target_rms=0.1,
        ):
            return original_generate(
                prompt_tokens,
                prompt_features_lens,
                prompt_features,
                prompt_rms,
                text,
                model,
                vocoder,
                tokenizer,
                num_step=num_step,
                guidance_scale=guidance_scale,
                speed=speed / 1.3,
                t_shift=t_shift,
                target_rms=target_rms,
            )

        luxvoice_module.generate = neutralized_generate

    run_meta = {
        "variant_name": variant_name,
        "model_name": "lux_tts_real",
        "num_steps": args.num_steps,
        "t_shift": args.t_shift,
        "speed": args.speed,
        "rms": args.rms,
        "ref_duration": args.ref_duration,
        "return_smooth": args.return_smooth,
        "neutralize_speed_hack": neutralize_speed_hack,
        "device": device,
        "voices": selected_voices,
        "prompts": [{"id": pid, "text": text} for pid, text in selected_prompts],
        "results": [],
    }

    total = len(selected_voices) * len(selected_prompts)
    done = 0
    started = time.time()
    for voice_name in selected_voices:
        ref_audio, transcript = load_voice(voice_name)
        print(f"Encoding prompt: {voice_name}")
        with torch.inference_mode():
            encoded = lux.encode_prompt(
                str(ref_audio),
                duration=args.ref_duration,
                rms=args.rms,
                transcript=transcript,
            )

        voice_out_dir = out_dir / voice_name
        voice_out_dir.mkdir(parents=True, exist_ok=True)
        for prompt_id, text in selected_prompts:
            done += 1
            wav_out = voice_out_dir / f"{prompt_id}.wav"
            print(f"[{done}/{total}] {variant_name} voice={voice_name} prompt={prompt_id}", flush=True)
            t0 = time.time()
            with torch.inference_mode():
                wav = lux.generate_speech(
                    text,
                    encoded,
                    num_steps=args.num_steps,
                    t_shift=args.t_shift,
                    speed=args.speed,
                    return_smooth=args.return_smooth,
                )
            elapsed = time.time() - t0
            arr = wav.numpy().squeeze()
            sf.write(str(wav_out), arr, 48000, subtype="PCM_16")
            run_meta["results"].append(
                {
                    "voice": voice_name,
                    "prompt_id": prompt_id,
                    "text": text,
                    "wav_path": str(wav_out),
                    "t": elapsed,
                    "wav_seconds": len(arr) / 48000,
                    "rtf": elapsed / (len(arr) / 48000),
                }
            )
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

    run_meta["total_seconds"] = time.time() - started
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"\nDone. Outputs: {out_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
