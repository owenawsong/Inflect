import argparse
import re
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = PROJECT_ROOT / ".chatterbox_pkgs"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

import soundfile as sf
import torch
import numpy as np

from chatterbox.tts_turbo import ChatterboxTurboTTS


DEFAULT_PROMPTS = [
    "That is actually insane. [laugh]",
    "Wait. What was that? [gasp]",
    "I don't know anymore. [sigh]",
    "I specifically asked for one thing, and it still did not happen.",
]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:60] or "sample"


def write_wav(path: Path, audio, sample_rate: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    sf.write(str(path), audio, sample_rate, format="WAV", subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", action="append", help="Prompt to synthesize. Repeatable.")
    parser.add_argument("--prompt-file", type=Path, help="Text file with one prompt per line.")
    parser.add_argument("--audio-prompt", type=Path, help="Reference wav for voice cloning.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "chatterbox_turbo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    args = parser.parse_args()

    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompts.extend(
            line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if not prompts:
        prompts = DEFAULT_PROMPTS

    print(f"Loading Chatterbox Turbo on {args.device}...")
    model = ChatterboxTurboTTS.from_pretrained(args.device)

    run_dir = args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts, start=1):
        t0 = time.time()
        wav = model.generate(
            text=prompt,
            audio_prompt_path=str(args.audio_prompt) if args.audio_prompt else None,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
        )
        elapsed = time.time() - t0
        out_path = run_dir / f"{i:02d}_{slugify(prompt)}.wav"
        write_wav(out_path, wav)
        print(f"[{i}/{len(prompts)}] {elapsed:.2f}s -> {out_path}")

    print(f"\nDone. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
