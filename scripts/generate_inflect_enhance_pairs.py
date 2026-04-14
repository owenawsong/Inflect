r"""
Generate (degraded, clean) audio pairs for Inflect-Enhance training.

Uses official k2-fsa/ZipVoice models to generate TTS audio (degraded),
then runs Resemble Enhance on each to produce the clean target.

Run with the voxcpm Python environment (has CUDA torch):
  .venv-voxcpm\Scripts\python.exe scripts/generate_inflect_enhance_pairs.py --prompts 10

Output layout:
  outputs/enhancer_pairs/
    degraded/      <-- ZipVoice outputs (24kHz)
    clean/         <-- Resemble Enhance outputs (48kHz)
    manifest.csv
"""

import argparse
import csv
import gc
import json
import random
import sys
import time
import traceback
import types
from pathlib import Path

import torch
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── ZipVoice-official on sys.path ────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / "ZipVoice-official"))
sys.path.insert(0, str(PROJECT_ROOT / ".resemble_enhance"))

DEFAULT_OUT_DIR  = PROJECT_ROOT / "outputs/enhancer_pairs"
REFERENCE_VOICES = PROJECT_ROOT / "reference_voices"
MANIFEST_COLS    = ["degraded_path", "clean_path", "duration_s", "voice", "prompt_text"]
HF_CACHE_ROOT    = Path.home() / ".cache" / "huggingface" / "hub" / "models--k2-fsa--ZipVoice" / "snapshots"

# ── Text prompts ──────────────────────────────────────────────────────────────
PROMPTS = [
    # Short / punchy
    ("Wait, what? I didn't hear you.", "short"),
    ("No way. Absolutely not. That's crazy.", "short"),
    ("That's actually hilarious, I love it.", "short"),
    ("I knew it. I knew this would happen.", "short"),
    ("Oh no. Oh no no no. This is bad.", "short"),
    ("This is fine. Everything is totally fine.", "short"),
    ("I have no idea what you're talking about.", "short"),
    # Conversational
    ("Hello, how are you doing today? I hope everything is going well.", "conversational"),
    ("The weather is absolutely beautiful right now, don't you think?", "conversational"),
    ("I can't believe how much time has passed since we last spoke.", "conversational"),
    ("Can you help me with this problem? I've been stuck for a while.", "conversational"),
    ("I've been thinking about what you said and I think you're right.", "conversational"),
    ("What time does the store close today? I need to pick something up.", "conversational"),
    ("I love spending time outdoors, especially in the morning.", "conversational"),
    ("That doesn't make any sense to me. Can you explain it again?", "conversational"),
    # Emotional
    ("This is really exciting news! I can't believe it finally happened!", "excited"),
    ("Oh no, I completely forgot about that! What are we going to do?", "worried"),
    ("I'm so proud of everything you've accomplished. You should be too.", "warm"),
    ("That was the most beautiful thing I've ever seen in my entire life.", "emotional"),
    ("I can't do this anymore. I'm exhausted and I just need a break.", "frustrated"),
    # Narrative
    ("The sun was setting over the mountains when she finally arrived at the cabin.", "narrative"),
    ("He had been waiting for three hours, but the call never came that evening.", "narrative"),
    ("She opened the letter slowly, as if afraid of what it might say inside.", "narrative"),
    ("The old house stood at the end of the road, silent and full of memories.", "narrative"),
    # Varied punctuation / rhythm
    ("One. Two. Three. Go! Run as fast as you can!", "rhythmic"),
    ("Wait — did you hear that? Something is out there in the dark.", "suspense"),
    ("I just... I don't know what to say. I'm completely speechless.", "sad"),
    ("Are you serious? Are you actually serious right now? I can't believe this.", "disbelief"),
    ("It was perfect. Absolutely perfect. Everything went exactly as planned.", "satisfied"),
    ("I mean... sure. If you really want to. I guess that's okay with me.", "reluctant"),
]


def _load_voice_pool():
    """Returns list of (wav_path, transcript) from reference_voices/ (only voices with .txt)."""
    pool = []
    if not REFERENCE_VOICES.exists():
        return pool
    for voice_dir in sorted(REFERENCE_VOICES.iterdir()):
        if not voice_dir.is_dir():
            continue
        for ext in ("*.wav", "*.mp3"):
            for wav in voice_dir.glob(ext):
                txt = wav.with_suffix(".txt")
                if txt.exists():
                    transcript = txt.read_text(encoding="utf-8").strip()
                    if transcript and transcript != "[NO_SPEECH]":
                        pool.append((wav, transcript))
    return pool


def _save_wav(path: Path, wav: torch.Tensor, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = wav.cpu().numpy()
    if data.ndim == 2:
        data = data.T  # (C, T) -> (T, C) for soundfile
    sf.write(str(path), data, sr, subtype="PCM_16")


def _find_cached_model_dir(model_name: str) -> Path | None:
    if not HF_CACHE_ROOT.exists():
        return None
    for snapshot_dir in sorted(HF_CACHE_ROOT.iterdir(), reverse=True):
        candidate = snapshot_dir / model_name
        if all((candidate / name).exists() for name in ("model.json", "model.pt", "tokens.txt")):
            return candidate
    return None


def _resolve_model_files(model_name: str, model_dir: Path | None):
    if model_dir is not None:
        chosen_dir = model_dir
        if not chosen_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {chosen_dir}")
        files = [chosen_dir / "model.json", chosen_dir / "model.pt", chosen_dir / "tokens.txt"]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model files in {chosen_dir}: {missing}")
        return tuple(str(p) for p in files)

    cached_dir = _find_cached_model_dir(model_name)
    if cached_dir is not None:
        print(f"Using cached {model_name} from {cached_dir}")
        return tuple(str(cached_dir / name) for name in ("model.json", "model.pt", "tokens.txt"))

    from huggingface_hub import hf_hub_download

    repo = "k2-fsa/ZipVoice"
    print(f"Downloading {model_name} from {repo}...")
    return (
        hf_hub_download(repo_id=repo, filename=f"{model_name}/model.json"),
        hf_hub_download(repo_id=repo, filename=f"{model_name}/model.pt"),
        hf_hub_download(repo_id=repo, filename=f"{model_name}/tokens.txt"),
    )


def _make_lux_anchor_sample(diffusion_wrapper):
    def _sample(
        self,
        x,
        text_condition,
        speech_condition,
        padding_mask,
        num_step=10,
        guidance_scale=0.0,
        t_start=0.0,
        t_end=1.0,
        t_shift=1.0,
        **kwargs,
    ):
        from zipvoice.models.modules.solver import get_time_steps

        device = x.device
        timesteps = get_time_steps(
            t_start=t_start,
            t_end=t_end,
            num_step=num_step,
            t_shift=t_shift,
            device=device,
        )

        for step in range(num_step):
            t_cur = timesteps[step]
            t_next = timesteps[step + 1]
            v = diffusion_wrapper(
                t=t_cur,
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs,
            )
            x_1_pred = x + (1.0 - t_cur) * v
            x_0_pred = x - t_cur * v
            if step < num_step - 1:
                x = (1.0 - t_next) * x_0_pred + t_next * x_1_pred
            else:
                x = x_1_pred
        return x

    return _sample


def _apply_solver_mode(model, solver_mode: str):
    if solver_mode == "official":
        return
    if solver_mode != "lux_anchor":
        raise ValueError(f"Unknown solver mode: {solver_mode}")
    model.solver.sample = types.MethodType(_make_lux_anchor_sample(model.solver.model), model.solver)


def _apply_runtime_preset(args):
    if args.preset == "none":
        return
    if args.preset == "inflect_base":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "official"
        args.t_shift = 0.9
        args.target_rms = 0.01
        return
    if args.preset == "inflect_base_solver":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "lux_anchor"
        args.t_shift = 0.9
        args.target_rms = 0.01
        return
    raise ValueError(f"Unknown preset: {args.preset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts",        type=int,   default=10)
    ap.add_argument("--out-dir",        type=Path,  default=DEFAULT_OUT_DIR)
    ap.add_argument("--device",         type=str,   default=None)
    ap.add_argument("--preset",         type=str,   default="none",
                    choices=["none", "inflect_base", "inflect_base_solver"])
    ap.add_argument("--model-name",     type=str,   default="zipvoice",
                    choices=["zipvoice", "zipvoice_distill"])
    ap.add_argument("--model-dir",      type=Path,  default=None,
                    help="Optional local directory containing model.json, model.pt, and tokens.txt.")
    ap.add_argument("--nfe",            type=int,   default=64,
                    help="Resemble Enhance NFE (64=quality, 1=speed)")
    ap.add_argument("--num-steps",      type=int,   default=None,
                    help="Sampling steps. Defaults follow the official model defaults.")
    ap.add_argument("--guidance-scale", type=float, default=None,
                    help="CFG scale. Defaults follow the official model defaults.")
    ap.add_argument("--solver-mode",    type=str,   default="official",
                    choices=["official", "lux_anchor"])
    ap.add_argument("--speed",          type=float, default=1.0)
    ap.add_argument("--t-shift",        type=float, default=0.5)
    ap.add_argument("--target-rms",     type=float, default=0.1)
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_defaults = {
        "zipvoice": {"num_steps": 16, "guidance_scale": 1.0},
        "zipvoice_distill": {"num_steps": 8, "guidance_scale": 3.0},
    }
    if args.num_steps is None:
        args.num_steps = model_defaults[args.model_name]["num_steps"]
    if args.guidance_scale is None:
        args.guidance_scale = model_defaults[args.model_name]["guidance_scale"]
    _apply_runtime_preset(args)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Model: {args.model_name}")
    print(f"Preset: {args.preset}")
    print(f"Sampling steps: {args.num_steps}")
    print(f"Solver mode: {args.solver_mode}")

    # ── Load ZipVoice model ───────────────────────────────────────────────────
    print(f"Loading {args.model_name}...")
    from zipvoice.models.zipvoice import ZipVoice
    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.checkpoint import load_checkpoint
    from zipvoice.utils.feature import VocosFbank
    from zipvoice.bin.infer_zipvoice import generate_sentence
    from vocos import Vocos

    model_json, model_pt, tokens_txt = _resolve_model_files(args.model_name, args.model_dir)
    with open(model_json, encoding="utf-8") as f:
        model_config = json.load(f)

    tokenizer = EmiliaTokenizer(token_file=tokens_txt)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    if args.model_name == "zipvoice":
        model = ZipVoice(**model_config["model"], **tokenizer_config)
    else:
        model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)
    load_checkpoint(model_pt, model)
    model = model.to(device).eval()
    _apply_solver_mode(model, args.solver_mode)

    if model_config["feature"]["type"].lower() not in {"vocos", "vocosfbank"}:
        raise ValueError(f"Unsupported ZipVoice feature type: {model_config['feature']['type']}")
    feature_extractor = VocosFbank()
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    # ── Load Resemble Enhance ─────────────────────────────────────────────────
    print("Loading Resemble Enhance...")
    from resemble_enhance.enhancer.inference import enhance as re_enhance

    # ── Voice pool ────────────────────────────────────────────────────────────
    voice_pool = _load_voice_pool()
    if not voice_pool:
        print("ERROR: No reference voices with transcripts found.")
        sys.exit(1)
    print(f"Voice pool: {len(voice_pool)} reference clips")

    # ── Output dirs ───────────────────────────────────────────────────────────
    degraded_dir = args.out_dir / "degraded"
    clean_dir    = args.out_dir / "clean"
    degraded_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.csv"
    write_header  = not manifest_path.exists()

    prompts_to_run = [PROMPTS[i % len(PROMPTS)] for i in range(args.prompts)]
    ok  = 0
    err = 0
    t0  = time.time()

    print(f"\nGenerating {len(prompts_to_run)} pairs...\n")

    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_COLS)
        if write_header:
            writer.writeheader()

        for i, (text, category) in enumerate(prompts_to_run, start=1):
            ref_wav_path, ref_transcript = random.choice(voice_pool)
            voice_name = ref_wav_path.parent.name

            stem     = f"pair_{i:04d}_{voice_name}"
            deg_path = degraded_dir / f"{stem}.wav"

            print(f"[{i}/{len(prompts_to_run)}] [{category}] {text[:45]}... voice={voice_name}", end=" ", flush=True)

            try:
                # Generate with ZipVoice-Distill
                with torch.inference_mode():
                    generate_sentence(
                        save_path=str(deg_path),
                        prompt_text=ref_transcript,
                        prompt_wav=str(ref_wav_path),
                        text=text,
                        model=model,
                        vocoder=vocoder,
                        tokenizer=tokenizer,
                        feature_extractor=feature_extractor,
                        device=device,
                        num_step=args.num_steps,
                        guidance_scale=args.guidance_scale,
                        speed=args.speed,
                        t_shift=args.t_shift,
                        target_rms=args.target_rms,
                    )

                # Load and check
                wav_out, sr_out = sf.read(str(deg_path), dtype="float32")
                duration = len(wav_out) / sr_out

                # Enhance with Resemble Enhance
                wav_tensor = torch.from_numpy(wav_out)
                with torch.inference_mode():
                    wav_enhanced, sr_enhanced = re_enhance(
                        wav_tensor, sr_out,
                        device=device,
                        nfe=args.nfe,
                        solver="midpoint",
                        lambd=0.9,
                        tau=0.5,
                    )

                clean_path = clean_dir / f"{stem}.wav"
                _save_wav(clean_path, wav_enhanced, sr_enhanced)

                elapsed = time.time() - t0
                rate    = int(ok / elapsed * 3600) if elapsed > 0 else 0
                print(f"OK ({duration:.1f}s audio, {rate}/hr)")

                writer.writerow({
                    "degraded_path": str(deg_path),
                    "clean_path":    str(clean_path),
                    "duration_s":    f"{duration:.2f}",
                    "voice":         voice_name,
                    "prompt_text":   text,
                })
                mf.flush()
                ok += 1
                if device == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"FAIL: {e}")
                traceback.print_exc()
                err += 1

    elapsed = time.time() - t0
    print(f"\nDone: {ok} pairs saved, {err} failed")
    print(f"Time: {elapsed/60:.1f}min")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext steps:\n  python inflect/enhancer/train.py --stage 1")


if __name__ == "__main__":
    main()
