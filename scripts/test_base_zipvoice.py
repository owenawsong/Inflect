r"""
Run a fixed ZipVoice benchmark grid for repeatable A/B comparisons.

Example:
  .\.venv-voxcpm\Scripts\python.exe scripts\test_base_zipvoice.py --model-name zipvoice
  .\.venv-voxcpm\Scripts\python.exe scripts\test_base_zipvoice.py --model-name zipvoice_distill --num-steps 4
"""

import argparse
import gc
import json
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ZipVoice-official"))

import numpy as np
import soundfile as sf
import torch
from vocos import Vocos
from torch.nn.utils import parametrize

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)


HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--k2-fsa--ZipVoice" / "snapshots"
REFERENCE_ROOT = PROJECT_ROOT / "reference_voices"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "zipvoice_bench"
LUX_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--YatharthS--LuxTTS" / "snapshots" / "527f245a276a0eb42ea103a7a512bcfd771eb9b6"


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


def find_cached_model_dir(model_name: str) -> Path | None:
    if not HF_CACHE_ROOT.exists():
        return None
    for snapshot_dir in sorted(HF_CACHE_ROOT.iterdir(), reverse=True):
        candidate = snapshot_dir / model_name
        if all((candidate / name).exists() for name in ("model.json", "model.pt", "tokens.txt")):
            return candidate
    return None


def resolve_model_files(model_name: str, model_dir: Path | None):
    if model_dir is not None:
        files = [model_dir / "model.json", model_dir / "model.pt", model_dir / "tokens.txt"]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model files: {missing}")
        return tuple(str(p) for p in files)

    cached_dir = find_cached_model_dir(model_name)
    if cached_dir is None:
        raise FileNotFoundError(
            f"No cached checkpoint found for {model_name}. "
            f"Download it first or pass --model-dir."
        )
    return tuple(str(cached_dir / name) for name in ("model.json", "model.pt", "tokens.txt"))


def load_voice(voice_name: str):
    voice_dir = REFERENCE_ROOT / voice_name
    if not voice_dir.exists():
        raise FileNotFoundError(f"Missing voice directory: {voice_dir}")
    wav = voice_dir / f"{voice_name}.wav"
    mp3 = voice_dir / f"{voice_name}.mp3"
    audio_path = wav if wav.exists() else mp3
    txt_path = voice_dir / f"{voice_name}.txt"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing prompt audio for {voice_name}")
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing transcript for {voice_name}")
    return audio_path, txt_path.read_text(encoding="utf-8").strip()


def prepare_prompt_audio(source_path: Path, out_path: Path, duration_cap: float | None):
    if duration_cap is None:
        return source_path

    audio, sr = sf.read(str(source_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    max_samples = int(duration_cap * sr)
    clipped = audio[:max_samples]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clipped.astype(np.float32), sr, subtype="PCM_16")
    return out_path


def default_variant_name(model_name: str, num_steps: int, guidance_scale: float):
    return f"{model_name}_{num_steps}step_cfg{guidance_scale}".replace(".", "p")


def preset_variant_name(preset: str):
    names = {
        "lux_stack": "inflect_trackA_lux_stack",
        "lux_stack_no48k": "inflect_trackA_lux_stack_no48k",
        "lux_stack_safe": "inflect_trackA_lux_stack_safe",
        "lux_stack_safe_no48k": "inflect_trackA_lux_stack_safe_no48k",
        "tracka_tuned": "inflect_trackA_tuned",
        "tracka_tuned_no48k": "inflect_trackA_tuned_no48k",
        "inflect_base": "inflect_base",
        "inflect_base_solver": "inflect_base_solver",
    }
    return names.get(preset)


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


def apply_solver_mode(model, solver_mode: str):
    if solver_mode == "official":
        return
    if solver_mode != "lux_anchor":
        raise ValueError(f"Unknown solver mode: {solver_mode}")
    model.solver.sample = types.MethodType(_make_lux_anchor_sample(model.solver.model), model.solver)


def decode_with_vocoder(vocoder, pred_features):
    decoded = vocoder.decode(pred_features)
    if decoded.ndim == 3 and decoded.shape[1] == 1:
        decoded = decoded.squeeze(1)
    return decoded.clamp(-1, 1)


def load_benchmark_vocoder(vocoder_mode: str, device: str):
    if vocoder_mode == "stock24k":
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
        return vocoder, 24000

    if vocoder_mode != "lux48k":
        raise ValueError(f"Unknown vocoder mode: {vocoder_mode}")

    sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "LinaCodec" / "src"))
    from linacodec.vocoder.vocos import Vocos as LinaVocos

    config_path = LUX_SNAPSHOT / "vocoder" / "config.yaml"
    state_path = LUX_SNAPSHOT / "vocoder" / "vocos.bin"
    if not config_path.exists() or not state_path.exists():
        raise FileNotFoundError("Missing cached Lux vocoder files.")

    vocoder = LinaVocos.from_hparams(str(config_path)).to(device).eval()
    upsampler = getattr(vocoder, "upsampler", None)
    upsample_layers = getattr(upsampler, "upsample_layers", None)
    if upsample_layers is not None:
        for idx in range(len(upsample_layers)):
            try:
                parametrize.remove_parametrizations(upsample_layers[idx], "weight")
            except Exception:
                pass
    state_dict = torch.load(str(state_path), map_location=device)
    vocoder.load_state_dict(state_dict)
    vocoder.freq_range = 12000
    vocoder.return_48k = True
    return vocoder, 48000


def generate_sentence_for_benchmark(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: str,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    prompt_sampling_rate: int = 24000,
    output_sampling_rate: int = 24000,
    max_duration: float = 100,
    remove_long_sil: bool = False,
):
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=prompt_sampling_rate)
    prompt_wav = remove_silence(prompt_wav, prompt_sampling_rate, only_edge=False, trail_sil=200)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / prompt_sampling_rate
    prompt_features = feature_extractor.extract(prompt_wav, sampling_rate=prompt_sampling_rate).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)
    tokens_str = tokenizer.texts_to_tokens([text])[0]
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]

    token_duration = (prompt_wav.shape[-1] / prompt_sampling_rate) / (len(prompt_tokens_str) * speed)
    max_tokens = int((25 - prompt_duration) / token_duration)
    chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
    chunked_tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
    prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])
    tokens_batches, chunked_index = batchify_tokens(chunked_tokens, max_duration, prompt_duration, token_duration)

    chunked_features = []
    start_t = time.time()
    for batch_tokens in tokens_batches:
        batch_prompt_tokens = prompt_tokens * len(batch_tokens)
        batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = torch.full((len(batch_tokens),), prompt_features.size(1), device=device)
        pred_features, pred_features_lens, _, _ = model.sample(
            tokens=batch_tokens,
            prompt_tokens=batch_prompt_tokens,
            prompt_features=batch_prompt_features,
            prompt_features_lens=batch_prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )
        pred_features = pred_features.permute(0, 2, 1) / feat_scale
        chunked_features.append((pred_features, pred_features_lens))

    chunked_wavs = []
    start_vocoder_t = time.time()
    for pred_features, pred_features_lens in chunked_features:
        for i in range(pred_features.size(0)):
            wav = decode_with_vocoder(vocoder, pred_features[i][None, :, : pred_features_lens[i]])
            if prompt_rms < target_rms:
                wav = wav * prompt_rms / target_rms
            chunked_wavs.append(wav)

    indexed_chunked_wavs = [(index, wav) for index, wav in zip(chunked_index, chunked_wavs)]
    sequential_chunked_wavs = [wav for _, wav in sorted(indexed_chunked_wavs, key=lambda x: x[0])]
    final_wav = cross_fade_concat(sequential_chunked_wavs, fade_duration=0.1, sample_rate=output_sampling_rate)
    final_wav = remove_silence(final_wav, output_sampling_rate, only_edge=(not remove_long_sil), trail_sil=0)

    metrics = {
        "t": time.time() - start_t,
        "t_no_vocoder": start_vocoder_t - start_t,
        "t_vocoder": time.time() - start_vocoder_t,
        "wav_seconds": final_wav.shape[-1] / output_sampling_rate,
    }
    metrics["rtf"] = metrics["t"] / metrics["wav_seconds"]
    metrics["rtf_no_vocoder"] = metrics["t_no_vocoder"] / metrics["wav_seconds"]
    metrics["rtf_vocoder"] = metrics["t_vocoder"] / metrics["wav_seconds"]

    sf.write(save_path, final_wav.squeeze(0).cpu().numpy(), output_sampling_rate, subtype="PCM_16")
    return metrics


def generate_sentence_lux_like(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: str,
    num_step: int = 4,
    guidance_scale: float = 3.0,
    speed: float = 1.0,
    lux_speed_multiplier: float = 1.3,
    t_shift: float = 0.9,
    target_rms: float = 0.01,
    feat_scale: float = 0.1,
    prompt_sampling_rate: int = 24000,
    output_sampling_rate: int = 24000,
):
    prompt_wav_tensor = load_prompt_wav(prompt_wav, sampling_rate=prompt_sampling_rate)
    prompt_wav_tensor, prompt_rms = rms_norm(prompt_wav_tensor, target_rms)
    prompt_features = feature_extractor.extract(prompt_wav_tensor, sampling_rate=prompt_sampling_rate).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    tokens = tokenizer.texts_to_token_ids([text])

    run_speed = speed * lux_speed_multiplier
    start_t = time.time()
    pred_features, _, _, _ = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=run_speed,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )
    pred_features = pred_features.permute(0, 2, 1) / feat_scale

    start_vocoder_t = time.time()
    wav = decode_with_vocoder(vocoder, pred_features)
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms

    metrics = {
        "t": time.time() - start_t,
        "t_no_vocoder": start_vocoder_t - start_t,
        "t_vocoder": time.time() - start_vocoder_t,
        "wav_seconds": wav.shape[-1] / output_sampling_rate,
    }
    metrics["rtf"] = metrics["t"] / metrics["wav_seconds"]
    metrics["rtf_no_vocoder"] = metrics["t_no_vocoder"] / metrics["wav_seconds"]
    metrics["rtf_vocoder"] = metrics["t_vocoder"] / metrics["wav_seconds"]

    sf.write(save_path, wav.squeeze(0).cpu().numpy(), output_sampling_rate, subtype="PCM_16")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="zipvoice", choices=["zipvoice", "zipvoice_distill"])
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--voices", type=str, default=None, help="Comma-separated benchmark voice names to run.")
    ap.add_argument("--prompt-ids", type=str, default=None, help="Comma-separated prompt ids to run.")
    ap.add_argument("--preset", type=str, default="none",
                    choices=[
                        "none",
                        "lux_stack",
                        "lux_stack_no48k",
                        "lux_stack_safe",
                        "lux_stack_safe_no48k",
                        "tracka_tuned",
                        "tracka_tuned_no48k",
                        "inflect_base",
                        "inflect_base_solver",
                    ],
                    help="Apply a bundled large-change runtime preset.")
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--guidance-scale", type=float, default=None)
    ap.add_argument("--solver-mode", type=str, default="official", choices=["official", "lux_anchor"])
    ap.add_argument("--vocoder-mode", type=str, default="stock24k", choices=["stock24k", "lux48k"])
    ap.add_argument("--generation-mode", type=str, default="official", choices=["official", "lux_like"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--lux-speed-multiplier", type=float, default=1.3)
    ap.add_argument("--t-shift", type=float, default=0.5)
    ap.add_argument("--target-rms", type=float, default=0.1)
    ap.add_argument("--max-duration", type=float, default=100.0)
    ap.add_argument("--prompt-duration-cap", type=float, default=None,
                    help="Optional reference prompt cap in seconds, applied before ZipVoice preprocessing.")
    ap.add_argument("--remove-long-sil", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    defaults = {
        "zipvoice": {"num_steps": 16, "guidance_scale": 1.0},
        "zipvoice_distill": {"num_steps": 8, "guidance_scale": 3.0},
    }
    if args.num_steps is None:
        args.num_steps = defaults[args.model_name]["num_steps"]
    if args.guidance_scale is None:
        args.guidance_scale = defaults[args.model_name]["guidance_scale"]

    if args.preset == "lux_stack":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "lux_anchor"
        args.vocoder_mode = "lux48k"
        args.generation_mode = "lux_like"
        args.t_shift = 0.9
        args.target_rms = 0.01
        if args.prompt_duration_cap is None:
            args.prompt_duration_cap = 5.0
    elif args.preset == "lux_stack_no48k":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "lux_anchor"
        args.vocoder_mode = "stock24k"
        args.generation_mode = "lux_like"
        args.t_shift = 0.9
        args.target_rms = 0.01
        if args.prompt_duration_cap is None:
            args.prompt_duration_cap = 5.0
    elif args.preset == "lux_stack_safe":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "lux_anchor"
        args.vocoder_mode = "lux48k"
        args.generation_mode = "official"
        args.t_shift = 0.9
        args.target_rms = 0.01
        if args.prompt_duration_cap is None:
            args.prompt_duration_cap = 5.0
    elif args.preset == "lux_stack_safe_no48k":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "lux_anchor"
        args.vocoder_mode = "stock24k"
        args.generation_mode = "official"
        args.t_shift = 0.9
        args.target_rms = 0.01
        if args.prompt_duration_cap is None:
            args.prompt_duration_cap = 5.0
    elif args.preset == "tracka_tuned":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "lux_anchor"
        args.vocoder_mode = "lux48k"
        args.generation_mode = "official"
        args.t_shift = 0.9
        args.target_rms = 0.01
        args.prompt_duration_cap = None
    elif args.preset == "tracka_tuned_no48k":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "lux_anchor"
        args.vocoder_mode = "stock24k"
        args.generation_mode = "official"
        args.t_shift = 0.9
        args.target_rms = 0.01
        args.prompt_duration_cap = None
    elif args.preset == "inflect_base":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "official"
        args.vocoder_mode = "lux48k"
        args.generation_mode = "official"
        args.t_shift = 0.9
        args.target_rms = 0.01
        args.prompt_duration_cap = None
    elif args.preset == "inflect_base_solver":
        args.model_name = "zipvoice_distill"
        args.num_steps = 4
        args.guidance_scale = 3.0
        args.solver_mode = "lux_anchor"
        args.vocoder_mode = "lux48k"
        args.generation_mode = "official"
        args.t_shift = 0.9
        args.target_rms = 0.01
        args.prompt_duration_cap = None

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

    if args.preset != "none":
        variant_name = preset_variant_name(args.preset) or default_variant_name(
            args.model_name, args.num_steps, args.guidance_scale
        )
    else:
        variant_name = default_variant_name(args.model_name, args.num_steps, args.guidance_scale)
        if args.solver_mode != "official":
            variant_name += f"_{args.solver_mode}"
        if args.t_shift != 0.5:
            variant_name += f"_t{args.t_shift}".replace(".", "p")
        if args.speed != 1.0:
            variant_name += f"_speed{args.speed}".replace(".", "p")
        if args.target_rms != 0.1:
            variant_name += f"_rms{args.target_rms}".replace(".", "p")
        if args.prompt_duration_cap is not None:
            variant_name += f"_pcap{args.prompt_duration_cap}".replace(".", "p")
        if args.vocoder_mode != "stock24k":
            variant_name += f"_{args.vocoder_mode}"
        if args.generation_mode != "official":
            variant_name += f"_{args.generation_mode}"
    out_dir = args.out_root / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Variant: {variant_name}")

    model_json, model_pt, tokens_txt = resolve_model_files(args.model_name, args.model_dir)
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
    apply_solver_mode(model, args.solver_mode)
    feature_extractor = VocosFbank()
    vocoder, output_sampling_rate = load_benchmark_vocoder(args.vocoder_mode, device)

    run_meta = {
        "variant_name": variant_name,
        "model_name": args.model_name,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "solver_mode": args.solver_mode,
        "vocoder_mode": args.vocoder_mode,
        "generation_mode": args.generation_mode,
        "preset": args.preset,
        "speed": args.speed,
        "lux_speed_multiplier": args.lux_speed_multiplier,
        "t_shift": args.t_shift,
        "target_rms": args.target_rms,
        "prompt_duration_cap": args.prompt_duration_cap,
        "device": device,
        "voices": selected_voices,
        "prompts": [{"id": pid, "text": text} for pid, text in selected_prompts],
        "results": [],
    }

    total = len(selected_voices) * len(selected_prompts)
    done = 0
    started = time.time()
    for voice_name in selected_voices:
        prompt_wav_src, prompt_text = load_voice(voice_name)
        prompt_wav = prepare_prompt_audio(
            prompt_wav_src,
            out_dir / "_prompt_cache" / f"{voice_name}.wav",
            args.prompt_duration_cap,
        )
        voice_out_dir = out_dir / voice_name
        voice_out_dir.mkdir(parents=True, exist_ok=True)

        for prompt_id, text in selected_prompts:
            done += 1
            wav_out = voice_out_dir / f"{prompt_id}.wav"
            print(f"[{done}/{total}] {variant_name} voice={voice_name} prompt={prompt_id}", flush=True)
            with torch.inference_mode():
                if args.generation_mode == "lux_like":
                    metrics = generate_sentence_lux_like(
                        save_path=str(wav_out),
                        prompt_text=prompt_text,
                        prompt_wav=str(prompt_wav),
                        text=text,
                        model=model,
                        vocoder=vocoder,
                        tokenizer=tokenizer,
                        feature_extractor=feature_extractor,
                        device=device,
                        num_step=args.num_steps,
                        guidance_scale=args.guidance_scale,
                        speed=args.speed,
                        lux_speed_multiplier=args.lux_speed_multiplier,
                        t_shift=args.t_shift,
                        target_rms=args.target_rms,
                        prompt_sampling_rate=24000,
                        output_sampling_rate=output_sampling_rate,
                    )
                else:
                    metrics = generate_sentence_for_benchmark(
                        save_path=str(wav_out),
                        prompt_text=prompt_text,
                        prompt_wav=str(prompt_wav),
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
                        prompt_sampling_rate=24000,
                        output_sampling_rate=output_sampling_rate,
                        max_duration=args.max_duration,
                        remove_long_sil=args.remove_long_sil,
                    )
            run_meta["results"].append(
                {
                    "voice": voice_name,
                    "prompt_id": prompt_id,
                    "text": text,
                    "wav_path": str(wav_out),
                    **metrics,
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
