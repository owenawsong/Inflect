"""
ZipVoice API server for TTS comparison.
Run with: .venv-voxcpm\Scripts\python.exe zipvoice_server.py
Serves at http://localhost:18084
"""

import base64
import io
import json
import random
import sys
import time
import types
from pathlib import Path

import soundfile as sf
import torch
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import scipy.signal

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ZipVoice-official"))

REFERENCE_VOICES = PROJECT_ROOT / "reference_voices"
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--k2-fsa--ZipVoice" / "snapshots"

app = FastAPI(title="ZipVoice TTS Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Globals (loaded at startup) ───────────────────────────────────────────────
MODEL = None
TOKENIZER = None
FEATURE_EXTRACTOR = None
VOCODER = None
VOICE_POOL = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS = 8
GUIDANCE_SCALE = 3.0


def _load_voice_pool():
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


def _find_cached_model_dir(model_name: str):
    if not HF_CACHE_ROOT.exists():
        return None
    for snapshot_dir in sorted(HF_CACHE_ROOT.iterdir(), reverse=True):
        candidate = snapshot_dir / model_name
        if all((candidate / n).exists() for n in ("model.json", "model.pt", "tokens.txt")):
            return candidate
    return None


def _load_audio_sf(path, target_sr=24000):
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = scipy.signal.resample_poly(data, target_sr, sr).astype(np.float32)
    return torch.from_numpy(data).unsqueeze(0), target_sr


@app.on_event("startup")
def load_model():
    global MODEL, TOKENIZER, FEATURE_EXTRACTOR, VOCODER, VOICE_POOL

    print(f"ZipVoice server starting on {DEVICE}")

    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.checkpoint import load_checkpoint
    from zipvoice.utils.feature import VocosFbank
    from vocos import Vocos

    model_name = "zipvoice_distill"
    cached = _find_cached_model_dir(model_name)
    if cached:
        model_json = cached / "model.json"
        model_pt = cached / "model.pt"
        tokens_txt = cached / "tokens.txt"
    else:
        from huggingface_hub import hf_hub_download
        print("Downloading ZipVoice-Distill...")
        model_json = hf_hub_download("k2-fsa/ZipVoice", f"{model_name}/model.json")
        model_pt = hf_hub_download("k2-fsa/ZipVoice", f"{model_name}/model.pt")
        tokens_txt = hf_hub_download("k2-fsa/ZipVoice", f"{model_name}/tokens.txt")

    with open(model_json) as f:
        model_config = json.load(f)

    TOKENIZER = EmiliaTokenizer(token_file=str(tokens_txt))
    tokenizer_cfg = {"vocab_size": TOKENIZER.vocab_size, "pad_id": TOKENIZER.pad_id}

    MODEL = ZipVoiceDistill(**model_config["model"], **tokenizer_cfg)
    load_checkpoint(str(model_pt), MODEL)
    MODEL = MODEL.to(DEVICE).eval()

    FEATURE_EXTRACTOR = VocosFbank()
    VOCODER = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE).eval()

    VOICE_POOL = _load_voice_pool()
    print(f"ZipVoice ready. Voice pool: {len(VOICE_POOL)} voices. Device: {DEVICE}")


@app.post("/generate")
async def generate(
    text: str = Form(...),
    prompt_audio: UploadFile | None = File(None),
):
    from zipvoice.bin.infer_zipvoice import generate_sentence

    t0 = time.time()

    # Load reference voice
    if prompt_audio is not None:
        raw = await prompt_audio.read()
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        import scipy.signal as ss
        if sr != 24000:
            data = ss.resample_poly(data, 24000, sr).astype(np.float32)
        ref_wav = torch.from_numpy(data).unsqueeze(0)
        ref_transcript = ""
    elif VOICE_POOL:
        voice_path, ref_transcript = random.choice(VOICE_POOL)
        ref_wav, _ = _load_audio_sf(voice_path, target_sr=24000)
    else:
        return JSONResponse(status_code=400, content={"error": "No reference voice available."})

    ref_wav = ref_wav.to(DEVICE)
    ref_feat = FEATURE_EXTRACTOR.extract(ref_wav, 24000)

    with torch.no_grad():
        wav = generate_sentence(
            model=MODEL,
            tokenizer=TOKENIZER,
            feature_extractor=FEATURE_EXTRACTOR,
            vocoder=VOCODER,
            text=text,
            ref_wav=ref_wav,
            ref_feat=ref_feat,
            device=DEVICE,
            num_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            speed=1.0,
            t_shift=0.5,
            target_rms=0.1,
        )

    elapsed = time.time() - t0

    # wav is (1, T) or (T,)
    wav_np = wav.cpu().numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np[0]

    buf = io.BytesIO()
    sf.write(buf, wav_np, 24000, format="WAV", subtype="PCM_16")
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    duration = len(wav_np) / 24000
    return {
        "audio_base64": audio_b64,
        "sample_rate": 24000,
        "elapsed_seconds": round(elapsed, 2),
        "duration_seconds": round(duration, 2),
        "rtf": round(elapsed / max(duration, 0.001), 2),
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": "zipvoice_distill"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=18084, log_level="info")
