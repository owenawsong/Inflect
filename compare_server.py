"""
TTS Comparison Server — ZipVoice vs MOSS-TTS-Nano
Serves the comparison UI + runs ZipVoice inline + proxies MOSS at :18083

Run with: .venv-voxcpm\Scripts\python.exe compare_server.py
Open:     http://localhost:18085
"""

import base64
import io
import json
import random
import sys
import time
from pathlib import Path

# ── Fix inflect shadowing: remove project root from sys.path BEFORE any imports
_HERE = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _HERE and p != ""]
sys.path.insert(0, str(_HERE / "ZipVoice-official"))

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

try:
    import httpx
    _HTTPX = True
except ImportError:
    import urllib.request, urllib.error
    _HTTPX = False

REFERENCE_VOICES = _HERE / "reference_voices"
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--k2-fsa--ZipVoice" / "snapshots"
MOSS_URL = "http://localhost:18083"

app = FastAPI(title="TTS Compare")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Globals ───────────────────────────────────────────────────────────────────
MODEL = TOKENIZER = FEATURE_EXTRACTOR = VOCODER = None
VOICE_POOL = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_voice_pool():
    pool = []
    if not REFERENCE_VOICES.exists():
        return pool
    for d in sorted(REFERENCE_VOICES.iterdir()):
        if not d.is_dir():
            continue
        for ext in ("*.wav", "*.mp3"):
            for wav in d.glob(ext):
                txt = wav.with_suffix(".txt")
                if txt.exists():
                    t = txt.read_text(encoding="utf-8").strip()
                    if t and t != "[NO_SPEECH]":
                        pool.append((wav, t))
    return pool


def _find_cached_model():
    if not HF_CACHE_ROOT.exists():
        return None
    for snap in sorted(HF_CACHE_ROOT.iterdir(), reverse=True):
        c = snap / "zipvoice_distill"
        if all((c / n).exists() for n in ("model.json", "model.pt", "tokens.txt")):
            return c
    return None


def _load_audio(path_or_bytes, target_sr=24000):
    if isinstance(path_or_bytes, (str, Path)):
        data, sr = sf.read(str(path_or_bytes), dtype="float32", always_2d=False)
    else:
        data, sr = sf.read(io.BytesIO(path_or_bytes), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = scipy.signal.resample_poly(data, target_sr, sr).astype(np.float32)
    return torch.from_numpy(data).unsqueeze(0)


@app.on_event("startup")
def _startup():
    global MODEL, TOKENIZER, FEATURE_EXTRACTOR, VOCODER, VOICE_POOL
    print(f"Loading ZipVoice-Distill on {DEVICE}...")

    from zipvoice.models.zipvoice_distill import ZipVoiceDistill
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
    from zipvoice.utils.checkpoint import load_checkpoint
    from zipvoice.utils.feature import VocosFbank
    from vocos import Vocos

    cached = _find_cached_model()
    if cached:
        mj, mp, mt = cached / "model.json", cached / "model.pt", cached / "tokens.txt"
    else:
        from huggingface_hub import hf_hub_download
        print("Downloading ZipVoice-Distill from HuggingFace...")
        mj = hf_hub_download("k2-fsa/ZipVoice", "zipvoice_distill/model.json")
        mp = hf_hub_download("k2-fsa/ZipVoice", "zipvoice_distill/model.pt")
        mt = hf_hub_download("k2-fsa/ZipVoice", "zipvoice_distill/tokens.txt")

    with open(mj) as f:
        cfg = json.load(f)

    TOKENIZER = EmiliaTokenizer(token_file=str(mt))
    MODEL = ZipVoiceDistill(**cfg["model"], vocab_size=TOKENIZER.vocab_size, pad_id=TOKENIZER.pad_id)
    load_checkpoint(str(mp), MODEL)
    MODEL.to(DEVICE).eval()

    FEATURE_EXTRACTOR = VocosFbank()
    VOCODER = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(DEVICE).eval()
    VOICE_POOL = _load_voice_pool()
    print(f"Ready. Device={DEVICE}, voices={len(VOICE_POOL)}")


# ── ZipVoice endpoint ─────────────────────────────────────────────────────────
@app.post("/generate/zipvoice")
async def gen_zipvoice(text: str = Form(...), prompt_audio: UploadFile | None = File(None)):
    import tempfile, os
    from zipvoice.bin.infer_zipvoice import generate_sentence

    t0 = time.time()

    # Pick reference voice
    if prompt_audio is not None:
        raw = await prompt_audio.read()
        # Save to temp wav
        tmp_prompt = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_data = _load_audio(raw)  # resample to 24kHz
        sf.write(tmp_prompt.name, audio_data.squeeze().numpy(), 24000, subtype="PCM_16")
        tmp_prompt.close()
        prompt_wav_path = tmp_prompt.name
        prompt_text = "Hello."
    elif VOICE_POOL:
        voice_path, prompt_text = random.choice(VOICE_POOL)
        # Resample to 24kHz into a temp file (handles RIFF-as-mp3 etc)
        audio_data = _load_audio(voice_path)
        tmp_prompt = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_prompt.name, audio_data.squeeze().numpy(), 24000, subtype="PCM_16")
        tmp_prompt.close()
        prompt_wav_path = tmp_prompt.name
    else:
        return JSONResponse(status_code=400, content={"error": "No reference voice."})

    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out.close()

    try:
        with torch.inference_mode():
            generate_sentence(
                save_path=tmp_out.name,
                prompt_text=prompt_text,
                prompt_wav=prompt_wav_path,
                text=text,
                model=MODEL,
                vocoder=VOCODER,
                tokenizer=TOKENIZER,
                feature_extractor=FEATURE_EXTRACTOR,
                device=DEVICE,
                num_step=8,
                guidance_scale=3.0,
                speed=1.0,
                t_shift=0.5,
                target_rms=0.1,
                feat_scale=0.1,
                sampling_rate=24000,
                max_duration=100,
            )

        elapsed = time.time() - t0
        wav_np, sr = sf.read(tmp_out.name, dtype="float32")
        duration = len(wav_np) / sr

        buf = io.BytesIO()
        sf.write(buf, wav_np, sr, format="WAV", subtype="PCM_16")
        return {
            "audio_base64": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": sr,
            "elapsed_seconds": round(elapsed, 2),
            "duration_seconds": round(duration, 2),
            "rtf": round(elapsed / max(duration, 0.001), 2),
        }
    finally:
        os.unlink(tmp_out.name)
        if prompt_audio is not None:
            os.unlink(prompt_wav_path)


# ── MOSS proxy endpoint ───────────────────────────────────────────────────────
@app.post("/generate/moss")
async def gen_moss(text: str = Form(...), prompt_audio: UploadFile | None = File(None)):
    import urllib.request, urllib.error, urllib.parse

    t0 = time.time()
    boundary = "----CompareFormBoundary"
    parts = []

    def field(name, value):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()

    parts.append(field("text", text))
    parts.append(field("enable_normalize_tts_text", "0"))

    if prompt_audio is not None:
        raw = await prompt_audio.read()
        fname = prompt_audio.filename or "voice.wav"
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"prompt_audio\"; filename=\"{fname}\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
            + raw + b"\r\n"
        )
    else:
        parts.append(field("demo_id", "demo-1"))

    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{MOSS_URL}/api/generate",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())
        return JSONResponse(status_code=e.code, content=err)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    elapsed = time.time() - t0
    data["elapsed_seconds"] = round(elapsed, 2)
    # Compute duration from base64 wav
    try:
        wav_bytes = base64.b64decode(data["audio_base64"])
        arr, sr = sf.read(io.BytesIO(wav_bytes))
        dur = len(arr) / sr
        data["duration_seconds"] = round(dur, 2)
        data["rtf"] = round(elapsed / max(dur, 0.001), 2)
    except Exception:
        pass
    return data


# ── Serve the comparison UI ───────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TTS Compare</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e0e0e0;padding:24px}
h1{font-size:20px;font-weight:600;margin-bottom:20px;color:#fff}
.card{background:#1a1a1a;border-radius:10px;padding:20px;margin-bottom:16px}
textarea{width:100%;height:72px;background:#111;color:#e0e0e0;border:1px solid #333;
  border-radius:6px;padding:10px;font-size:14px;resize:vertical}
label{font-size:12px;color:#777}
input[type=file]{font-size:13px;color:#aaa;background:#111;border:1px solid #333;
  border-radius:6px;padding:5px 10px;margin-top:4px;display:block}
button{background:#5b5ef7;color:#fff;border:none;border-radius:8px;
  padding:11px 28px;font-size:15px;font-weight:600;cursor:pointer;margin-top:14px}
button:hover{background:#4a4dd6}button:disabled{background:#333;color:#555;cursor:not-allowed}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
h2{font-size:15px;font-weight:600;margin-bottom:2px}
.sub{font-size:11px;color:#555;margin-bottom:12px}
.status{font-size:13px;color:#666;min-height:18px;margin-bottom:8px}
.running{color:#f0a500}.done{color:#4caf50}.error{color:#f44336}
audio{width:100%;margin-top:6px}
.stats{font-size:12px;color:#555;margin-top:6px}
.badge{display:inline-block;background:#222;border-radius:4px;padding:2px 7px;
  font-size:11px;color:#888;margin-right:4px}
.win{background:#1a2e1a;color:#4caf50}
</style>
</head>
<body>
<h1>TTS Comparison</h1>
<div class="card">
  <textarea id="txt">That's actually hilarious, I love it.</textarea>
  <div style="margin-top:10px">
    <label>Reference voice (optional — uses random from pool if empty)</label>
    <input type="file" id="voice" accept="audio/*">
  </div>
  <button id="btn" onclick="run()">Generate Both</button>
</div>
<div class="grid">
  <div class="card">
    <h2>ZipVoice-Distill</h2>
    <div class="sub">Flow matching · 8-step · same server</div>
    <div class="status" id="zs">Ready</div>
    <audio id="za" controls style="display:none"></audio>
    <div class="stats" id="zst"></div>
  </div>
  <div class="card">
    <h2>MOSS-TTS-Nano</h2>
    <div class="sub">100M · Autoregressive · proxied :18083</div>
    <div class="status" id="ms">Ready</div>
    <audio id="ma" controls style="display:none"></audio>
    <div class="stats" id="mst"></div>
  </div>
</div>
<script>
async function run(){
  const text=document.getElementById('txt').value.trim();
  if(!text)return;
  const f=document.getElementById('voice').files[0];
  document.getElementById('btn').disabled=true;
  ['z','m'].forEach(x=>{
    set(x,'running','Generating...');
    document.getElementById(x+'a').style.display='none';
    document.getElementById(x+'st').innerHTML='';
  });
  await Promise.all([call('z','/generate/zipvoice',text,f), call('m','/generate/moss',text,f)]);
  document.getElementById('btn').disabled=false;
  winner();
}
async function call(id,url,text,file){
  const fd=new FormData();
  fd.append('text',text);
  if(file)fd.append('prompt_audio',file);
  try{
    const r=await fetch(url,{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok){set(id,'error','Error: '+(d.error||r.status));return;}
    const el=document.getElementById(id+'a');
    el.src='data:audio/wav;base64,'+d.audio_base64;
    el.style.display='block';
    el._wall=d.elapsed_seconds;
    el._dur=d.duration_seconds;
    set(id,'done','Done');
    document.getElementById(id+'st').innerHTML=
      badge('Wall: '+d.elapsed_seconds+'s')+
      (d.duration_seconds?badge('Audio: '+d.duration_seconds+'s'):'')+
      (d.rtf?badge('RTF: '+d.rtf+'x'):'');
  }catch(e){set(id,'error','Error: '+e.message);}
}
function set(id,cls,msg){const e=document.getElementById(id+'s');e.className='status '+cls;e.textContent=msg;}
function badge(t,cls=''){return`<span class="badge ${cls}">${t}</span>`;}
function winner(){
  const z=document.getElementById('za'),m=document.getElementById('ma');
  if(!z._wall||!m._wall)return;
  const faster=z._wall<m._wall?'z':'m';
  document.getElementById(faster+'st').innerHTML+=badge('⚡ Faster','win');
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=18085, log_level="warning")
