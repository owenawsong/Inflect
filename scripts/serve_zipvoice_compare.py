r"""
Serve a small local site for comparing ZipVoice benchmark outputs.

Usage:
  .\.venv-voxcpm\Scripts\python.exe scripts\serve_zipvoice_compare.py
  .\.venv-voxcpm\Scripts\python.exe scripts\serve_zipvoice_compare.py --port 8810
"""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = PROJECT_ROOT / "outputs" / "zipvoice_bench"

DISPLAY_NAMES = {
    "zipvoice": "Base ZipVoice (16-step)",
    "zipvoice_distill_8step_cfg3p0": "Distill 8-step",
    "zipvoice_distill_4step_cfg3p0": "Distill 4-step",
    "zipvoice_distill_4step_cfg3p0_lux48k": "Distill 4-step + Lux 48k vocoder",
    "zipvoice_distill_4step_cfg3p0_lux_anchor": "Distill 4-step + Lux solver",
    "zipvoice_distill_4step_cfg3p0_t0p7": "Distill 4-step + t_shift 0.7",
    "zipvoice_distill_4step_cfg3p0_t0p9": "Distill 4-step + t_shift 0.9",
    "zipvoice_distill_4step_cfg3p0_t1p0": "Distill 4-step + t_shift 1.0",
    "zipvoice_distill_4step_cfg3p0_pcap3p0": "Distill 4-step + prompt cap 3s",
    "zipvoice_distill_4step_cfg3p0_pcap5p0": "Distill 4-step + prompt cap 5s",
    "zipvoice_distill_4step_cfg3p0_pcap7p0": "Distill 4-step + prompt cap 7s",
    "inflect_trackA_lux_stack_no48k": "Track A bundle (Lux-like, stock 24k)",
    "inflect_trackA_lux_stack": "Track A bundle (Lux-like + 48k)",
    "inflect_trackA_lux_stack_safe_no48k": "Track A safe bundle (stock 24k)",
    "inflect_trackA_lux_stack_safe": "Track A safe bundle (+ Lux 48k)",
    "inflect_trackA_tuned_no48k": "Track A tuned (stock 24k)",
    "inflect_trackA_tuned": "Track A tuned (+ Lux 48k)",
    "inflect_base": "Inflect base",
    "inflect_base_solver": "Inflect base + Lux solver",
    "lux_tts_real": "LuxTTS (real, original speed)",
    "lux_tts_real_normal_speed": "LuxTTS (real, normal speed)",
}

VARIANT_ORDER = [
    "zipvoice",
    "zipvoice_distill_8step_cfg3p0",
    "zipvoice_distill_4step_cfg3p0",
    "zipvoice_distill_4step_cfg3p0_lux48k",
    "zipvoice_distill_4step_cfg3p0_lux_anchor",
    "zipvoice_distill_4step_cfg3p0_t0p7",
    "zipvoice_distill_4step_cfg3p0_t0p9",
    "zipvoice_distill_4step_cfg3p0_t1p0",
    "zipvoice_distill_4step_cfg3p0_pcap3p0",
    "zipvoice_distill_4step_cfg3p0_pcap5p0",
    "zipvoice_distill_4step_cfg3p0_pcap7p0",
    "inflect_trackA_lux_stack_safe_no48k",
    "inflect_trackA_lux_stack_safe",
    "inflect_trackA_tuned_no48k",
    "inflect_trackA_tuned",
    "inflect_base",
    "inflect_base_solver",
    "lux_tts_real_normal_speed",
    "lux_tts_real",
    "inflect_trackA_lux_stack_no48k",
    "inflect_trackA_lux_stack",
]


def build_manifest(allowed_variants=None):
    variants = {}
    prompt_texts = {}
    voices = set()
    prompt_ids = set()

    if not BENCH_ROOT.exists():
        return {"variants": {}, "voices": [], "prompt_ids": [], "prompt_texts": {}}

    allowed = set(allowed_variants) if allowed_variants else None
    for variant_dir in sorted(p for p in BENCH_ROOT.iterdir() if p.is_dir()):
        if allowed is not None and variant_dir.name not in allowed:
            continue
        metadata_path = variant_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                for item in metadata.get("prompts", []):
                    prompt_texts[item["id"]] = item["text"]
            except Exception:
                pass

        variant_entry = {}
        for voice_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
            voice_map = {}
            for wav in sorted(voice_dir.glob("*.wav")):
                voice_map[wav.stem] = f"/files/{wav.relative_to(PROJECT_ROOT).as_posix()}"
                prompt_ids.add(wav.stem)
            if voice_map:
                variant_entry[voice_dir.name] = voice_map
                voices.add(voice_dir.name)
        if variant_entry:
            variants[variant_dir.name] = variant_entry

    variant_order = [v for v in VARIANT_ORDER if v in variants] + [v for v in sorted(variants) if v not in VARIANT_ORDER]

    if variant_order:
        pair_sets = []
        for variant in variant_order:
            pairs = set()
            for voice_name, voice_map in variants[variant].items():
                for prompt_id in voice_map:
                    pairs.add((voice_name, prompt_id))
            pair_sets.append(pairs)
        valid_pairs = set.intersection(*pair_sets) if pair_sets else set()
    else:
        valid_pairs = set()

    valid_voice_to_prompts = {}
    for voice_name, prompt_id in sorted(valid_pairs):
        valid_voice_to_prompts.setdefault(voice_name, []).append(prompt_id)

    filtered_variants = {}
    for variant in variant_order:
        filtered_voice_map = {}
        for voice_name, voice_map in variants[variant].items():
            filtered_prompt_map = {
                prompt_id: file_url
                for prompt_id, file_url in voice_map.items()
                if (voice_name, prompt_id) in valid_pairs
            }
            if filtered_prompt_map:
                filtered_voice_map[voice_name] = filtered_prompt_map
        if filtered_voice_map:
            filtered_variants[variant] = filtered_voice_map

    return {
        "variants": filtered_variants,
        "display_names": DISPLAY_NAMES,
        "variant_order": [v for v in variant_order if v in filtered_variants],
        "voices": sorted(valid_voice_to_prompts),
        "prompt_ids": sorted({prompt_id for _, prompt_id in valid_pairs}),
        "voice_to_prompts": valid_voice_to_prompts,
        "prompt_texts": prompt_texts,
    }


def render_html(manifest):
    data = json.dumps(manifest)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZipVoice Compare</title>
  <style>
    :root {{
      --bg: #0f1417;
      --panel: #161d21;
      --panel-2: #1b252b;
      --text: #eef3f5;
      --muted: #9db0b8;
      --line: #28353d;
      --accent: #a8ffcf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(132, 224, 172, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(110, 168, 254, 0.12), transparent 24%),
        linear-gradient(180deg, #0d1114, var(--bg));
      color: var(--text);
      font: 14px/1.45 Consolas, "SFMono-Regular", Menlo, monospace;
      min-height: 100vh;
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 20px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .control, .prompt-box, .card {{
      background: linear-gradient(180deg, var(--panel), var(--panel-2));
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .control label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    select, button {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #33434d;
      background: #0f1519;
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
      background: #122018;
      border-color: #294c35;
    }}
    .row-buttons {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .prompt-box {{
      margin-bottom: 16px;
    }}
    .prompt-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .prompt-id {{
      color: var(--accent);
      margin-bottom: 6px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .card h2 {{
      margin: 0 0 6px;
      font-size: 16px;
    }}
    .path {{
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
      margin-bottom: 12px;
      min-height: 34px;
    }}
    audio {{
      width: 100%;
    }}
    .missing {{
      color: #ffb4b4;
      font-style: italic;
      margin-top: 6px;
    }}
    @media (max-width: 900px) {{
      .controls {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 640px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>ZipVoice Compare</h1>
    <div class="sub">Compare benchmark outputs across official and experimental variants from one page.</div>

    <div class="controls">
      <div class="control">
        <label for="voice">Voice</label>
        <select id="voice"></select>
      </div>
      <div class="control">
        <label for="prompt">Prompt</label>
        <select id="prompt"></select>
      </div>
      <div class="control">
        <label>Prompt Navigation</label>
        <div class="row-buttons">
          <button id="prevPrompt">Previous</button>
          <button id="nextPrompt">Next</button>
        </div>
      </div>
      <div class="control">
        <label>Voice Navigation</label>
        <div class="row-buttons">
          <button id="prevVoice">Previous</button>
          <button id="nextVoice">Next</button>
        </div>
      </div>
    </div>

    <div class="prompt-box">
      <div class="prompt-label">Prompt Text</div>
      <div class="prompt-id" id="promptId"></div>
      <div id="promptText"></div>
    </div>

    <div class="cards" id="cards"></div>
  </div>

  <script>
    const manifest = {data};
    const variantNames = manifest.variant_order || Object.keys(manifest.variants);
    const voices = manifest.voices;
    const allPrompts = manifest.prompt_ids;
    const voiceToPrompts = manifest.voice_to_prompts || {{}};

    const voiceSelect = document.getElementById('voice');
    const promptSelect = document.getElementById('prompt');
    const cards = document.getElementById('cards');
    const promptId = document.getElementById('promptId');
    const promptText = document.getElementById('promptText');

    function fillSelect(select, values) {{
      select.innerHTML = '';
      for (const value of values) {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }}
    }}

    fillSelect(voiceSelect, voices);

    function syncPromptOptions(preferredPrompt = null) {{
      const voice = voiceSelect.value;
      const promptOptions = voiceToPrompts[voice] || allPrompts;
      const nextPrompt = preferredPrompt && promptOptions.includes(preferredPrompt)
        ? preferredPrompt
        : (promptOptions.includes(promptSelect.value) ? promptSelect.value : promptOptions[0]);
      fillSelect(promptSelect, promptOptions);
      if (nextPrompt) {{
        promptSelect.value = nextPrompt;
      }}
    }}

    function updateCards() {{
      const voice = voiceSelect.value;
      const prompt = promptSelect.value;
      promptId.textContent = prompt || '';
      promptText.textContent = manifest.prompt_texts[prompt] || '(No prompt text found in metadata.)';
      cards.innerHTML = '';

      for (const variant of variantNames) {{
        const fileUrl = manifest.variants?.[variant]?.[voice]?.[prompt] || null;
        const card = document.createElement('div');
        card.className = 'card';

        const title = document.createElement('h2');
        title.textContent = manifest.display_names?.[variant] || variant;
        card.appendChild(title);

        const path = document.createElement('div');
        path.className = 'path';
        path.textContent = fileUrl ? fileUrl.replace('/files/', '') : 'No file for this variant / voice / prompt';
        card.appendChild(path);

        if (fileUrl) {{
          const audio = document.createElement('audio');
          audio.controls = true;
          audio.preload = 'none';
          audio.src = fileUrl;
          card.appendChild(audio);
        }} else {{
          const missing = document.createElement('div');
          missing.className = 'missing';
          missing.textContent = 'Missing';
          card.appendChild(missing);
        }}
        cards.appendChild(card);
      }}
    }}

    function moveSelect(select, delta) {{
      const values = [...select.options].map(o => o.value);
      const idx = values.indexOf(select.value);
      const nextIdx = (idx + delta + values.length) % values.length;
      select.value = values[nextIdx];
      if (select === voiceSelect) {{
        syncPromptOptions();
      }}
      updateCards();
    }}

    voiceSelect.addEventListener('change', () => {{
      syncPromptOptions();
      updateCards();
    }});
    promptSelect.addEventListener('change', updateCards);
    document.getElementById('prevPrompt').addEventListener('click', () => moveSelect(promptSelect, -1));
    document.getElementById('nextPrompt').addEventListener('click', () => moveSelect(promptSelect, 1));
    document.getElementById('prevVoice').addEventListener('click', () => moveSelect(voiceSelect, -1));
    document.getElementById('nextVoice').addEventListener('click', () => moveSelect(voiceSelect, 1));

    if (voices.length > 0) voiceSelect.value = voices[0];
    syncPromptOptions();
    updateCards();
  </script>
</body>
</html>"""


class CompareHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = render_html(build_manifest(self.server.allowed_variants)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/manifest":
            payload = json.dumps(build_manifest(self.server.allowed_variants)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed.path.startswith("/files/"):
            rel = unquote(parsed.path[len("/files/"):])
            file_path = (PROJECT_ROOT / rel).resolve()
            try:
                file_path.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                self.send_error(403, "Forbidden")
                return
            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, "Not found")
                return
            ctype, _ = mimetypes.guess_type(str(file_path))
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404, "Not found")

    def log_message(self, format, *args):
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--only-variants", default=None, help="Optional comma-separated variant names to show.")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), CompareHandler)
    server.allowed_variants = [x.strip() for x in args.only_variants.split(",")] if args.only_variants else None
    print(f"ZipVoice compare site: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
