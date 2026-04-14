"""
Quick test: encode a dataset clip, find the nearest Kokoro voice pack
by cosine similarity, and generate with that real voice pack.

This proves the encoder is learning meaningful speaker features —
the "nearest voice" should sound somewhat like the input clip's speaker.
It won't be a perfect clone, but it should pick male/female correctly,
match accent/tone roughly, etc.

Usage:
    python test_nearest_voice.py
"""

import sys
import warnings
import random
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE      = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
CKPT_DIR  = BASE / "checkpoints"
VOICE_DIR = BASE / "data" / "kokoro_voices"
OUT_DIR   = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT = (
    "Hello! This voice was selected by the Inflect encoder "
    "as the closest match to the reference speaker."
)

def main():
    # Load encoder
    from train_ve import VoiceEncoder
    encoder = VoiceEncoder().to(DEVICE)
    ckpt = torch.load(CKPT_DIR / "ve_latest.pt", map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()

    # Load all voice pack centroids
    voice_files = list(VOICE_DIR.glob("**/*.pt"))
    voice_names, voice_vecs, voice_packs = [], [], []
    for vf in sorted(voice_files):
        pack = torch.load(vf, map_location="cpu", weights_only=True)  # [511, 1, 256]
        centroid = F.normalize(pack[:, 0, :].mean(dim=0), dim=0)      # [256]
        voice_names.append(vf.stem)
        voice_vecs.append(centroid)
        voice_packs.append(pack)

    voice_mat = torch.stack(voice_vecs).to(DEVICE)  # [N, 256]
    print(f"Loaded {len(voice_names)} voice packs")

    # Pick 5 random dataset clips and test each
    df = pd.read_csv(BASE / "data" / "manifest.csv")
    df = df[df["duration"] >= 4.0]
    samples = df.sample(5)

    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a")

    for _, row in samples.iterrows():
        spk = row["speaker_id"]
        try:
            mel = torch.load(row["mel_path"], weights_only=True)  # [80, T]
        except Exception:
            continue

        # Crop to 160 frames
        T = mel.shape[1]
        start = random.randint(0, max(0, T - 160))
        mel_chunk = mel[:, start:start+160] if T > 160 else F.pad(mel, (0, 160-T))
        batch = mel_chunk.unsqueeze(0).to(DEVICE)  # [1, 80, 160]

        with torch.no_grad():
            enc_emb = encoder.embed(batch)           # [1, 256]
            sims = torch.mm(enc_emb, voice_mat.T)    # [1, N]
            top3_idx = sims[0].topk(3).indices.tolist()

        top3_voices = [voice_names[i] for i in top3_idx]
        top3_sims   = [sims[0][i].item() for i in top3_idx]
        best_pack   = voice_packs[top3_idx[0]]       # [511, 1, 256]

        print(f"\nSpeaker: {spk}")
        print(f"  Top matches: {top3_voices[0]} ({top3_sims[0]:.3f}), "
              f"{top3_voices[1]} ({top3_sims[1]:.3f}), "
              f"{top3_voices[2]} ({top3_sims[2]:.3f})")

        # Generate with the nearest real voice pack
        VOICE_KEY = f"_nearest_{spk}"
        pipeline.voices[VOICE_KEY] = best_pack

        audio_chunks = []
        for chunk in pipeline(TEXT, voice=VOICE_KEY, speed=1.0):
            if chunk.audio is not None:
                audio_chunks.append(chunk.audio)

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
            out_path = OUT_DIR / f"nearest_{spk}_{top3_voices[0]}.wav"
            sf.write(str(out_path), audio, 24000)
            print(f"  Saved: {out_path.name}  ({len(audio)/24000:.1f}s)")

    print("\nDone. Check outputs/ folder.")
    print("Listen: does the voice roughly match the speaker type (male/female, accent)?")
    print("If yes -> encoder works, just need better adapter training for Phase 1C.")

if __name__ == "__main__":
    main()
