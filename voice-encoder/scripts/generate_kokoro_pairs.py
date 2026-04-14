"""
Inflect Phase 1C — Generate Kokoro Alignment Pairs

For each of the 28 Kokoro voice packs, synthesize CLIPS_PER_VOICE short
clips using phonetically diverse text. Saves mels + manifest.

These become perfectly-labelled (mel, voice_pack) pairs for train_1c_alignment.py.

Runtime: ~10-15 min on RTX 3060.
Usage:
    python generate_kokoro_pairs.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE      = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
VOICE_DIR = BASE / "data" / "kokoro_voices"
OUT_DIR   = BASE / "data" / "1c_pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR       = 24_000
N_MELS          = 80
CLIPS_PER_VOICE = 50   # 28 voices × 50 = 1400 pairs

# 50 phonetically diverse sentences — varied rhythm, phoneme coverage, length
SENTENCES = [
    "The sun rises over the eastern hills and fills the valley with golden light.",
    "She carefully placed the old photograph back inside the wooden box.",
    "Every morning I wake up and drink a cup of strong black coffee.",
    "The children laughed and ran through the sprinklers in the backyard.",
    "He opened the door slowly and peered into the darkened hallway.",
    "The train departed at exactly seven fifteen in the evening.",
    "She had been working on the same painting for three weeks now.",
    "The dog barked loudly when the mail carrier arrived at the door.",
    "We drove for six hours through the mountains before finding a hotel.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Thunder rolled across the sky as the first drops of rain began to fall.",
    "He spent the afternoon reading a novel in the library.",
    "The city lights reflected beautifully on the surface of the river.",
    "She smiled when she saw the flowers on the kitchen table.",
    "The meeting was scheduled for three o'clock on Thursday afternoon.",
    "A cold wind blew through the empty streets of the town.",
    "He typed the last line of code and leaned back in his chair.",
    "The cat stretched out lazily in a patch of afternoon sunlight.",
    "They walked along the beach as the waves crashed against the shore.",
    "The professor wrote several equations on the chalkboard.",
    "She put on her coat and grabbed her umbrella before heading outside.",
    "The old clock on the wall ticked steadily through the quiet night.",
    "He called his mother every Sunday evening without fail.",
    "The bakery on the corner opened at six in the morning.",
    "Stars were visible through the gap in the clouds.",
    "She typed a quick reply and closed her laptop.",
    "The highway stretched endlessly across the flat desert landscape.",
    "He picked up the guitar and played a few quiet chords.",
    "The package arrived two days earlier than expected.",
    "A small bird landed on the windowsill and looked inside.",
    "The market was crowded with people buying fresh vegetables.",
    "She finished the report just before the deadline.",
    "The river flowed gently through the heart of the old city.",
    "He adjusted his tie in the mirror before leaving for work.",
    "The librarian handed her a stack of books on ancient history.",
    "Rain tapped lightly against the window throughout the night.",
    "The team celebrated after winning the championship game.",
    "She poured herself a glass of water and sat down at the table.",
    "The garden looked beautiful in the early morning mist.",
    "He folded the letter carefully and slid it into an envelope.",
    "The flight was delayed by two hours due to bad weather.",
    "She hummed a soft tune while washing the dishes.",
    "The mountains were dusted with fresh snow overnight.",
    "He read the newspaper while waiting for his coffee to cool.",
    "The classroom was quiet as students worked on their exams.",
    "She brought homemade cookies to share with her colleagues.",
    "The engine sputtered once and then fell silent.",
    "He noticed a small crack in the ceiling above his bed.",
    "The street was lined with tall oak trees that turned orange in autumn.",
    "She took a deep breath and stepped up to the microphone.",
]

assert len(SENTENCES) == CLIPS_PER_VOICE, "Need exactly CLIPS_PER_VOICE sentences"

_mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=1024, hop_length=256,
    win_length=1024, n_mels=N_MELS, f_max=8000.0,
)


def audio_to_mel(audio: np.ndarray) -> torch.Tensor:
    wav = torch.from_numpy(audio.astype("float32")).unsqueeze(0)
    wav = wav[:, :TARGET_SR * 10]
    mel = _mel_transform(wav)
    log_mel = torch.log(mel + 1e-5).squeeze(0)  # [80, T]
    T_ = log_mel.shape[1]
    if T_ >= 160:
        log_mel = log_mel[:, :160]
    else:
        log_mel = F.pad(log_mel, (0, 160 - T_))
    return log_mel


def main():
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a")

    voice_files = sorted(VOICE_DIR.glob("**/*.pt"))
    print(f"Found {len(voice_files)} voice packs")
    print(f"Generating {CLIPS_PER_VOICE} clips/voice = {len(voice_files) * CLIPS_PER_VOICE} total pairs")

    records = []
    skipped = 0

    for vf in tqdm(voice_files, desc="Voices"):
        voice_name = vf.stem
        pack = torch.load(vf, map_location="cpu", weights_only=True)  # [511,1,256]
        pipeline.voices[voice_name] = pack

        voice_out = OUT_DIR / voice_name
        voice_out.mkdir(exist_ok=True)

        for i, sentence in enumerate(SENTENCES):
            mel_path = voice_out / f"clip_{i:03d}.pt"

            # Skip already-generated clips (resume-friendly)
            if mel_path.exists():
                records.append({
                    "mel_path":   str(mel_path),
                    "voice_name": voice_name,
                    "voice_file": str(vf),
                    "clip_idx":   i,
                })
                continue

            chunks = []
            try:
                for chunk in pipeline(sentence, voice=voice_name, speed=1.0):
                    if chunk.audio is not None:
                        chunks.append(chunk.audio)
            except Exception as e:
                tqdm.write(f"  [{voice_name}] clip {i} error: {e}")
                skipped += 1
                continue

            if not chunks:
                skipped += 1
                continue

            audio = np.concatenate(chunks)
            mel   = audio_to_mel(audio)
            torch.save(mel, mel_path)

            records.append({
                "mel_path":   str(mel_path),
                "voice_name": voice_name,
                "voice_file": str(vf),
                "clip_idx":   i,
            })

    manifest_path = BASE / "data" / "1c_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    print(f"\nDone.")
    print(f"  Generated: {len(records)} clips")
    print(f"  Skipped:   {skipped}")
    print(f"  Manifest:  {manifest_path}")
    print(f"\nNext: python train_1c_alignment.py")


if __name__ == "__main__":
    main()
