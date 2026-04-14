#!/usr/bin/env python3
"""
Build a generation plan for a multi-speaker VoxCPM2 cloning dataset.

The plan deliberately reuses each text across different voices while avoiding
repeated voice+text pairs. For a 20k text corpus and 60k target clips, this
produces ~3 voices per text on average.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "outputs" / "corpora" / "voxcpm_texts_20000_v2.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "corpora" / "voxcpm_generation_plan_v2_60k.csv"
DEFAULT_VOICES = PROJECT_ROOT / "reference_voices"
FALLBACK_VOICES = PROJECT_ROOT / "voice-encoder" / "data" / "paralinguistic"


class VoicePool:
    def __init__(self, paths: list[Path]):
        self.voices: dict[str, dict] = {}
        for base in paths:
            if not base.exists():
                continue
            for voice_dir in sorted(base.iterdir()):
                if not voice_dir.is_dir():
                    continue
                wavs = sorted(voice_dir.glob("*.wav")) + sorted(voice_dir.glob("*.mp3"))
                if not wavs:
                    continue
                meta_path = voice_dir / "meta.json"
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                transcripts = {}
                for wav in wavs:
                    txt = wav.with_suffix(".txt")
                    if txt.exists():
                        transcripts[wav.name] = txt.read_text(encoding="utf-8").strip()
                self.voices[voice_dir.name] = {
                    "wavs": wavs,
                    "meta": meta,
                    "transcripts": transcripts,
                }

    def names(self) -> list[str]:
        return sorted(self.voices)


def load_corpus(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            category = (row.get("category") or "misc").strip() or "misc"
            if text:
                rows.append((text, category))
    if not rows:
        raise SystemExit(f"Corpus is empty: {path}")
    return rows


def compute_repeat_plan(n_texts: int, total_clips: int, rng: random.Random) -> list[int]:
    base = total_clips // n_texts
    remainder = total_clips % n_texts
    repeats = [base] * n_texts
    if remainder:
        picks = list(range(n_texts))
        rng.shuffle(picks)
        for idx in picks[:remainder]:
            repeats[idx] += 1
    return repeats


def pick_distinct_voices(
    *,
    voice_names: list[str],
    global_counts: Counter,
    k: int,
    rng: random.Random,
) -> list[str]:
    chosen: list[str] = []
    available = set(voice_names)
    for _ in range(k):
        ranked = sorted((global_counts[name], rng.random(), name) for name in available)
        choice = ranked[0][2]
        chosen.append(choice)
        available.remove(choice)
        global_counts[choice] += 1
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-file", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--target-clips", type=int, default=60000)
    ap.add_argument("--voices", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = load_corpus(args.corpus_file)

    voice_paths = [args.voices] if args.voices else [p for p in [DEFAULT_VOICES, FALLBACK_VOICES] if p.exists()]
    pool = VoicePool(voice_paths)
    voice_names = pool.names()
    if not voice_names:
        raise SystemExit("No voices found for plan generation.")

    if args.target_clips < len(corpus):
        raise SystemExit("target-clips must be >= number of texts.")

    repeats = compute_repeat_plan(len(corpus), args.target_clips, rng)
    global_counts: Counter = Counter()
    planned_rows: list[dict[str, str]] = []

    text_indices = list(range(len(corpus)))
    rng.shuffle(text_indices)

    for idx in text_indices:
        text, category = corpus[idx]
        repeat_count = repeats[idx]
        if repeat_count > len(voice_names):
            raise SystemExit(
                f"Text requires {repeat_count} distinct voices but only {len(voice_names)} voices are available."
            )
        chosen_voices = pick_distinct_voices(
            voice_names=voice_names,
            global_counts=global_counts,
            k=repeat_count,
            rng=rng,
        )
        for rep_idx, voice_name in enumerate(chosen_voices, start=1):
            entry = pool.voices[voice_name]
            wav = rng.choice(entry["wavs"])
            prompt_text = entry["transcripts"].get(wav.name, "")
            plan_id = hashlib.md5(f"{text}|{category}|{voice_name}|{rep_idx}".encode("utf-8")).hexdigest()[:16]
            planned_rows.append(
                {
                    "plan_id": plan_id,
                    "text": text,
                    "category": category,
                    "voice_id": voice_name,
                    "reference_wav": str(wav.resolve()),
                    "prompt_text": prompt_text,
                    "repeat_index": str(rep_idx),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["plan_id", "text", "category", "voice_id", "reference_wav", "prompt_text", "repeat_index"],
        )
        writer.writeheader()
        writer.writerows(planned_rows)

    print(f"Wrote {len(planned_rows)} planned clips to {args.out}")
    print(f"Corpus texts: {len(corpus)}")
    print(f"Voices: {len(voice_names)}")
    print(f"Average clips/text: {len(planned_rows) / len(corpus):.2f}")
    print("Per-voice clip counts:")
    for voice_name in sorted(voice_names):
        print(f"  {voice_name:20s} {global_counts[voice_name]:4d}")


if __name__ == "__main__":
    main()
