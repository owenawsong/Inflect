"""
Paralinguistic training data generator.

Calls ElevenLabs API (or Fish-Speech locally) to generate audio for every
text in training_texts.py, across multiple voices.

Output structure:
    data/paralinguistic/
        rachel/
            0000_rachel.wav    + 0000_rachel.txt
            0001_rachel.wav    + 0001_rachel.txt
            ...
        josh/
            ...
        ...
    data/paralinguistic/manifest.csv   ← all (audio_path, text, voice, tags) rows

Usage:
    # ElevenLabs (uses API key from env or --key)
    python generate_paralinguistic_data.py --source elevenlabs --key YOUR_KEY

    # Dry run — shows what would be generated, no API calls
    python generate_paralinguistic_data.py --source elevenlabs --dry-run

    # Only specific voices
    python generate_paralinguistic_data.py --source elevenlabs --voices Rachel Josh Bella

    # Limit texts (for testing / credit budgeting)
    python generate_paralinguistic_data.py --source elevenlabs --limit 10

    # Check how many characters the full run would use
    python generate_paralinguistic_data.py --count-chars
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

BASE      = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE / "data" / "paralinguistic"
TEXTS_MOD = BASE / "data" / "training_texts.py"

# ── Load texts ────────────────────────────────────────────────────────────────

sys.path.insert(0, str(BASE / "data"))
from training_texts import TEXTS, ELEVENLABS_VOICES


def extract_tags(text: str) -> list[str]:
    """Pull all [tag] markers from a text."""
    return re.findall(r'\[([^\]]+)\]', text)


def strip_tags(text: str) -> str:
    """Remove [tags] from text for plain transcript."""
    return re.sub(r'\[[^\]]+\]', '', text).strip()


# ── ElevenLabs generation ─────────────────────────────────────────────────────

def generate_elevenlabs(texts: list[str], voices: list[str], api_key: str,
                        dry_run: bool = False, out_dir: Path = DATA_DIR) -> int:
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import save as el_save
    except ImportError:
        print("elevenlabs not installed. Run: pip install elevenlabs")
        sys.exit(1)

    client  = ElevenLabs(api_key=api_key)
    model   = "eleven_multilingual_v2"   # v3 if your tier supports it
    total   = 0
    skipped = 0

    # Build manifest rows
    manifest_path = out_dir / "manifest.csv"
    existing = set()
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(row["audio_path"])

    manifest_file = open(manifest_path, "a", newline="", encoding="utf-8")
    writer        = csv.DictWriter(manifest_file,
                                   fieldnames=["audio_path", "text", "plain_text",
                                               "voice", "tags"])
    if manifest_path.stat().st_size == 0:
        writer.writeheader()

    for voice_name in voices:
        v_dir = out_dir / voice_name.lower()
        v_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(texts):
            wav_path = v_dir / f"{i:04d}_{voice_name.lower()}.wav"
            txt_path = v_dir / f"{i:04d}_{voice_name.lower()}.txt"

            if str(wav_path) in existing:
                skipped += 1
                continue

            tags = extract_tags(text)
            plain = strip_tags(text)

            if dry_run:
                char_count = len(text)
                print(f"  [DRY RUN] {voice_name} / {i:04d}  ({char_count} chars)  tags={tags}")
                print(f"    \"{text[:80]}\"")
                total += char_count
                continue

            print(f"  Generating: {voice_name} / {i:04d}  tags={tags}")
            try:
                audio_gen = client.generate(
                    text=text,
                    voice=voice_name,
                    model=model,
                )
                el_save(audio_gen, str(wav_path))
                txt_path.write_text(text, encoding="utf-8")

                writer.writerow({
                    "audio_path": str(wav_path),
                    "text":       text,
                    "plain_text": plain,
                    "voice":      voice_name,
                    "tags":       "|".join(tags),
                })
                manifest_file.flush()
                total += 1

                # Polite rate limiting
                time.sleep(0.5)

            except Exception as e:
                print(f"    [ERROR] {e}")
                time.sleep(2)
                continue

    manifest_file.close()
    if dry_run:
        print(f"\nTotal chars: {total:,}  (~{total/1000:.1f}K ElevenLabs credits)")
    else:
        print(f"\nGenerated {total} clips, skipped {skipped} existing.")
    return total


# ── Character count helper ────────────────────────────────────────────────────

def count_chars(texts: list[str], voices: list[str]) -> None:
    total_chars = sum(len(t) for t in texts)
    print(f"\nTexts:        {len(texts)}")
    print(f"Voices:       {len(voices)}")
    print(f"Total clips:  {len(texts) * len(voices)}")
    print(f"Chars/text:   {total_chars / len(texts):.0f} avg")
    print(f"Total chars:  {total_chars * len(voices):,}")
    print(f"\nElevenLabs free tier = 10,000 chars/month")
    print(f"Full run needs:       {total_chars * len(voices):,} chars")
    print(f"  ({(total_chars * len(voices)) / 10000:.1f}x free tier)")
    print(f"\nWith 1 voice only:    {total_chars:,} chars  ({total_chars/10000:.1f}x free)")
    print(f"\nTo fit in 10K credits, use --limit {10000 // (total_chars // len(texts))} texts")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      choices=["elevenlabs", "fish"], default="elevenlabs")
    parser.add_argument("--key",         type=str, default=os.environ.get("ELEVENLABS_API_KEY", ""))
    parser.add_argument("--voices",      nargs="+", default=None,
                        help="Subset of voices to use (default: all in ELEVENLABS_VOICES)")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Only generate first N texts")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print what would be generated, don't call API")
    parser.add_argument("--count-chars", action="store_true",
                        help="Show character/credit budget and exit")
    parser.add_argument("--out",         type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    texts  = TEXTS[:args.limit] if args.limit else TEXTS
    voices = args.voices if args.voices else ELEVENLABS_VOICES
    out    = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.count_chars:
        count_chars(texts, voices)
        return

    print(f"Source:  {args.source}")
    print(f"Texts:   {len(texts)}")
    print(f"Voices:  {voices}")
    print(f"Out dir: {out}")
    if args.dry_run:
        print("Mode:    DRY RUN (no API calls)\n")

    if args.source == "elevenlabs":
        if not args.key and not args.dry_run:
            print("ERROR: --key required for ElevenLabs, or set ELEVENLABS_API_KEY env var")
            sys.exit(1)
        generate_elevenlabs(texts, voices, args.key,
                            dry_run=args.dry_run, out_dir=out)

    elif args.source == "fish":
        print("Fish-Speech support coming — need to install fish-speech first.")
        print("For now, use --source elevenlabs.")


if __name__ == "__main__":
    main()
