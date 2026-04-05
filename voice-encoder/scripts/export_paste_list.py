"""
Exports training texts as a numbered paste list.

Open the output .txt file, follow the numbers, paste each into ElevenLabs,
generate, download. That's it.

Usage:
    python export_paste_list.py                    # all 105 texts
    python export_paste_list.py --limit 50         # first 50 (fits in 10K free credits)
    python export_paste_list.py --tags laughs sighs whispers   # only texts with these tags
    python export_paste_list.py --out my_list.txt
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from training_texts import TEXTS

BASE = Path(__file__).resolve().parent.parent


def has_tag(text: str, tags: list[str]) -> bool:
    found = [t.lower() for t in re.findall(r'\[([^\]]+)\]', text)]
    return any(t in found for t in tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=None, help="Only first N texts")
    parser.add_argument("--tags",   nargs="+", default=None,
                        help="Only texts containing these tags, e.g. --tags laughs sighs")
    parser.add_argument("--out",    type=str, default=None, help="Output file (default: print)")
    args = parser.parse_args()

    texts = TEXTS
    if args.tags:
        tags_lower = [t.lower() for t in args.tags]
        texts = [t for t in texts if has_tag(t, tags_lower)]
        print(f"Filtered to {len(texts)} texts containing: {args.tags}", file=sys.stderr)

    if args.limit:
        texts = texts[:args.limit]

    total_chars = sum(len(t) for t in texts)

    lines = []
    lines.append("=" * 70)
    lines.append(f"INFLECT PARALINGUISTIC TRAINING — {len(texts)} texts")
    lines.append(f"Total characters: {total_chars:,}  (ElevenLabs free = 10,000/mo)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("INSTRUCTIONS:")
    lines.append("  1. Pick a voice in ElevenLabs")
    lines.append("  2. Paste each text below, generate, download")
    lines.append("  3. Put all downloads in one folder named after the voice")
    lines.append("     e.g.  data/paralinguistic/downloads/rachel/")
    lines.append("  4. Repeat for each voice")
    lines.append("  5. Run import_elevenlabs_downloads.py to sort everything")
    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    for i, text in enumerate(texts):
        tags = re.findall(r'\[([^\]]+)\]', text)
        lines.append(f"[{i+1:03d}]  tags: {', '.join(tags)}")
        lines.append(text)
        lines.append("")

    output = "\n".join(lines)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = BASE / "data" / "paralinguistic_paste_list.txt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Texts: {len(texts)}  |  Chars: {total_chars:,}  |  ~{total_chars/10000:.1f}x free tier")


if __name__ == "__main__":
    main()
