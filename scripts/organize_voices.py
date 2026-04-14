#!/usr/bin/env python3
"""
Organize ElevenLabs voice preview files into reference_voices/ structure
for the VoxCPM dataset generator.

Also inventories paralinguistic clips (the numbered ones) as future
ZipVoice training data.

Usage:
  python scripts/organize_voices.py
  python scripts/organize_voices.py --downloads "C:/path/to/downloads"
"""

import argparse
import json
import re
import shutil
from pathlib import Path

DOWNLOADS   = Path("C:/Users/Owen/Downloads")
OUT_VOICES  = Path("reference_voices")
OUT_PARA    = Path("outputs/paralinguistic_raw")  # ZipVoice training data later

# ── Voice metadata inference ──────────────────────────────────────────────────
# Maps keywords in description → (gender, accent, style)
FEMALE_NAMES = {
    "aerisita", "addison", "belinda", "christina", "clara", "eryn",
    "florence", "hannah", "isla", "juno", "labhaoise", "maria", "shannon",
    "sia", "stacy", "tamsin", "verity",
}
MALE_NAMES = {
    "adam", "ak", "alex", "arthur", "chuck", "david", "denzel", "eno",
    "fanz", "frank", "grampa", "jake", "joel", "luke", "mark", "micky",
    "peter", "ron", "rory", "sam", "steven", "toby", "tony", "will", "wf",
}

ACCENT_KEYWORDS = [
    ("scottish",    "scottish"),
    ("british",     "british"),
    ("australian",  "australian"),
    ("canadian",    "canadian"),
    ("jamaican",    "jamaican"),
    ("indian",      "indian"),
    ("new zealand", "new_zealand"),
    ("northern",    "northern_british"),
    ("irish",       "irish"),
    ("american",    "american"),
]

STYLE_KEYWORDS = [
    ("asmr",        "asmr_whisper"),
    ("whisper",     "whisper"),
    ("raspy",       "raspy"),
    ("gravely",     "gravelly"),
    ("deep",        "deep"),
    ("warm",        "warm"),
    ("soft",        "soft"),
    ("energetic",   "energetic"),
    ("upbeat",      "upbeat"),
    ("lively",      "lively"),
    ("fast",        "fast"),
    ("calm",        "calm"),
    ("old",         "elderly"),
    ("cranky",      "elderly"),
    ("young",       "young"),
    ("youthful",    "young"),
    ("dark",        "dark"),
    ("bubbly",      "bubbly"),
    ("sweet",       "sweet"),
    ("soothing",    "soothing"),
    ("dry",         "dry"),
    ("detached",    "dry"),
    ("posh",        "posh"),
    ("storyteller", "storyteller"),
    ("narrator",    "narrator"),
    ("natural",     "natural"),
    ("clear",       "clear"),
    ("professional","professional"),
]

def infer_meta(first_name: str, description: str) -> dict:
    desc = description.lower()
    name = first_name.lower()

    # Gender
    if any(w in desc for w in ["feminine", "female", "woman", "girl"]):
        gender = "female"
    elif any(w in desc for w in ["male", "man", "boy", "gentleman"]):
        gender = "male"
    elif name in FEMALE_NAMES:
        gender = "female"
    elif name in MALE_NAMES:
        gender = "male"
    else:
        gender = "unknown"

    # Accent
    accent = "american"  # default
    for kw, val in ACCENT_KEYWORDS:
        if kw in desc:
            accent = val
            break

    # Style (take first match)
    style = "natural"
    for kw, val in STYLE_KEYWORDS:
        if kw in desc:
            style = val
            break

    return {"gender": gender, "accent": accent, "style": style, "description": description.strip()}


def parse_voice_preview(filename: str):
    """
    Parse 'voice_preview_adam - american, dark and tough.mp3'
    Returns (folder_name, first_name, description, is_duplicate_clip)
    """
    # Strip prefix and extension
    stem = filename.replace("voice_preview_", "")
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)  # remove (1) suffix
    stem = re.sub(r"\.mp3$", "", stem, flags=re.IGNORECASE)

    # Split on first " - " or " – " separator
    parts = re.split(r"\s[–-]\s", stem, maxsplit=1)
    first_name = parts[0].strip().lower()
    description = parts[1].strip() if len(parts) > 1 else ""

    # Normalize folder name
    folder = re.sub(r"[^a-z0-9]+", "_", first_name).strip("_")

    # If two voices share a first name (e.g., adam american vs adam scottish),
    # append accent disambiguation
    accent_hint = ""
    for kw, val in ACCENT_KEYWORDS:
        if kw in description.lower():
            accent_hint = f"_{val}"
            break

    return folder, first_name, description, accent_hint


def is_duplicate_clip(filename: str) -> bool:
    return bool(re.search(r"\(\d+\)", filename))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", type=Path, default=DOWNLOADS)
    ap.add_argument("--out-voices", type=Path, default=OUT_VOICES)
    ap.add_argument("--out-para",   type=Path, default=OUT_PARA)
    ap.add_argument("--move", action="store_true",
                    help="Move files instead of copying")
    args = ap.parse_args()

    dl   = args.downloads
    vout = args.out_voices
    pout = args.out_para

    vout.mkdir(parents=True, exist_ok=True)
    pout.mkdir(parents=True, exist_ok=True)

    # ── Group voice_preview files by voice ─────────────────────────────────────
    voice_files = sorted(dl.glob("voice_preview_*.mp3"))
    voices: dict[str, dict] = {}  # folder → {clips, meta}

    # First pass: detect name collisions (e.g., two adams)
    name_descriptions: dict[str, list] = {}
    for f in voice_files:
        folder, first, desc, _ = parse_voice_preview(f.name)
        name_descriptions.setdefault(folder, [])
        if desc not in name_descriptions[folder]:
            name_descriptions[folder].append(desc)

    collisions = {k for k, v in name_descriptions.items() if len(v) > 1}

    for f in voice_files:
        folder, first, desc, accent_hint = parse_voice_preview(f.name)

        # Disambiguate collisions (adam_american, adam_scottish)
        if folder in collisions and accent_hint:
            folder = folder + accent_hint

        if folder not in voices:
            voices[folder] = {
                "clips": [],
                "meta": infer_meta(first, desc),
            }
        voices[folder]["clips"].append(f)

    # ── Copy/move voices ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Organizing {len(voices)} voices → {vout.resolve()}")
    print(f"{'─'*60}")

    for folder, data in sorted(voices.items()):
        vdir = vout / folder
        vdir.mkdir(exist_ok=True)

        for i, src in enumerate(data["clips"]):
            suffix = f"_{i+1}" if len(data["clips"]) > 1 else ""
            dst = vdir / f"{folder}{suffix}.mp3"
            if args.move:
                shutil.move(str(src), dst)
            else:
                shutil.copy2(src, dst)

        # Write meta.json
        (vdir / "meta.json").write_text(
            json.dumps(data["meta"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        clips_str = f"{len(data['clips'])} clip{'s' if len(data['clips']) > 1 else ' '}"
        m = data["meta"]
        print(f"  {folder:<30} {clips_str}  [{m['gender']}, {m['accent']}, {m['style']}]")

    # ── Collect paralinguistic clips ────────────────────────────────────────────
    all_mp3 = list(dl.glob("*.mp3"))
    para_clips = [
        f for f in all_mp3
        if not f.name.startswith("voice_preview_")
        and not any(skip in f.name.lower() for skip in [
            "after effects", "boy in space", "cold", "sojourn", "fade",
            "tell-me", "testing", "test_download", "いつまでも",
        ])
        and f.stem[-1].isdigit() or f.name.replace("(", "").replace(")", "").strip().endswith(".mp3")
    ]

    # Simpler: anything not voice_preview and not music
    MUSIC_PATTERNS = [
        "after effects", "boy in space", "cold instrumental", "sojourn",
        "fade to black", "tell-me-you-love-me", "testing 123", "test_download",
        "いつまでも",
    ]
    para_clips = [
        f for f in all_mp3
        if not f.name.startswith("voice_preview_")
        and not any(p in f.name.lower() for p in MUSIC_PATTERNS)
        and not f.name.startswith("68c6f2")  # random uuid
    ]

    if para_clips:
        print(f"\n{'─'*60}")
        print(f"  Paralinguistic clips found: {len(para_clips)}")
        print(f"  → Will be used for ZipVoice fine-tuning later")
        print(f"  → Copying to {pout.resolve()}")
        print(f"{'─'*60}")

        for f in para_clips:
            dst = pout / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

        # Write a quick inventory
        inv_path = pout / "inventory.txt"
        with open(inv_path, "w", encoding="utf-8") as fp:
            for f in sorted(para_clips):
                fp.write(f.name + "\n")
        print(f"  Inventory → {inv_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  ✓ {len(voices)} voice folders created in {vout}")
    print(f"  ✓ {len(para_clips)} paralinguistic clips saved for ZipVoice")
    print(f"\n  Now run the generator:")
    print(f"    python scripts/generate_voxcpm_dataset.py")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
