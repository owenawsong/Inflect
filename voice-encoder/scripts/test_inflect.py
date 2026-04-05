"""
Inflect feature test suite.

Runs one clip per feature, all saved into outputs/test_<timestamp>/.
Open that folder in Explorer and listen through them in order.

Usage:
    python test_inflect.py                    # test everything
    python test_inflect.py --only sprites     # just sprite tags
    python test_inflect.py --only effects     # stutter + whisper
    python test_inflect.py --only fillers     # um/uh/hmm etc
    python test_inflect.py --only blend       # voice blending
    python test_inflect.py --only cloning     # reference conditioning
    python test_inflect.py --ref my_clip.wav  # use your own voice for cloning test
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PY      = sys.executable

def run(label: str, args: list[str], batch: str):
    cmd = [PY, str(SCRIPTS / "inflect_tts.py")] + args + ["--batch", batch, "--out", label]
    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(SCRIPTS.parent.parent))
    if result.returncode != 0:
        print(f"  [FAILED] {label}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        choices=["sprites", "effects", "fillers", "blend", "cloning", "pause"])
    parser.add_argument("--ref",  type=str, default=None,
                        help="Reference audio clip for cloning test")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch folder name (default: test_<timestamp>)")
    args = parser.parse_args()

    batch = args.batch or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    only  = args.only

    print(f"\nInflect Test Suite")
    print(f"Batch: {batch}")
    print(f"Output: voice-encoder/outputs/{batch}/\n")

    # ── 1. Sprite tags ────────────────────────────────────────────────────────
    if only in (None, "sprites"):
        run("01_laugh",  ["Hello! [laugh] That's so funny, I can't stop.", "--voice", "af_heart"], batch)
        run("02_sigh",   ["I don't know... [sigh] it's just really complicated.", "--voice", "af_heart"], batch)
        run("03_cough",  ["Excuse me. [cough] Sorry about that.", "--voice", "bm_fable"], batch)
        run("04_throat", ["Ahem. [throat] As I was saying.", "--voice", "bm_fable"], batch)
        run("05_sneeze", ["Oh no, I feel it coming... [sneeze] Ugh, sorry.", "--voice", "af_heart"], batch)
        run("06_sniff",  ["I walked into the bakery. [sniff] It smelled amazing.", "--voice", "af_heart"], batch)

    # ── 2. Effect tags ────────────────────────────────────────────────────────
    if only in (None, "effects"):
        run("07_stutter", ["Wait, [stutter] what did you just say to me?", "--voice", "af_heart"], batch)
        run("08_whisper", ["[whisper] I have a secret. Nobody else can know.", "--voice", "af_heart"], batch)
        run("09_stutter_mid", ["I was totally fine and then I just, [stutter] forgot everything.", "--voice", "bm_fable"], batch)

    # ── 3. Filler tags ────────────────────────────────────────────────────────
    if only in (None, "fillers"):
        run("10_um_uh",   ["I was thinking, [um] maybe we could, [uh] go somewhere?", "--voice", "af_heart"], batch)
        run("11_hmm",     ["[hmm] That's actually a pretty interesting idea.", "--voice", "af_heart"], batch)
        run("12_mixed_fillers", ["So I was like, [um] I don't know, [uh] it's weird. [hmm] Yeah.", "--voice", "bm_fable"], batch)

    # ── 4. Pause tags ─────────────────────────────────────────────────────────
    if only in (None, "pause"):
        run("13_pause_default", ["And then she said... [pause] nothing.", "--voice", "af_heart"], batch)
        run("14_pause_long",    ["The door opened. [pause:2.5] It was empty.", "--voice", "bm_fable"], batch)
        run("15_pause_short",   ["Ready? [pause:0.3] Go!", "--voice", "af_heart"], batch)

    # ── 5. Voice blending ─────────────────────────────────────────────────────
    if only in (None, "blend"):
        run("16_blend_50_50",   ["This is a blended voice, halfway between two speakers.", "--blend", "af_heart:0.5,bm_fable:0.5"], batch)
        run("17_blend_70_30",   ["This is mostly one voice with a hint of another.", "--blend", "af_heart:0.7,bm_fable:0.3"], batch)
        run("18_blend_sprites", ["I was like, [um] I don't know. [sigh] Whatever.", "--blend", "bf_isabella:0.6,am_michael:0.4"], batch)

    # ── 6. Pitch shift ────────────────────────────────────────────────────────
    if only in (None, "blend"):
        run("19_pitch_high",    ["Hello! This voice is pitched up a bit.", "--pitch", "1.15", "--voice", "af_heart"], batch)
        run("20_pitch_low",     ["Hello. This voice is pitched down a bit.", "--pitch", "0.88", "--voice", "bm_fable"], batch)

    # ── 7. Reference conditioning (voice cloning) ─────────────────────────────
    if only in (None, "cloning"):
        ref = args.ref
        if ref and Path(ref).exists():
            run("21_cloned",         ["Say something in my voice. Let's see how close this gets.", "--input", ref], batch)
            run("22_cloned_sprites", ["[hmm] Interesting. [sigh] I'm not sure about this. [laugh]", "--input", ref], batch)
        else:
            # Fall back to using an existing Kokoro output as reference
            fallback = str(Path(__file__).resolve().parent.parent / "outputs" / "inflect_out_af_heart.wav")
            if Path(fallback).exists():
                print(f"\n  [cloning] No --ref provided, using fallback: {fallback}")
                print(f"  (Pass --ref your_recording.wav for a real test)")
                run("21_cloned_fallback", ["Say something. Testing reference conditioning.", "--input", fallback], batch)
            else:
                print(f"\n  [cloning] Skipping — no ref clip found. Pass --ref your_recording.wav")

    # ── 8. Complex / kitchen-sink ─────────────────────────────────────────────
    if only is None:
        run("23_kitchen_sink", [
            "So I walked in and was like, [um] hello? [pause:0.8] "
            "[stutter] nobody answered. [sigh] I don't know, it was weird. [laugh]",
            "--voice", "af_heart"
        ], batch)

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / batch
    print(f"\n{'='*60}")
    print(f"All tests done.")
    print(f"Open: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
