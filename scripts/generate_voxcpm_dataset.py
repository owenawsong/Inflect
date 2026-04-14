#!/usr/bin/env python3
"""
VoxCPM Dataset Generator  —  HuggingFace-ready
================================================
Sends concurrent generation requests to the VoxCPM-Demo HF Space and saves
outputs in AudioFolder format, ready to push_to_hub.

Output layout:
  <out>/
    audio/           ← all .wav files
    metadata.csv     ← HF AudioFolder manifest (file_name col = relative path)
    README.md        ← auto-generated dataset card
    voice_transcripts.json  ← cached transcripts for ultimate-clone mode

Usage:
  python scripts/generate_voxcpm_dataset.py                  # run generator
  python scripts/generate_voxcpm_dataset.py --discover       # print Space API
  python scripts/generate_voxcpm_dataset.py --workers 4 --out outputs/my_ds

Adding voices:
  Drop folders of .wav files under reference_voices/<voice_name>/
  Optional: add reference_voices/<voice_name>/meta.json with
    {"gender": "female", "accent": "american", "style": "warm"}
  The script auto-discovers everything.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import shutil
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

try:
    from gradio_client import Client, handle_file
except ImportError:
    raise SystemExit("pip install gradio_client")

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

# ── Config ───────────────────────────────────────────────────────────────────
SPACE            = "openbmb/VoxCPM-Demo"
API_ENDPOINT     = "/generate"
DEFAULT_WORKERS  = 4
DEFAULT_OUT      = Path("outputs/voxcpm_dataset")
DEFAULT_VOICES   = Path("reference_voices")
FALLBACK_VOICES  = Path("voice-encoder/data/paralinguistic")  # existing ElevenLabs clips

# Generation mode probabilities
P_ULTIMATE = 1.00   # ultimate clone  — best voice ID accuracy + transcript guidance
P_CLONE    = 0.00   # standard clone  — disabled (control instructions don't work with reference audio anyway)
P_DESIGN   = 0.00   # disabled

# ── Voice design descriptions ─────────────────────────────────────────────────
VOICE_DESIGNS = [
    "warm and friendly young woman, conversational tone",
    "calm middle-aged man with a deep resonant voice, clear articulation",
    "energetic young man, enthusiastic and upbeat delivery",
    "soft-spoken older woman, gentle and reassuring tone",
    "confident professional woman, authoritative but warm",
    "cheerful and bright young woman, natural pacing",
    "thoughtful man in his thirties, measured and precise speech",
    "expressive storyteller, dynamic and engaging cadence",
    "casual and relaxed young woman, slightly breathy",
    "warm baritone man, deliberate and commanding",
    "young woman with a slight British accent, polished",
    "older gentleman, distinguished and unhurried",
    "enthusiastic teenager, rapid and energetic",
    "calm narrator voice, smooth and neutral",
    "southern American woman, warm with regional inflection",
]

# ── Text corpus (500+ sentences, varied length / topic / emotion) ─────────────
# Each entry: (text, category)
CORPUS = [
    # ── SHORT / PUNCHY (1–3 s) ────────────────────────────────────────────────
    ("Wait, what?", "short"),
    ("No way. Absolutely not.", "short"),
    ("I can't believe you just said that.", "short"),
    ("That's actually hilarious.", "short"),
    ("Hold on a second.", "short"),
    ("Are you serious right now?", "short"),
    ("I knew it.", "short"),
    ("That's not what I expected.", "short"),
    ("Oh no. Oh no no no.", "short"),
    ("Okay. Okay okay okay.", "short"),
    ("That's fair.", "short"),
    ("I have no idea.", "short"),
    ("You have to be kidding me.", "short"),
    ("This is fine.", "short"),
    ("Nope. Not doing that.", "short"),
    ("I mean, sure.", "short"),
    ("What do you mean?", "short"),
    ("That's kind of wild actually.", "short"),
    ("I did not see that coming.", "short"),
    ("Yeah, no. Hard pass.", "short"),
    ("Right. Moving on.", "short"),
    ("Noted.", "short"),
    ("That tracks.", "short"),
    ("Bold of you to assume.", "short"),
    ("Okay but hear me out.", "short"),
    ("Not my finest moment.", "short"),
    ("In hindsight, obvious.", "short"),
    ("We don't talk about that.", "short"),
    ("I stand by what I said.", "short"),
    ("Not ideal, but workable.", "short"),
    ("I've said too much.", "short"),
    ("You're not wrong.", "short"),
    ("We'll circle back.", "short"),
    ("I mean. Yeah. Kind of.", "short"),

    # ── CONVERSATIONAL MEDIUM (3–7 s) ─────────────────────────────────────────
    ("So I was just sitting there, completely minding my own business.", "conversational"),
    ("The funniest part is he didn't even realize what happened.", "conversational"),
    ("You have to hear this story because it's almost too good to be true.", "conversational"),
    ("I've been thinking about it all day and I still can't figure it out.", "conversational"),
    ("She looked at me and said nothing, and somehow that was worse.", "conversational"),
    ("It wasn't until later that I realized what had actually happened.", "conversational"),
    ("I keep telling myself it doesn't matter, but it kind of does.", "conversational"),
    ("The thing is, I actually agree with you, which is strange.", "conversational"),
    ("I don't know how to explain it, but something just felt off.", "conversational"),
    ("We've been talking about this for weeks and we're still nowhere.", "conversational"),
    ("He called his teacher mom in front of the whole class and never recovered.", "conversational"),
    ("I applied to fourteen jobs this month. Two responses. Both rejections.", "conversational"),
    ("She said she'd think about it. That was three weeks ago.", "conversational"),
    ("We got there and the entire place was completely packed.", "conversational"),
    ("He kept talking and I kept nodding and neither of us was listening.", "conversational"),
    ("I don't know. It's just been one of those weeks, you know?", "conversational"),
    ("She finally said what everyone was thinking and the room went silent.", "conversational"),
    ("I thought I had it all figured out, and then this happened.", "conversational"),
    ("I have the most insane idea and you need to just hear me out.", "conversational"),
    ("We ended up staying until two in the morning talking about nothing.", "conversational"),
    ("He called it a minor setback. I called it a disaster.", "conversational"),
    ("So that's the situation. I know. I know.", "conversational"),
    ("She showed up three hours late and acted like nothing had happened.", "conversational"),
    ("I told myself one more episode. That was five hours ago.", "conversational"),
    ("The meeting that could have been an email lasted two hours.", "conversational"),
    ("He said he was five minutes away for forty-five minutes.", "conversational"),
    ("I wasn't even supposed to be there. That's the wild part.", "conversational"),
    ("We ordered way too much food and then finished all of it anyway.", "conversational"),
    ("I made a pros and cons list and it did not help at all.", "conversational"),
    ("She knew exactly what she was doing the whole time.", "conversational"),

    # ── NARRATIVE / STORYTELLING (5–10 s) ─────────────────────────────────────
    ("So this is the part of the story where everything goes completely sideways.", "narrative"),
    ("I remember the exact moment I realized I had made a huge mistake.", "narrative"),
    ("There was this one day last summer when everything just clicked into place.", "narrative"),
    ("We had been driving for six hours when we finally saw the sign for the town.", "narrative"),
    ("She walked into the room and everyone went quiet, not because she was loud, but because she wasn't.", "narrative"),
    ("I had rehearsed what I was going to say a hundred times, and when the moment came, I forgot all of it.", "narrative"),
    ("The weird thing about growing up is that you don't notice it happening until it already has.", "narrative"),
    ("I've been thinking a lot about that conversation, the one we had in the parking lot after dinner.", "narrative"),
    ("There's something about that specific kind of silence that I've never been able to describe.", "narrative"),
    ("He told the story three different times that night and somehow it got better each time.", "narrative"),
    ("The moment I opened the door I knew something was different, even if I couldn't say what.", "narrative"),
    ("She had a way of making you feel like the most interesting person in the room.", "narrative"),
    ("We sat on the roof for hours watching the city lights and not saying much of anything.", "narrative"),
    ("The thing nobody tells you about starting over is how ordinary most of it feels.", "narrative"),
    ("I don't remember making the decision. I just remember that one day everything was different.", "narrative"),
    ("It was the kind of afternoon where nothing important happened, but you remember it anyway.", "narrative"),
    ("The drive back was quiet in a way that felt intentional.", "narrative"),
    ("I wrote it all down that night so I wouldn't forget, and I still haven't looked at it.", "narrative"),
    ("He had this theory, and the more time passes, the more I think he was right.", "narrative"),
    ("That was the last summer before everything changed, though we didn't know it yet.", "narrative"),

    # ── EMOTIONAL / EXPRESSIVE ─────────────────────────────────────────────────
    ("He said he was proud of me and nobody had ever said that to me before.", "emotional"),
    ("I was so excited I could barely form a complete sentence.", "emotional"),
    ("It's the kind of thing that stays with you even years later.", "emotional"),
    ("I honestly did not think I was going to make it through that year.", "emotional"),
    ("She started laughing and I started laughing and we couldn't stop for a full minute.", "emotional"),
    ("I was devastated. I'm not going to pretend otherwise.", "emotional"),
    ("There's this feeling you get when something finally works after weeks of failure.", "emotional"),
    ("I've never been so relieved in my entire life.", "emotional"),
    ("It's hard to explain how much that moment meant to me.", "emotional"),
    ("I didn't expect to feel that way about it, but I did.", "emotional"),
    ("She cried. I cried. We all just sort of stood there crying.", "emotional"),
    ("That was the scariest thing I've ever done, and also maybe the best.", "emotional"),
    ("I remember thinking: this is it. This is the thing I've been waiting for.", "emotional"),
    ("It hurt more than I expected, and I thought I was prepared.", "emotional"),
    ("The joy of it was almost overwhelming, like too much good all at once.", "emotional"),
    ("I was furious. Not the loud kind. The quiet kind that lasts for days.", "emotional"),
    ("Something about that moment broke something open in me that I hadn't realized was closed.", "emotional"),
    ("I kept it together until I got to the car, and then I completely fell apart.", "emotional"),
    ("It's the kind of happy that makes you nervous because you know it can't last.", "emotional"),
    ("I have never felt more seen in my entire life than in that moment.", "emotional"),

    # ── TECHNICAL / INSTRUCTIONAL ─────────────────────────────────────────────
    ("The first thing you want to do is make sure everything is backed up.", "technical"),
    ("It's a simple process once you understand the underlying structure.", "technical"),
    ("The key thing to remember is that order matters here more than you'd think.", "technical"),
    ("If you run into that error, the fix is usually just restarting the service.", "technical"),
    ("The tricky part isn't the setup, it's making sure all dependencies line up.", "technical"),
    ("Double-check that value before you push anything to production.", "technical"),
    ("The system runs on three core components that all have to stay in sync.", "technical"),
    ("Once it's configured correctly, it basically runs itself.", "technical"),
    ("There's a known issue with that version that causes failures under load.", "technical"),
    ("The performance improvement is real, but it comes with tradeoffs.", "technical"),
    ("You'll want to isolate that variable before drawing any conclusions.", "technical"),
    ("The logs are your best friend when something behaves unexpectedly.", "technical"),
    ("Always test on a copy before modifying the original.", "technical"),
    ("The latency goes up significantly at scale, which is expected.", "technical"),
    ("Version pinning will save you a lot of pain later down the line.", "technical"),

    # ── QUESTIONS / INQUISITIVE ────────────────────────────────────────────────
    ("Have you ever thought about why we keep doing things the same way when they stop working?", "question"),
    ("What exactly did she say when you told her?", "question"),
    ("Do you think it would have gone differently if we'd left earlier?", "question"),
    ("When did you first realize something was wrong?", "question"),
    ("Why does asking for help feel so hard when you actually need it?", "question"),
    ("Is there a version of this where it actually works out okay?", "question"),
    ("How long have you been sitting on this?", "question"),
    ("What are we actually trying to accomplish here?", "question"),
    ("Does it matter at this point, honestly?", "question"),
    ("Who told you that was the right way to do it?", "question"),
    ("Have you ever just stopped and thought about how strange all of this is?", "question"),
    ("What would you do differently if you could go back?", "question"),
    ("Is that something you actually want, or something you think you should want?", "question"),
    ("How do you explain something that doesn't make sense even to you?", "question"),
    ("What exactly are we waiting for at this point?", "question"),

    # ── DESCRIPTIONS / OBSERVATIONS ───────────────────────────────────────────
    ("The sky that day was a strange shade of orange I'd never seen before.", "descriptive"),
    ("There's a coffee shop on the corner that's been there longer than anyone can remember.", "descriptive"),
    ("The city at night looks completely different from how it feels during the day.", "descriptive"),
    ("It's one of those places where everyone seems to know each other already.", "descriptive"),
    ("The room was filled with the quiet that means something is about to change.", "descriptive"),
    ("She had a habit of tilting her head slightly when thinking hard about something.", "descriptive"),
    ("The rain started slowly, and then all at once, the way it usually does.", "descriptive"),
    ("There's a smell in old buildings that's impossible to describe but immediately familiar.", "descriptive"),
    ("The light at that time of afternoon is perfect for about twenty minutes.", "descriptive"),
    ("It's a small town. Not the kind you leave. The kind you leave and come back to.", "descriptive"),
    ("The kind of quiet you only get at three in the morning in a city.", "descriptive"),
    ("It had the look of a place that used to mean something to someone.", "descriptive"),
    ("The whole place smelled like pine and old wood and cold air.", "descriptive"),
    ("It was the sort of morning where the whole world looks rinsed clean.", "descriptive"),
    ("There's a specific kind of tired that sleep doesn't fix.", "descriptive"),

    # ── LONG / COMPLEX (10–18 s) ──────────────────────────────────────────────
    ("I've thought a lot about what I would do differently if I could go back, and the honest answer is probably not as much as I'd like to think.", "long"),
    ("The hardest part wasn't making the decision, it was sitting with the uncertainty of not knowing whether it was the right one for months afterward.", "long"),
    ("There's an assumption that people who seem confident all the time actually feel confident, and in my experience that's almost never true.", "long"),
    ("We spent three years building something we were genuinely proud of, and then the market shifted and none of it mattered the way we thought it would.", "long"),
    ("I used to think the goal was to get to a point where nothing was hard anymore, and now I think that point doesn't exist and I'm not sure I'd want it to.", "long"),
    ("She told me once that the most important skill she'd developed was learning to disagree with someone without making them feel they were wrong.", "long"),
    ("The thing about making something good is that you don't usually know it's good while you're making it. You just know you're not done yet.", "long"),
    ("There were days last year when I genuinely didn't know how I was going to get through it, and somehow I did, and I still don't fully understand how.", "long"),
    ("He had a theory that most arguments aren't actually about what they seem to be about, and the longer I live the more I think he was onto something.", "long"),
    ("I've learned more from the things that went wrong than from the things that worked, which is a frustrating way to have to learn things but apparently the only way that stuck.", "long"),
    ("The strange thing about really wanting something is that getting it never quite feels the way you imagined it would, and not getting it never quite feels as bad.", "long"),
    ("I think the most underrated skill in the world is knowing when to stop explaining yourself and just let people draw their own conclusions.", "long"),
    ("There's a version of being busy that's actually just a very elaborate way of avoiding the one thing you know you should be doing.", "long"),
    ("The older I get, the more I think that most of what we call wisdom is just pattern recognition built from having been wrong enough times to start noticing.", "long"),
    ("If I'm being completely honest, I had no idea what I was doing for most of it, and I think that might be the most normal thing about the whole experience.", "long"),

    # ── MIXED / SPECIFIC SCENARIOS ─────────────────────────────────────────────
    ("You got engaged? When? How? Tell me everything right now.", "conversational"),
    ("That's the worst idea I've ever heard and I'm completely in.", "conversational"),
    ("I've been faking that I understood this the entire time.", "conversational"),
    ("She still doesn't know it was me, and I'd like to keep it that way.", "conversational"),
    ("He walked face-first into the glass door. I tried not to laugh. I failed.", "conversational"),
    ("I specifically asked for one thing. One thing. And it still didn't happen.", "emotional"),
    ("Don't look now but I think that's him over there.", "conversational"),
    ("Oh my god. This is the best thing I've ever tasted.", "emotional"),
    ("We never talked about it again, which I think was the right call.", "narrative"),
    ("I have fought armies. I have crossed oceans. I did not come this far to lose.", "narrative"),
    ("The moment you stop expecting perfection, things get a lot more interesting.", "descriptive"),
    ("It was a disaster in the best possible way.", "conversational"),
    ("I would not say that again in this room if I were you.", "conversational"),
    ("You're going to laugh at this. Actually you're already laughing.", "conversational"),
    ("She knew. She had always known. She just waited for the right moment.", "narrative"),
    ("I need you to understand how close that was to not working.", "conversational"),
    ("He looked at me like I had said something in a language he didn't speak.", "narrative"),
    ("It took three tries, a lot of frustration, and one very long phone call.", "conversational"),
    ("And somehow, despite all of that, it actually turned out okay.", "narrative"),
    ("I would describe the situation as controlled chaos, emphasis on chaos.", "conversational"),
    ("The plan was simple. The execution was anything but.", "narrative"),
    ("Nobody told me it was going to be this hard, and honestly, I'm grateful.", "emotional"),
    ("It's one of those decisions that makes total sense in the moment.", "conversational"),
    ("She handled it with a grace I genuinely did not expect.", "narrative"),
    ("We improvised the entire second half and somehow nobody noticed.", "conversational"),
    ("This is either going to be a great story or a terrible lesson.", "conversational"),
    ("I cannot stress enough how much I did not want to be right about this.", "emotional"),
    ("At some point you just have to commit and see what happens.", "conversational"),
    ("The timing could not have been worse, and yet here we are.", "conversational"),
    ("That's the part of the story I usually leave out.", "narrative"),

    # ── READING ALOUD / BOOK STYLE ─────────────────────────────────────────────
    ("Chapter seven. In which everything that could go wrong did, and a few things that couldn't.", "narrative"),
    ("The letter arrived on a Tuesday, which felt appropriate somehow.", "narrative"),
    ("He stood at the edge of the cliff and wondered, not for the first time, how he'd gotten here.", "narrative"),
    ("It was the sort of place that made you feel like you were being watched, even when you weren't.", "narrative"),
    ("She opened the door slowly, as if she expected to find something she wasn't ready for.", "narrative"),
    ("The village had a name, but nobody used it. Everyone just called it home.", "narrative"),
    ("Three things happened on the morning of the fourteenth, and none of them made sense alone.", "narrative"),
    ("He had exactly seven minutes to change his mind, and he spent all of them standing still.", "narrative"),
    ("The note was short, which was somehow more frightening than a long one would have been.", "narrative"),
    ("They had met before, both of them knew it, and neither of them mentioned it.", "narrative"),
]

# ── Stats ─────────────────────────────────────────────────────────────────────
_lock       = threading.Lock()
_total_gen  = 0
_total_err  = 0
_reserved   = 0
_start_time = time.time()
_existing   = 0  # clips already in output dir on start
_corpus_lock = threading.Lock()
_stop_after = None


def _is_space_transport_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in [
            "unexpected sse line",
            "<!doctype html",
            "<html",
            "too many people",
            "queue",
            "busy",
            "connection",
            "timed out",
            "temporarily unavailable",
            "service unavailable",
            "502",
            "503",
            "504",
        ]
    )


def _new_client():
    return Client(SPACE, verbose=False)

def _try_reserve_slot() -> bool:
    global _reserved
    with _lock:
        total = _existing + _total_gen + _reserved
        if _stop_after is not None and total >= _stop_after:
            return False
        _reserved += 1
        return True

def _finish_reserved(success: bool):
    global _total_gen, _total_err, _reserved
    with _lock:
        _reserved = max(0, _reserved - 1)
        if success:
            _total_gen += 1
        else:
            _total_err += 1

def _print_stats():
    elapsed = time.time() - _start_time
    rate    = _total_gen / (elapsed / 3600) if elapsed > 1 else 0
    total   = _existing + _total_gen
    print(f"\r  ✓ {_total_gen} new  ({total} total)  ✗ {_total_err} err  "
          f"⏱ {elapsed/60:.1f}m  {rate:.0f}/hr   ", end="", flush=True)

# ── Voice loader ──────────────────────────────────────────────────────────────
class VoicePool:
    """Discovers reference voices and their metadata from a directory tree."""

    def __init__(self, paths: list[Path]):
        self.voices = {}   # name → {wavs: [], meta: {}}
        self._load(paths)

    def _load(self, paths):
        for base in paths:
            if not base.exists():
                continue
            for voice_dir in sorted(base.iterdir()):
                if not voice_dir.is_dir():
                    continue
                # Accept both .wav and .mp3 files
                wavs = sorted(voice_dir.glob("*.wav")) + sorted(voice_dir.glob("*.mp3"))
                if not wavs:
                    continue
                name = voice_dir.name
                meta_path = voice_dir / "meta.json"
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        pass
                # build transcript map from .txt sidecars
                transcripts = {}
                for w in wavs:
                    t = w.with_suffix(".txt")
                    if t.exists():
                        transcripts[w.name] = t.read_text(encoding="utf-8").strip()
                self.voices[name] = {
                    "wavs": wavs,
                    "meta": meta,
                    "transcripts": transcripts,
                }

    def pick(self):
        """Returns (voice_name, wav_path, transcript_or_None, meta)."""
        if not self.voices:
            return None, None, None, {}
        name  = random.choice(list(self.voices.keys()))
        entry = self.voices[name]
        wav   = random.choice(entry["wavs"])
        tx    = entry["transcripts"].get(wav.name)
        return name, wav, tx, entry["meta"]

    def __len__(self):
        return len(self.voices)

    def names(self):
        return sorted(self.voices.keys())


class TextPool:
    """Cycles through texts in shuffled passes so repeats stay balanced."""

    def __init__(self, corpus: list[tuple[str, str]], existing_rows: list[dict] | None = None, seed: int = 42):
        self.corpus = list(corpus)
        self.rng = random.Random(seed)
        self.by_key = {(text, category): (text, category) for text, category in self.corpus}
        self.counts = Counter()
        self.pending = Counter()
        self.queue = deque()
        if existing_rows:
            for row in existing_rows:
                key = (row.get("text", ""), row.get("category", ""))
                if key in self.by_key:
                    self.counts[key] += 1
        self._rebuild_queue()

    def _rebuild_queue(self):
        if not self.corpus:
            return
        min_count = min(
            self.counts.get((text, category), 0) + self.pending.get((text, category), 0)
            for text, category in self.corpus
        )
        candidates = [
            (text, category)
            for text, category in self.corpus
            if self.counts.get((text, category), 0) + self.pending.get((text, category), 0) == min_count
        ]
        self.rng.shuffle(candidates)
        self.queue.extend(candidates)

    def claim(self) -> tuple[str, str]:
        with _corpus_lock:
            if not self.queue:
                self._rebuild_queue()
            text, category = self.queue.popleft()
            self.pending[(text, category)] += 1
            return text, category

    def finalize(self, key: tuple[str, str], success: bool):
        with _corpus_lock:
            if self.pending[key] > 0:
                self.pending[key] -= 1
            if success:
                self.counts[key] += 1


class VoiceChooser:
    """Prefer unseen voices for each text before repeating a text-voice pair."""

    def __init__(self, pool: VoicePool, existing_rows: list[dict] | None = None, seed: int = 42):
        self.pool = pool
        self.rng = random.Random(seed)
        self.counts = {}
        self.pending = {}
        if existing_rows:
            for row in existing_rows:
                key = (row.get("text", ""), row.get("category", ""))
                voice = row.get("voice_id", "")
                if not voice:
                    continue
                self.counts.setdefault(key, Counter())[voice] += 1

    def claim(self, key: tuple[str, str]):
        names = self.pool.names()
        used = self.counts.setdefault(key, Counter())
        pend = self.pending.setdefault(key, Counter())
        totals = {name: used.get(name, 0) + pend.get(name, 0) for name in names}
        min_count = min(totals.values()) if totals else 0
        candidates = [name for name, count in totals.items() if count == min_count]
        voice_name = self.rng.choice(candidates)
        pend[voice_name] += 1
        entry = self.pool.voices[voice_name]
        wav = self.rng.choice(entry["wavs"])
        transcript = entry["transcripts"].get(wav.name)
        return voice_name, wav, transcript, entry["meta"]

    def finalize(self, key: tuple[str, str], voice_name: str, success: bool):
        pend = self.pending.setdefault(key, Counter())
        if pend.get(voice_name, 0) > 0:
            pend[voice_name] -= 1
        if success:
            self.counts.setdefault(key, Counter())[voice_name] += 1

# ── Manifest ──────────────────────────────────────────────────────────────────
MANIFEST_COLS = [
    "file_name", "text", "category", "voice_id", "mode",
    "speaker_gender", "speaker_accent", "speaker_style",
    "control_instruction", "prompt_text", "cfg_value",
    "duration_s", "generated_at",
]

def _init_manifest(path: Path):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_COLS).writeheader()

def _append_row(path: Path, row: dict):
    with _lock:
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
            w.writerow(row)

def _load_existing(manifest_path: Path) -> set[str]:
    """Return set of file_names already in manifest (for resume support)."""
    seen = set()
    if not manifest_path.exists():
        return seen
    try:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row.get("file_name", ""))
    except Exception:
        pass
    return seen


def _load_existing_rows(manifest_path: Path) -> list[dict]:
    rows = []
    if not manifest_path.exists():
        return rows
    try:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    except Exception:
        pass
    return rows


def _load_corpus(corpus_path: Path | None) -> list[tuple[str, str]]:
    if corpus_path is None:
        return list(CORPUS)

    suffix = corpus_path.suffix.lower()
    entries: list[tuple[str, str]] = []

    if suffix == ".json":
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("items", [])
        for item in data:
            text = str(item.get("text", "")).strip()
            category = str(item.get("category", "misc")).strip() or "misc"
            if text:
                entries.append((text, category))
    else:
        with open(corpus_path, newline="", encoding="utf-8") as f:
            sample = f.read(2048)
            f.seek(0)
            if "\t" in sample:
                reader = csv.reader(f, delimiter="\t")
            else:
                reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) == 1:
                    text = row[0].strip()
                    category = "misc"
                else:
                    text = row[0].strip()
                    category = row[1].strip() or "misc"
                if text and text.lower() != "text":
                    entries.append((text, category))

    dedup = []
    seen = set()
    for text, category in entries:
        key = (text, category)
        if key not in seen:
            seen.add(key)
            dedup.append(key)
    return dedup

# ── Audio duration ─────────────────────────────────────────────────────────────
def _dur(path: Path) -> str:
    if not _HAS_SF:
        return ""
    try:
        return f"{sf.info(str(path)).duration:.2f}"
    except Exception:
        return ""

# ── Single generation ──────────────────────────────────────────────────────────
def _generate_one(
    client,
    audio_dir: Path,
    manifest_path: Path,
    voice_pool: VoicePool,
    text_pool: TextPool,
    voice_chooser: VoiceChooser,
):
    text, category = text_pool.claim()
    key = (text, category)
    cfg = random.choice([1.9, 2.0, 2.0, 2.1])

    roll = random.random()

    # Initialize all generation params
    mode = None
    voice_id = None
    wav = None
    transcript = None
    meta = {}
    kwargs = {}

    if voice_pool and roll < P_ULTIMATE:
        # ── Ultimate clone ─────────────────────────────────────────────────────
        voice_id, wav, transcript, meta = voice_chooser.claim(key)
        if wav is None:
            roll = 1.0  # fall through to design
        else:
            mode        = "ultimate"
            has_tx      = transcript is not None
            kwargs = dict(
                text_input               = text,
                control_instruction      = "",
                reference_wav_path_input = handle_file(str(wav)),
                use_prompt_text          = has_tx,
                prompt_text_input        = transcript or "",
                cfg_value_input          = cfg,
                do_normalize             = True,
                denoise                  = False,
            )

    if voice_pool and roll >= P_ULTIMATE and roll < P_ULTIMATE + P_CLONE:
        # ── Standard clone ─────────────────────────────────────────────────
        voice_id, wav, _, meta = voice_chooser.claim(key)
        if wav is None:
            roll = 1.0
        else:
            mode   = "clone"
            kwargs = dict(
                text_input               = text,
                control_instruction      = "",
                reference_wav_path_input = handle_file(str(wav)),
                use_prompt_text          = False,
                prompt_text_input        = "",
                cfg_value_input          = cfg,
                do_normalize             = True,
                denoise                  = False,
            )
            transcript = None

    if roll >= P_ULTIMATE + P_CLONE or not voice_pool or mode is None:
        # ── Voice design ───────────────────────────────────────────────────────
        mode        = "design"
        voice_id    = "design"
        ctrl        = random.choice(VOICE_DESIGNS)
        transcript  = None
        meta        = {}
        wav         = None
        kwargs = dict(
            text_input               = text,
            control_instruction      = ctrl,
            reference_wav_path_input = None,
            use_prompt_text          = False,
            prompt_text_input        = "",
            cfg_value_input          = cfg,
            do_normalize             = True,
            denoise                  = False,
        )

    # ── Call Space ─────────────────────────────────────────────────────────────
    result = client.predict(**kwargs, api_name=API_ENDPOINT)

    # gradio_client returns a filepath string for Audio outputs
    if isinstance(result, str):
        src = Path(result)
    elif isinstance(result, (list, tuple)):
        src = Path(result[0]) if result and isinstance(result[0], str) else None
    else:
        src = None

    if src is None or not src.exists():
        raise RuntimeError("No output file returned from space")

    # ── Save ───────────────────────────────────────────────────────────────────
    uid      = hashlib.md5(f"{text}{voice_id}{time.time_ns()}".encode()).hexdigest()[:12]
    filename = f"{voice_id}_{mode}_{uid}.wav"
    rel_path = f"audio/{filename}"
    dst      = audio_dir / filename
    shutil.copy2(src, dst)

    _append_row(manifest_path, {
        "file_name"           : rel_path,
        "text"                : text,
        "category"            : category,
        "voice_id"            : voice_id,
        "mode"                : mode,
        "speaker_gender"      : meta.get("gender", ""),
        "speaker_accent"      : meta.get("accent", ""),
        "speaker_style"       : meta.get("style", ""),
        "control_instruction" : kwargs.get("control_instruction", ""),
        "prompt_text"         : transcript or "",
        "cfg_value"           : cfg,
        "duration_s"          : _dur(dst),
        "generated_at"        : datetime.now().isoformat(timespec="seconds"),
    })
    return key, voice_id

# ── Dataset card ───────────────────────────────────────────────────────────────
def _write_card(out_dir: Path, pool: VoicePool, n_texts: int):
    voices_str = ", ".join(pool.names()) if pool else "voice design only"
    card = f"""---
license: other
task_categories:
  - text-to-speech
language:
  - en
tags:
  - voxcpm2
  - tts
  - voice-cloning
  - synthetic
  - distillation
---

# VoxCPM2 Synthetic TTS Dataset

Generated using the [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) model
via the [VoxCPM-Demo](https://huggingface.co/spaces/openbmb/VoxCPM-Demo) HF Space.

Intended for TTS model fine-tuning and knowledge distillation.

Important licensing note:

- this dataset is synthetic audio, not the original prompt-reference audio
- prompt-source rights and upstream model terms still matter
- the dataset is marked `license: other` because it should not inherit the repo code license automatically

## Generation details

| Field | Value |
|---|---|
| Source model | VoxCPM2 (2B) |
| Mode split | 100% ultimate clone (best voice ID accuracy with transcript guidance) |
| Text corpus | {n_texts} unique sentences |
| Reference voices | {voices_str} |
| CFG range | 1.9 – 2.1 |

## Columns

| Column | Description |
|---|---|
| `file_name` | Relative path to audio file |
| `text` | Spoken text |
| `category` | Sentence category (short / conversational / narrative / emotional / ...) |
| `voice_id` | Reference voice name or "design" |
| `mode` | Generation mode: ultimate / clone / design |
| `speaker_gender` | Gender metadata if provided |
| `speaker_accent` | Accent metadata if provided |
| `speaker_style` | Style metadata if provided |
| `control_instruction` | Voice design description (design mode only) |
| `prompt_text` | Reference transcript (ultimate clone mode only) |
| `cfg_value` | CFG guidance scale used |
| `duration_s` | Audio duration in seconds |
| `generated_at` | ISO timestamp |
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")

# ── Worker ─────────────────────────────────────────────────────────────────────
def _worker(
    wid: int,
    audio_dir: Path,
    manifest_path: Path,
    pool: VoicePool,
    text_pool: TextPool,
    voice_chooser: VoiceChooser,
    stop: threading.Event,
):
    print(f"[w{wid}] connecting…")
    client = None
    try:
        client = _new_client()
    except Exception as e:
        print(f"\n[w{wid}] connect failed: {e}")
        return
    print(f"[w{wid}] ready")
    backoff = 1
    consecutive_failures = 0
    while not stop.is_set():
        if client is None:
            try:
                print(f"[w{wid}] reconnecting…")
                client = _new_client()
                print(f"[w{wid}] ready")
            except Exception as reconnect_error:
                print(f"[w{wid}] reconnect failed: {reconnect_error}")
                time.sleep(min(backoff, 60))
                backoff = min(backoff * 2, 45)
                continue
        if not _try_reserve_slot():
            stop.set()
            break
        key = None
        voice_id = None
        try:
            key, voice_id = _generate_one(client, audio_dir, manifest_path, pool, text_pool, voice_chooser)
            text_pool.finalize(key, True)
            if voice_id != "design":
                voice_chooser.finalize(key, voice_id, True)
            _finish_reserved(True)
            backoff = 1
            consecutive_failures = 0
        except Exception as e:
            if 'key' in locals():
                text_pool.finalize(key, False)
            if 'voice_id' in locals() and voice_id and voice_id != "design":
                voice_chooser.finalize(key, voice_id, False)
            _finish_reserved(False)
            consecutive_failures += 1
            transport_error = _is_space_transport_error(e)
            wait = backoff * (2 if transport_error else 3)
            wait = min(wait, 90)
            jitter = random.uniform(0.5, 2.0)
            print(f"\n[w{wid}] error x{consecutive_failures}: {type(e).__name__}: {e}")
            print(f"[w{wid}] backing off {wait + jitter:.1f}s")
            try:
                del client
            except Exception:
                pass
            client = None
            if transport_error or consecutive_failures >= 3:
                try:
                    time.sleep(wait + jitter)
                    client = _new_client()
                    print(f"[w{wid}] ready")
                except Exception as reconnect_error:
                    print(f"[w{wid}] reconnect failed: {reconnect_error}")
            else:
                time.sleep(wait + jitter)
            backoff = min(backoff * 2, 45)

# ── Organize existing files ────────────────────────────────────────────────────
def organize_existing(base_out_dir: Path):
    """Move flat audio files from base dir into audio/ subdirectory with versioning."""
    if not base_out_dir.exists():
        print(f"Output directory not found: {base_out_dir}")
        return

    flat_wavs = sorted(base_out_dir.glob("*_clone_*.wav")) + sorted(base_out_dir.glob("*_ultimate_*.wav")) + sorted(base_out_dir.glob("*_design_*.wav"))

    if not flat_wavs:
        print(f"No audio files to organize in {base_out_dir}")
        return

    # Create versioned folder
    version_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = base_out_dir / version_folder
    audio_dir = version_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Organizing {len(flat_wavs)} audio files into {version_folder}/audio/")
    for wav in flat_wavs:
        dst = audio_dir / wav.name
        shutil.move(str(wav), str(dst))
        print(f"  {wav.name}")

    # Move metadata.csv if it exists
    old_manifest = base_out_dir / "metadata.csv"
    if old_manifest.exists():
        shutil.move(str(old_manifest), str(version_dir / "metadata.csv"))
        print(f"  metadata.csv")

    # Move README.md if it exists
    old_readme = base_out_dir / "README.md"
    if old_readme.exists():
        shutil.move(str(old_readme), str(version_dir / "README.md"))
        print(f"  README.md")

    print(f"\nOK: Organized into {version_dir.resolve()}")

# ── Discover ───────────────────────────────────────────────────────────────────
def discover():
    print(f"Connecting to {SPACE}…")
    Client(SPACE, verbose=False).view_api()

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--workers",  type=int,  default=DEFAULT_WORKERS)
    ap.add_argument("--out",      type=Path, default=DEFAULT_OUT)
    ap.add_argument("--voices",   type=Path, default=None,
                    help="Voice root dir (default: reference_voices/ then fallback ElevenLabs)")
    ap.add_argument("--version",  type=str, default=None,
                    help="Version folder name (default: auto-generated from timestamp)")
    ap.add_argument("--corpus-file", type=Path, default=None,
                    help="Optional CSV/TSV/JSON corpus file. Rows should be text,category or JSON items with text/category.")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Stop automatically after this many total clips in the version folder.")
    ap.add_argument("--organize", action="store_true",
                    help="Reorganize existing files into audio/ subdirectory with versioning")
    args = ap.parse_args()

    if args.discover:
        discover()
        return

    base_out_dir = args.out

    if args.organize:
        organize_existing(base_out_dir)
        return

    # Handle versioning
    if args.version:
        version_folder = args.version
    else:
        version_folder = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Versioned output directory
    out_dir      = base_out_dir / version_folder
    audio_dir    = out_dir / "audio"
    manifest_path = out_dir / "metadata.csv"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Voice pool
    voice_paths = []
    if args.voices:
        voice_paths = [args.voices]
    else:
        if DEFAULT_VOICES.exists():
            voice_paths.append(DEFAULT_VOICES)
        if FALLBACK_VOICES.exists():
            voice_paths.append(FALLBACK_VOICES)

    pool = VoicePool(voice_paths)

    # Resume: count existing
    global _existing, _stop_after
    existing_files = set(audio_dir.glob("*.wav"))
    _existing = len(existing_files)
    _stop_after = args.max_clips

    corpus = _load_corpus(args.corpus_file)
    if not corpus:
        raise SystemExit("Corpus is empty. Provide a valid --corpus-file or keep the built-in corpus.")
    existing_rows = _load_existing_rows(manifest_path)
    text_pool = TextPool(corpus, existing_rows=existing_rows, seed=42)
    voice_chooser = VoiceChooser(pool, existing_rows=existing_rows, seed=42)

    _init_manifest(manifest_path)
    _write_card(out_dir, pool, len(corpus))

    print(f"\nVoxCPM Dataset Generator")
    print(f"  Space:    {SPACE}")
    print(f"  Workers:  {args.workers}")
    print(f"  Output:   {out_dir.resolve()}")
    print(f"  Version:  {version_folder}")
    print(f"  Voices:   {len(pool)} voices — {', '.join(pool.names()) or 'none (design-only)'}")
    print(f"  Corpus:   {len(corpus)} sentences")
    print(f"  Existing: {_existing} clips (resume mode)")
    if args.max_clips:
        print(f"  Max clips:{args.max_clips}")
    print(f"  Modes:    {int(P_ULTIMATE*100)}% ultimate / {int(P_CLONE*100)}% clone / {int(P_DESIGN*100)}% design")
    print(f"  Ctrl+C to stop\n")

    stop   = threading.Event()
    threads = []
    for i in range(args.workers):
        t = threading.Thread(
            target=_worker,
            args=(i, audio_dir, manifest_path, pool, text_pool, voice_chooser, stop),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(1.2)  # stagger starts

    try:
        while not stop.is_set():
            time.sleep(5)
            _print_stats()
    except KeyboardInterrupt:
        print(f"\nStopping…")
        stop.set()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.time() - _start_time
        total   = _existing + _total_gen
        print(f"Done. {_total_gen} new clips  ({total} total)  →  {out_dir.resolve()}")
        print(f"Errors: {_total_err}  |  Time: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
