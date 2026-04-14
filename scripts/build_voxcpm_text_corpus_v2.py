#!/usr/bin/env python3
"""
Build a higher-quality English text corpus for VoxCPM2 synthetic cloning data.

Goals:
- English only
- strong conversational / narrative / expressive coverage
- enough text diversity for a 20k or 30k text pool
- cleaner than the old heavily self-referential technical corpus

Output CSV columns:
  text,category
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "corpora" / "voxcpm_texts_20000_v2.csv"


TARGETS = {
    20000: {
        "short": 1700,
        "conversational": 5750,
        "narrative": 4000,
        "descriptive": 1700,
        "instructional": 900,
        "long": 2500,
        "emotional": 1800,
        "question": 500,
        "dialogue": 600,
        "punctuation": 550,
    },
}


NAMES = [
    "Amelia", "Marcus", "Priya", "Jordan", "Elena", "Theo", "Sofia", "Noah",
    "Maya", "Lucas", "Ava", "Ethan", "Clara", "Julian", "Nina", "Leo",
    "Hazel", "Miles", "Vivian", "Daniel", "Iris", "Nathan", "Layla", "Owen",
]

RELATIONS = [
    "my friend", "my brother", "my sister", "my cousin", "my manager", "my neighbor",
    "our teacher", "the driver", "the barista", "the producer", "the editor", "the nurse",
    "the receptionist", "the coach", "the host", "the designer", "the engineer", "the lawyer",
]

PLACES = [
    "the coffee shop", "the train station", "the airport gate", "the hotel lobby", "the bookstore",
    "the hallway", "the studio", "the classroom", "the office kitchen", "the waiting room",
    "the parking garage", "the rooftop", "the rehearsal room", "the front desk", "the conference room",
    "the grocery store", "the pharmacy", "the apartment lobby", "the bus stop", "the hospital corridor",
]

TIME_MARKERS = [
    "Right after lunch", "At seven thirty this morning", "Ten minutes into the meeting",
    "Halfway through the drive", "Just before midnight", "A few minutes after sunrise",
    "By the end of the afternoon", "Early on Saturday morning", "Around nine fifteen",
    "As soon as the doors opened", "Before the first speaker finished", "Later that evening",
]

EVERYDAY_EVENTS = [
    "showed up twenty minutes late", "missed the last turn", "spilled coffee on the notes",
    "forgot the charger again", "closed the tab with the final draft open", "laughed at exactly the wrong moment",
    "rewrote the whole plan on the way there", "walked in like nothing had happened", "sent the message too early",
    "asked the most obvious question in the room", "kept talking long after the point was clear",
    "changed the subject halfway through dinner", "left the receipt on the counter", "lost the key card twice",
    "read the wrong date out loud", "mixed up the names on the list", "forgot where they parked",
    "tried to improvise and made it worse", "brought the wrong bag", "looked at me like I already knew",
]

REACTIONS = [
    "and nobody knew how to respond", "and somehow made everything worse", "and the whole room felt it",
    "and then acted like it was normal", "which honestly tracked", "before anyone could explain anything",
    "and I still do not know why", "and somehow it still worked out", "and that was the turning point",
    "which should have been our first warning", "and then the conversation went silent",
]

NARRATIVE_REALIZATIONS = [
    "I realized we had been solving the wrong problem", "nobody wanted to say what was obvious",
    "the easy version of the story disappeared", "the silence was doing most of the talking",
    "everyone else had seen it before I did", "the original plan had quietly stopped making sense",
    "the whole room changed in a second", "we were all waiting for someone else to say it first",
    "the decision had already been made", "the hardest part was still ahead of us",
]

DESCRIPTIVE_IMAGES = [
    "felt washed in cold afternoon light", "smelled faintly of rain and old wood",
    "looked brighter than it had all week", "had the kind of quiet that makes you pay attention",
    "felt smaller the longer we stayed there", "looked sharp and colorless under the ceiling lights",
    "carried every footstep a little too far", "felt warmer near the windows and colder by the door",
    "had a stillness that did not feel peaceful", "looked ordinary until you stood there long enough",
]

EMOTIONS = [
    "relieved", "overwhelmed", "furious", "embarrassed", "hopeful", "uneasy", "grateful",
    "shaken", "excited", "heartbroken", "stunned", "proud", "nervous", "frustrated",
]

EMOTIONAL_EVENTS = [
    "when they said my name", "when the call finally went through", "when the message appeared",
    "when the crowd started cheering", "when the room went quiet", "when the door opened",
    "when I heard the recording back", "when I realized what it meant", "when the answer finally came",
    "when I saw the finished version", "when the car pulled away", "when the deadline actually passed",
]

QUESTIONS = [
    "What are we actually trying to protect here",
    "Why does that explanation sound cleaner than it is",
    "How did we let something this small turn into such a mess",
    "Who decided this was ready without another pass",
    "When exactly did the easy option become the only option",
    "Do you really think the current version survives real usage",
    "What would you change first if you only had one day",
    "How much of that story do you genuinely believe",
    "Why does asking for help still feel harder than it should",
    "What happens if the quiet part is the important part",
]

INSTRUCTION_OPENERS = [
    "Before you ship anything", "The first step is simple", "If the result looks wrong",
    "When the estimate changes late", "For a cleaner handoff", "If the recording sounds brittle",
    "To keep the output stable", "When the numbers stop lining up", "If a voice drifts off target",
    "Before you trust the metric",
]

INSTRUCTION_ACTIONS = [
    "check the source files again", "compare the previous run instead of guessing",
    "write down the exact settings you used", "test the smallest change first",
    "listen to the raw output before touching post-processing", "keep the reference clips short and clean",
    "verify the transcript matches the reference exactly", "save the failed sample instead of discarding it",
    "separate quality problems from timing problems", "reproduce the issue on one prompt before scaling up",
]

INSTRUCTION_ENDINGS = [
    "before you change anything else", "so the next run is easier to trust",
    "instead of treating it like a mystery", "because one noisy sample can waste a full evening",
    "and then decide whether it is actually worth keeping", "before the comparison gets messy",
]

LONG_REFLECTIONS = [
    "The hardest part was not making the decision itself, but living with the uncertainty afterward and trying not to rewrite the story in my head every few hours.",
    "I used to think the goal was to reach a version of life where nothing felt unstable anymore, and now I think the real skill is staying useful while things are still changing.",
    "There is an awkward gap between knowing something is not good enough and being able to explain exactly why, and most of the real work seems to happen inside that gap.",
    "The strange thing about getting what you wanted is that the moment rarely feels dramatic in real time; it usually feels quiet, delayed, and a little harder to trust than you expected.",
    "We spent so much time trying to avoid the wrong failure that we almost missed the slower problem forming right in front of us, which in hindsight should have been easier to see.",
    "I keep coming back to the same conclusion: most frustrating situations do not collapse because one part is impossible, but because too many small compromises pile up without anyone stopping to name them.",
]

PUNCT_PREFIXES = [
    "For the record:", "Honestly,", "Look,", "At 7:45 a.m.,", "By 11:30 p.m.,", "In one sentence:",
    "The short version is this:", "No, really:", "Three things mattered most:", "Here is the problem:",
]

PUNCT_CLAUSES = [
    "the price jumped from $48.20 to $63.75, and nobody noticed until checkout.",
    "she said, \"Give me ten minutes,\" and then disappeared for nearly an hour.",
    "we had one clean take, two acceptable backups, and exactly zero extra time.",
    "the note in the margin read, \"Do not send yet,\" which would have been useful sooner.",
    "the answer was yes; the tone, however, was not exactly encouraging.",
    "I wrote down the list - batteries, cables, adapters, receipts - and still forgot one thing.",
    "the train left at 6:12, the text arrived at 6:14, and the apology came much later.",
    "the room was quiet, not calm; there is a difference, and everyone could feel it.",
    "he kept saying, \"Almost done,\" as if those words changed anything.",
    "the estimate was precise to the cent, which made the guesswork feel even stranger.",
]

SHORT_LINES = [
    "Wait, what happened?", "No, that is not right.", "That sounded better.", "I need a second.",
    "That was too close.", "Absolutely not.", "That explains a lot.", "I knew it.", "Not ideal.",
    "Now that is interesting.", "That was weird.", "I am listening.", "That is the problem.",
    "We are late again.", "No chance.", "That felt different.", "I did not expect that.",
]


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    text = text.replace(" ;", ";").replace(" :", ":")
    return text


def add(items: set[tuple[str, str]], text: str, category: str) -> None:
    text = normalize(text)
    if not text:
        return
    if len(text) < 6 or len(text) > 260:
        return
    if text.count("  "):
        return
    items.add((text, category))


def build_short() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for line in SHORT_LINES:
        add(items, line, "short")
    subjects = [
        "the timing", "the answer", "this plan", "that version", "the room", "the signal",
        "the estimate", "the recording", "the mood", "the result", "the reply",
        "the backup", "the schedule", "the second take", "the invoice", "the queue",
    ]
    endings = ["today", "again", "this time", "at all", "right now", "for long", "so quickly", "under pressure", "after that", "in public"]
    patterns = [
        "I did not expect {subject} to shift {ending}.",
        "There is no way {subject} stays stable {ending}.",
        "{subject_cap} feels different {ending}.",
        "We cannot keep {subject} like this {ending}.",
        "{subject_cap} is not holding together {ending}.",
        "Something about {subject} changed {ending}.",
    ]
    for subject in subjects:
        for ending in endings:
            for pattern in patterns:
                add(items, pattern.format(subject=subject, subject_cap=subject.capitalize(), ending=ending), "short")
    descriptors = [
        "stable", "clean", "clear", "convincing", "usable", "safe", "finished", "coherent", "quiet", "balanced",
    ]
    for subject in subjects:
        for desc in descriptors:
            add(items, f"{subject.capitalize()} is not {desc} enough.", "short")
            add(items, f"I thought {subject} would stay {desc}.", "short")
    verbs = ["shifted", "broke", "held", "slipped", "stalled", "landed", "drifted", "tightened"]
    qualifiers = ["too fast", "too early", "for no reason", "all at once", "in public", "under pressure", "again"]
    for subject in subjects:
        for verb in verbs:
            for qualifier in qualifiers:
                add(items, f"{subject.capitalize()} {verb} {qualifier}.", "short")
    return sorted(items)


def build_conversational() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    openers = [
        "The strange part is", "The funny part is", "The part that still gets me is",
        "What surprised me most is", "I should probably admit that", "The real problem started when",
        "The reason I remember it so clearly is that", "The honest answer is that",
    ]
    for opener in openers:
        for person in RELATIONS + NAMES:
            for event in EVERYDAY_EVENTS:
                for reaction in REACTIONS:
                    add(items, f"{opener} {person} {event}, {reaction}", "conversational")
    for place in PLACES:
        for person in RELATIONS:
            for event in EVERYDAY_EVENTS[:12]:
                add(items, f"At {place}, {person} {event}, and I knew the day had changed.", "conversational")
    return sorted(items)


def build_narrative() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for marker in TIME_MARKERS:
        for person in RELATIONS + NAMES:
            for place in PLACES:
                for realization in NARRATIVE_REALIZATIONS:
                    add(items, f"{marker}, {person} reached {place}, and {realization}.", "narrative")
                    add(items, f"{marker}, when {person} stepped into {place}, {realization}.", "narrative")
    return sorted(items)


def build_descriptive() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    modifiers = [
        "in the blue light before sunset", "under flat fluorescent light", "after the rain stopped",
        "in that stale mid-afternoon quiet", "while the air still smelled like coffee",
        "with the windows cracked open", "while everyone pretended to stay busy",
        "while the heater hummed in the corner", "with traffic moving softly outside",
        "while the last train announcement echoed through the hall",
    ]
    for place in PLACES:
        for image in DESCRIPTIVE_IMAGES:
            add(items, f"{place.capitalize()} {image}.", "descriptive")
            for mod in modifiers:
                add(items, f"{place.capitalize()} {image} {mod}.", "descriptive")
                add(items, f"In {place}, everything {image} {mod}.", "descriptive")
    return sorted(items)


def build_instructional() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for opener in INSTRUCTION_OPENERS:
        for action in INSTRUCTION_ACTIONS:
            for ending in INSTRUCTION_ENDINGS:
                add(items, f"{opener}, {action} {ending}.", "instructional")
    specifics = [
        "The invoice closed at $247.18 after tax.",
        "The train to Seattle leaves from track twelve at 8:05.",
        "The backup copy should finish in about fourteen minutes.",
        "The meeting was moved from Tuesday to Thursday at 3:30.",
        "The replacement cable is exactly one meter longer than the original.",
        "The final draft has twenty-four pages and three unresolved comments.",
    ]
    for line in specifics:
        add(items, line, "instructional")
    checks = [
        "timing", "transcript alignment", "file names", "speaker labels", "the last checkpoint",
        "the raw waveform", "the final export", "the reference clip", "the actual sample rate", "the comparison notes",
    ]
    reasons = [
        "before you call it done", "before you scale the run up", "before you trust the result",
        "before you overwrite the previous version", "before you make another change",
    ]
    for check in checks:
        for reason in reasons:
            add(items, f"Double-check {check} {reason}.", "instructional")
            add(items, f"Write down {check} {reason}.", "instructional")
    symptoms = [
        "sounds too bright", "comes out too fast", "drifts off target", "cuts off early",
        "feels unstable", "gets muddy in the middle", "stops matching the reference",
        "loses the punctuation", "collapses on long prompts", "starts clean and ends rough",
    ]
    fixes = [
        "reduce the moving parts first", "listen to one sample end to end",
        "save the exact reference that caused it", "compare against the last known good run",
        "separate the timing issue from the timbre issue", "check whether the transcript is exact",
    ]
    for symptom in symptoms:
        for fix in fixes:
            add(items, f"If the output {symptom}, {fix}.", "instructional")
    tasks = [
        "the sample rate", "the transcript", "the timing map", "the holdout list", "the warmup clip",
        "the loudness target", "the exported manifest", "the reference pair", "the validation notes", "the fallback path",
    ]
    outcomes = [
        "looks wrong", "drifts out of sync", "stops matching the speaker", "comes back empty",
        "fails twice in a row", "sounds too sharp", "sounds too dull", "moves too fast",
        "breaks on one specific prompt", "changes after a restart",
    ]
    for task in tasks:
        for outcome in outcomes:
            add(items, f"If {task} {outcome}, compare it with the last stable run before changing anything else.", "instructional")
            add(items, f"When {task} {outcome}, save the exact example first and isolate the smallest reproducible case.", "instructional")
    return sorted(items)


def build_long() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for line in LONG_REFLECTIONS:
        add(items, line, "long")
    clauses_a = [
        "what bothered me most was not the mistake itself",
        "the difficult part was never the public version of the story",
        "for a while I thought the problem was timing",
        "it took me longer than I want to admit to understand",
        "the lesson that stayed with me had less to do with success than with attention",
    ]
    clauses_b = [
        "but the pattern underneath it",
        "but the way everybody adjusted around it",
        "and that changed the way I heard the whole conversation",
        "because the warning signs were there long before the failure was obvious",
        "which is probably why the result still feels unfinished even now",
    ]
    clauses_c = [
        "and I think that is why the memory still feels so sharp.",
        "and I do not think I understood that at the time.",
        "and that is the part I keep returning to when I replay it.",
        "which makes the final outcome feel almost secondary.",
        "and that small shift changed the rest of the week.",
    ]
    for a in clauses_a:
        for b in clauses_b:
            for c in clauses_c:
                add(items, f"{a}, {b}, {c}", "long")
    clause_d = [
        "even though the visible problem looked simple from the outside",
        "because everyone in the room was reacting to a slightly different version of events",
        "which made the next decision feel heavier than it should have",
        "and by then the easiest answer had already stopped being the honest one",
        "while the rest of us were still trying to name what had changed",
    ]
    clause_e = [
        "That is the version I trust now.",
        "I wish I had understood that sooner.",
        "It took me longer than I want to admit to see it clearly.",
        "That is probably why the result still feels unfinished.",
        "I think that is why it still sounds different in my head.",
    ]
    for a in clauses_a:
        for b in clauses_b:
            for d in clause_d:
                for e in clause_e:
                    add(items, f"{a}, {b}, {d}, and {e}", "long")
    openers = [
        "What still bothers me about that day is",
        "The part I keep replaying is",
        "I did not understand until much later that",
        "If I am honest about it now,",
        "Looking back, I think",
        "The most useful lesson from that whole mess is that",
        "I thought the hard part was over, but",
        "For a while I blamed the wrong thing because",
        "What made the moment heavier than it looked was that",
        "The story sounds simpler from the outside, but",
    ]
    middles = [
        "everyone was responding to a slightly different version of the same event",
        "the clean answer arrived a little too late to be easy",
        "nobody wanted to say the obvious part out loud",
        "the smaller compromise had been accumulating for weeks",
        "the tone of the room changed before the facts did",
        "the first explanation felt tidy only because it ignored the messy part",
        "the person with the clearest view said the least",
        "what looked like confidence was mostly momentum",
        "the silence carried more information than the conversation",
        "the result started drifting before anyone admitted it",
    ]
    transitions = [
        "and that is why the ending still feels stranger than the beginning",
        "which is probably why I still do not trust the easy version",
        "and once I saw that, the rest of the story changed shape",
        "which made the final outcome feel less surprising and more inevitable",
        "and that was the first moment the whole pattern became visible",
    ]
    closers = [
        "even if I could not name it at the time.",
        "and I think that is the detail that stayed with me.",
        "which is the part I wish I had noticed sooner.",
        "and that quiet realization mattered more than the loud conclusion.",
        "which is why I still hear the whole thing differently now.",
    ]
    for opener in openers:
        for middle in middles:
            for transition in transitions:
                for closer in closers:
                    add(items, f"{opener} {middle}, {transition} {closer}", "long")
    return sorted(items)


def build_emotional() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for emotion in EMOTIONS:
        for event in EMOTIONAL_EVENTS:
            add(items, f"When {event}, I felt completely {emotion}.", "emotional")
            add(items, f"I did not expect to feel so {emotion} {event}.", "emotional")
    endings = [
        "and I could hear it in my own voice",
        "and I had to stop for a second",
        "which caught me off guard",
        "and that feeling stayed with me all night",
        "even though I thought I was prepared",
    ]
    for emotion in EMOTIONS:
        for ending in endings:
            add(items, f"I was {emotion}, {ending}.", "emotional")
    people = ["my mother", "my father", "my friend", "the host", "the doctor", "the teacher", "my brother", "my sister"]
    statements = [
        "said they were proud of me", "looked me in the eye and told me the truth",
        "finally answered the question", "laughed before I could stop myself",
        "admitted they had been scared too", "stood there without saying anything",
    ]
    for person in people:
        for statement in statements:
            for emotion in EMOTIONS:
                add(items, f"When {person} {statement}, I felt unexpectedly {emotion}.", "emotional")
    places = ["in the hallway", "at the station", "outside the theater", "in the car", "by the front desk", "under the kitchen light"]
    for place in places:
        for event in EMOTIONAL_EVENTS:
            for emotion in EMOTIONS:
                add(items, f"{place.capitalize()}, {event}, and I felt more {emotion} than I wanted to admit.", "emotional")
    return sorted(items)


def build_question() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    suffixes = [
        "", " after seeing the evidence", " when the pressure gets real", " if the next run sounds cleaner",
        " once the room goes quiet", " if the easy answer stops working",
    ]
    for q in QUESTIONS:
        for suffix in suffixes:
            add(items, q + suffix + "?", "question")
    starters = ["Why did", "How did", "What happens when", "Who decided that", "When did"]
    middles = [
        "the clean version stop sounding honest", "the schedule get this fragile",
        "the quiet answer become the loudest problem", "we start trusting the wrong signal",
        "the safer option become harder to defend", "everyone agree before the details were clear",
        "the recording stop feeling like the same person", "the simplest fix start costing the most time",
    ]
    endings = ["", " in public", " under pressure", " this late in the process", " after all that work"]
    for starter in starters:
        for middle in middles:
            for ending in endings:
                add(items, f"{starter} {middle}{ending}?", "question")
    subjects = [
        "the tone", "the timing", "the pacing", "the reference", "the silence", "the estimate",
        "the final answer", "the clean version", "the easy explanation", "the first draft",
    ]
    predicates = [
        "start sounding wrong", "feel better than it really was", "stop matching the room",
        "turn into the whole problem", "get harder to defend", "become impossible to ignore",
        "sound cleaner but less true", "survive real usage", "drift away from the reference",
        "change the decision",
    ]
    for subject in subjects:
        for predicate in predicates:
            add(items, f"Why did {subject} {predicate}?", "question")
            add(items, f"How did {subject} {predicate} so quickly?", "question")
            add(items, f"What happens when {subject} {predicate} under pressure?", "question")
            add(items, f"Do you really think {subject} can {predicate} and still sound natural?", "question")
    return sorted(items)


def build_dialogue() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    tags = ["she said", "he said", "I asked", "they replied", "the host whispered", "my manager said"]
    quotes_a = [
        "We are not doing this twice", "Give me ten minutes", "Tell me the honest version",
        "That is not what you promised", "I need the short answer", "You already know the problem",
        "Say it clearly", "This is the last clean chance we get", "Do not send that yet",
    ]
    beats = [
        "and nobody argued", "and the room went quiet", "but nobody moved",
        "while the lights hummed above us", "and I believed them for exactly one second",
        "before the phone rang again", "and that was somehow worse",
    ]
    replies = [
        "\"I know,\" I said.", "\"That is the problem,\" she replied.", "\"Then fix it,\" he said softly.",
        "\"I am trying,\" I told him.", "\"We are out of time,\" they said.",
    ]
    for quote in quotes_a:
        for tag in tags:
            add(items, f"\"{quote},\" {tag}.", "dialogue")
            for beat in beats:
                add(items, f"\"{quote},\" {tag}, {beat}.", "dialogue")
            for reply in replies:
                add(items, f"\"{quote},\" {tag}. {reply}", "dialogue")
    return sorted(items)


def build_punctuation() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for prefix in PUNCT_PREFIXES:
        for clause in PUNCT_CLAUSES:
            add(items, f"{prefix} {clause}", "punctuation")
    extras = [
        "At 4:17 p.m., the total changed from $18.40 to $21.05.",
        "She asked for three things: clarity, timing, and proof.",
        "Look, if the first take sounds honest, keep it.",
        "I had one job - literally one - and still managed to miss it.",
        "No, really, write it down: name, date, time, location.",
        "The note said, \"Try again tomorrow,\" which was not exactly comforting.",
    ]
    for line in extras:
        add(items, line, "punctuation")
    dates = ["April 3rd", "June 14th", "September 21st", "Monday morning", "Friday night"]
    amounts = ["$18.40", "$247.18", "$1,204.00", "$63.75", "$9.95"]
    for date in dates:
        for amount in amounts:
            add(items, f"On {date}, the estimate moved from {amount} to {amount}; nobody liked the explanation.", "punctuation")
            add(items, f"Here is the exact problem: on {date}, the total changed to {amount}, and the note still said \"close enough.\"", "punctuation")
    fragments = [
        "one clean take", "two rough backups", "three unanswered messages", "four missed calls",
        "five minutes of silence", "six tiny corrections", "seven lines of notes",
    ]
    endings = [
        "and none of them solved the real issue.", "which should have made the answer obvious.",
        "but the room still felt uncertain.", "and somehow the simplest option still won.",
        "and that was before the awkward part started.",
    ]
    for prefix in PUNCT_PREFIXES:
        for fragment in fragments:
            for ending in endings:
                add(items, f"{prefix} {fragment}; {ending}", "punctuation")
    times = ["6:12", "7:45", "9:30", "11:05", "2:17"]
    notes = [
        "\"send the clean version first\"",
        "\"do not touch the timing\"",
        "\"check the transcript twice\"",
        "\"keep the reference exactly as is\"",
        "\"compare before you replace\"",
    ]
    for time_value in times:
        for note in notes:
            add(items, f"At {time_value}, the note read {note}; by noon, nobody agreed on what it meant.", "punctuation")
            add(items, f"Here is what changed at {time_value}: the room got quieter, the answer got shorter, and the note still read {note}.", "punctuation")
    return sorted(items)


def build_candidates() -> dict[str, list[tuple[str, str]]]:
    return {
        "short": build_short(),
        "conversational": build_conversational(),
        "narrative": build_narrative(),
        "descriptive": build_descriptive(),
        "instructional": build_instructional(),
        "long": build_long(),
        "emotional": build_emotional(),
        "question": build_question(),
        "dialogue": build_dialogue(),
        "punctuation": build_punctuation(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20000, choices=sorted(TARGETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    targets = TARGETS[args.count]
    by_cat = build_candidates()
    rng = random.Random(args.seed)
    selected: list[tuple[str, str]] = []

    for category, needed in targets.items():
        pool = by_cat[category]
        if len(pool) < needed:
            raise SystemExit(f"Category {category} only has {len(pool)} candidates, needs {needed}.")
        selected.extend(rng.sample(pool, needed))

    rng.shuffle(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(selected)

    print(f"Wrote {len(selected)} texts to {args.out}")
    for category, needed in targets.items():
        print(f"{category:16s} {needed:5d} / {len(by_cat[category]):5d} candidates")


if __name__ == "__main__":
    main()
