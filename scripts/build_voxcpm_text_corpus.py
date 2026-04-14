#!/usr/bin/env python3
"""
Build a balanced English text corpus for synthetic TTS generation.

Output CSV columns:
  text,category
"""

import argparse
import csv
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "corpora" / "voxcpm_texts_5000.csv"


TARGET_COUNTS = {
    "short": 500,
    "conversational": 1350,
    "narrative": 900,
    "descriptive": 500,
    "technical": 600,
    "long": 450,
    "emotional": 400,
    "question": 200,
    "dialogue": 100,
}


SHORT_LINES = [
    "Wait, what?",
    "No, that cannot be right.",
    "That is actually wild.",
    "Please tell me you're joking.",
    "I knew this would happen.",
    "That explains a lot.",
    "I need a second.",
    "This is getting ridiculous.",
    "That is not ideal.",
    "I did not expect that.",
    "Now that is interesting.",
    "Absolutely not.",
    "Well, that changed quickly.",
    "Okay, I'm listening.",
    "That was weird.",
    "I can't believe that worked.",
    "That is a problem.",
    "No chance.",
    "There it is.",
    "That was close.",
]

SHORT_SUBJECTS = [
    "this",
    "that",
    "the whole thing",
    "the plan",
    "this outcome",
    "that version",
    "the timing",
    "the situation",
]

SHORT_DESCRIPTORS = [
    "normal",
    "sustainable",
    "a good sign",
    "even slightly ideal",
    "what I expected",
    "a stable solution",
    "remotely convincing",
    "easy to explain",
    "safe",
    "finished",
]

SHORT_EVENTS = [
    "this to fail",
    "that to hold",
    "the timing to get worse",
    "the room to go quiet",
    "the plan to survive",
    "the fallback to last",
    "the noise to be that obvious",
    "the meeting to drag on",
    "the answer to be this simple",
    "the fix to be that small",
]

SHORT_OUTCOMES = [
    "end cleanly",
    "sound that rough",
    "stay stable for long",
    "take this much effort",
    "look convincing",
    "get approved that easily",
    "go wrong that fast",
    "turn into a bigger mess",
]

SHORT_ENDINGS = [
    "right now",
    "today",
    "this time",
    "again",
    "that quickly",
    "this late",
    "at this point",
    "that easily",
    "that cleanly",
    "anymore",
    "so suddenly",
    "that late",
]

PEOPLE = [
    "my brother",
    "my sister",
    "my friend",
    "my neighbor",
    "my coworker",
    "our manager",
    "the teacher",
    "the driver",
    "the doctor",
    "the host",
    "the engineer",
    "the designer",
    "the producer",
    "the student",
    "the coach",
    "the receptionist",
    "the editor",
    "the intern",
]

PLACES = [
    "the coffee shop",
    "the train station",
    "the airport gate",
    "the hotel lobby",
    "the bookstore",
    "the front desk",
    "the conference room",
    "the office kitchen",
    "the studio",
    "the rehearsal room",
    "the classroom",
    "the hallway",
    "the waiting room",
    "the parking garage",
    "the lobby",
]

CONVERSATIONAL_EVENTS = [
    "showed up twenty minutes late",
    "forgot the plan",
    "changed the subject halfway through",
    "made the whole thing more awkward",
    "acted like nothing had happened",
    "asked the most obvious question imaginable",
    "missed the turn",
    "sent the email too early",
    "closed the tab with the final draft still open",
    "brought the wrong charger",
    "laughed at exactly the wrong moment",
    "tried to improvise and made it worse",
    "walked back in like the conversation was still going",
    "stopped mid-sentence and stared at the wall",
    "rewrote the plan while we were still discussing it",
]

FOLLOWUPS = [
    "and somehow made everything worse",
    "and pretended that was normal",
    "and nobody knew how to respond",
    "and then asked us to stay calm",
    "before anyone could explain anything",
    "while the rest of us just stared",
    "and somehow it still worked out",
    "like that was the obvious move",
    "and I still do not know why",
    "which honestly tracks",
]

CONV_OPENERS = [
    "The funniest part is",
    "The strange thing is",
    "What nobody mentions is that",
    "The part that still gets me is that",
    "I should probably admit that",
    "The problem started when",
    "What surprised me most was that",
    "The real turning point was when",
]

CONV_REACTIONS = [
    "nobody was ready for that conversation",
    "I had to try very hard not to laugh",
    "the room went quiet for a full second",
    "everyone suddenly looked at me",
    "the whole meeting lost the plot",
    "it somehow became my problem",
    "I knew the day was about to get longer",
    "we all understood the situation instantly",
]

TIME_MARKERS = [
    "By the time we got there",
    "At some point during the meeting",
    "Right after lunch",
    "Ten minutes into the call",
    "Halfway through the drive",
    "Before the first speaker even finished",
    "As soon as the doors opened",
    "About an hour later",
]

NARRATIVE_SCENES = [
    "the room went completely quiet",
    "everything started to make sense",
    "I realized how far behind we really were",
    "the whole plan suddenly felt fragile",
    "nobody wanted to be the first to react",
    "the mood in the room changed instantly",
    "the easy version of the story disappeared",
    "we all knew the day had shifted",
]

NARRATIVE_REALIZATIONS = [
    "I had misread the situation from the start",
    "the decision had already been made without me",
    "we were arguing about the wrong problem",
    "the hardest part was still ahead of us",
    "everyone else had seen it before I did",
    "the original plan had quietly stopped making sense",
    "we were all waiting for someone else to say it first",
    "the silence was doing most of the talking",
]

DESCRIPTIVE_SUBJECTS = [
    "The hallway",
    "The lobby",
    "The studio",
    "The classroom",
    "The street outside",
    "The waiting room",
    "The office kitchen",
    "The apartment",
    "The station platform",
    "The conference room",
]

DESCRIPTIVE_STATES = [
    "felt brighter than it had all week",
    "looked calm, but nobody inside seemed relaxed",
    "had the kind of quiet that makes you pay attention",
    "felt smaller the longer we stayed there",
    "looked washed in cold afternoon light",
    "was full of noise without feeling alive",
    "felt more temporary than usual",
    "had that strange stillness that usually means something is about to change",
    "looked exactly the same and somehow completely different",
    "felt ordinary in a way that made the moment stranger",
]

DESCRIPTIVE_LIGHTS = [
    "under flat morning light",
    "in the last pale light of the afternoon",
    "under harsh fluorescent bulbs",
    "with rain-muted light coming through the windows",
    "in the blue light just before sunset",
    "under the kind of lighting that makes everyone look tired",
]

DESCRIPTIVE_DETAILS = [
    "every chair felt slightly out of place",
    "the air carried the smell of coffee and wet pavement",
    "nobody seemed willing to speak above a low voice",
    "the walls seemed to hold onto every sound for too long",
    "the whole place felt more tense than busy",
    "it looked staged until someone actually spoke",
    "the room felt warmer than the weather outside",
    "everything inside looked sharper than it should have",
]

WEATHER_LINES = [
    "The rain started slowly and then all at once.",
    "The morning light made the whole block look softer than usual.",
    "The air outside felt colder than it had any right to.",
    "By sunset, the whole street had turned quiet and orange.",
    "It was the kind of afternoon that made everything feel temporary.",
    "The sky looked pale, sharp, and impossibly far away.",
    "The whole neighborhood smelled like cold air and wet pavement.",
    "The light in the windows shifted every few minutes as the clouds moved.",
]

TECH_SUBJECTS = [
    "the deployment",
    "the service",
    "the dataset",
    "the training run",
    "the inference path",
    "the tokenizer",
    "the scheduler",
    "the audio pipeline",
    "the checkpoint loader",
    "the evaluation script",
    "the export step",
    "the batch job",
]

TECH_ISSUES = [
    "fails under load",
    "uses too much memory",
    "hangs during startup",
    "breaks on Windows",
    "produces noisy outputs",
    "drops part of the sentence",
    "mispronounces uncommon names",
    "becomes unstable after long runs",
    "works only with the older config",
    "takes too long to warm up",
    "doesn't recover cleanly after interruption",
    "silently falls back to the wrong path",
]

TECH_FIXES = [
    "if we pin the dependencies",
    "once we reduce the batch size",
    "if we cache the transcripts",
    "after we rewrite the inference wrapper",
    "when the manifest is cleaned up",
    "if we stop mixing incompatible vocoders",
    "once the prompts are normalized",
    "when the reference clips are shorter",
    "after we audit the checkpoint loading",
    "if we switch back to the official recipe",
]

LONG_OPENERS = [
    "I've been thinking about this for a while, and",
    "If I'm being completely honest,",
    "The hardest part to explain is that",
    "What surprised me most was that",
    "Looking back on it now,",
    "By the time we finally understood what was happening,",
    "Nobody really tells you that",
    "The strange thing is that",
]

LONG_BODIES = [
    "the problem was never just the decision itself, but the uncertainty that followed it for weeks afterward",
    "everyone in the room understood what was happening, even though nobody wanted to say it out loud first",
    "the version that finally worked came from a series of small corrections rather than one dramatic breakthrough",
    "I thought getting the opportunity would be the hard part, and then I learned keeping up with it was harder",
    "some of the best outcomes start as deeply inconvenient problems that force you to get more precise",
    "confidence often looks effortless from a distance, but up close it is mostly repetition, correction, and patience",
    "the plan only started making sense after we stopped pretending the earlier version could still work",
    "most of what people call certainty is really just a temporary pause between two better questions",
]

LONG_ENDINGS = [
    "and I think that is why it stayed with me for so long.",
    "which is probably why the whole thing still feels unfinished.",
    "and that lesson ended up mattering more than the original result.",
    "even though it did not feel important at the time.",
    "and I would probably make the same call again.",
    "which is frustrating, but also weirdly reassuring.",
    "and I suspect that is why I still keep coming back to it.",
    "which is not elegant, but it is at least true.",
]

EMOTION_SCENES = [
    "when the call finally came through",
    "when I opened the message",
    "when she said my name",
    "when the room went quiet",
    "when the train pulled away",
    "when we saw the result on screen",
    "when the crowd started cheering",
    "when I realized what had happened",
    "when he finally looked up",
    "when the door opened",
]

EMOTION_REACTIONS = [
    "I almost cried",
    "I started laughing before I could stop myself",
    "I froze for a second",
    "I felt my stomach drop",
    "I could barely get a full sentence out",
    "I finally relaxed",
    "I had to look away for a moment",
    "I did not know what to say",
    "I felt everything hit me at once",
    "I had to steady my breathing",
]

EMOTION_FEELINGS = [
    "relieved",
    "nervous",
    "furious",
    "hopeful",
    "embarrassed",
    "surprised",
    "restless",
    "confident",
    "uncertain",
    "overwhelmed",
]

EMOTION_ALLOWED = {
    "I almost cried": ["relieved", "hopeful", "embarrassed", "overwhelmed"],
    "I started laughing before I could stop myself": ["relieved", "surprised", "restless", "overwhelmed"],
    "I froze for a second": ["nervous", "surprised", "uncertain", "overwhelmed"],
    "I felt my stomach drop": ["nervous", "uncertain", "embarrassed", "overwhelmed"],
    "I could barely get a full sentence out": ["relieved", "nervous", "hopeful", "overwhelmed"],
    "I finally relaxed": ["relieved", "hopeful", "confident"],
    "I had to look away for a moment": ["embarrassed", "furious", "overwhelmed", "relieved"],
    "I did not know what to say": ["surprised", "uncertain", "overwhelmed", "hopeful"],
    "I felt everything hit me at once": ["overwhelmed", "relieved", "furious", "hopeful"],
    "I had to steady my breathing": ["nervous", "relieved", "overwhelmed", "hopeful"],
}

QUESTION_LINES = [
    "Why do people keep repeating the same mistake?",
    "How did that plan become the default?",
    "When did everything get so complicated so quickly?",
    "Why did we stop trusting the original process?",
    "Why did the room get quiet at exactly that moment?",
    "Why did nobody challenge the first assumption?",
    "Why do small problems turn into major setbacks?",
    "Why did everyone act like that was acceptable?",
    "Why does the easiest answer keep sounding convincing?",
    "Why do we wait so long to say the obvious thing?",
    "What happens when a small problem is ignored for too long?",
    "Who decided that this was the stable version?",
    "Do you really think that explanation holds up?",
    "What would change if we started over from the cleaner path?",
    "Have you noticed how quickly uncertainty spreads in a room?",
    "How often does the obvious fix get ignored because it looks too simple?",
]

DIALOGUE_STARTS = [
    "I said,",
    "She looked at me and said,",
    "He paused and asked,",
    "The first thing I heard was,",
    "My manager leaned back and said,",
    "The driver glanced at me and said,",
]

DIALOGUE_QUOTES = [
    "\"This is not the version we agreed on.\"",
    "\"Tell me exactly what changed.\"",
    "\"We can fix this, but not by pretending it's fine.\"",
    "\"If we're doing this, we're doing it properly.\"",
    "\"That explains a lot more than you think.\"",
    "\"I need five minutes and a cleaner answer.\"",
    "\"You already know what the problem is.\"",
    "\"Let's not make this harder than it has to be.\"",
    "\"I want the honest version, not the polished one.\"",
    "\"Give me the short answer first.\"",
]

DIALOGUE_BEATS = [
    "Nobody answered right away.",
    "The room stayed quiet for a second.",
    "I knew exactly what that meant.",
    "Nobody argued with that.",
    "That was the moment the tone changed.",
    "Everyone suddenly looked more awake.",
]


def sentence_case(text: str) -> str:
    text = normalize(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def place_phrase(place: str) -> str:
    if place.startswith(("the airport", "the train station", "the front desk", "the coffee shop", "the bookstore")):
        return f"at {place}"
    return f"in {place}"


def normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def add(items: set[tuple[str, str]], text: str, category: str):
    text = normalize(text)
    if not text:
        return
    if text[-1] == '"' and len(text) >= 2 and text[-2] in ".?!":
        pass
    elif text[-1] not in ".?!":
        text += "."
    items.add((text, category))


def build_short() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for line in SHORT_LINES:
        add(items, line, "short")
    for subject in SHORT_SUBJECTS:
        for descriptor in SHORT_DESCRIPTORS:
            add(items, f"{sentence_case(subject)} is not {descriptor}", "short")
            add(items, f"{sentence_case(subject)} was never going to be {descriptor}", "short")
        for ending in SHORT_ENDINGS:
            add(items, f"{sentence_case(subject)} is not normal {ending}", "short")
            add(items, f"I did not expect {subject} to unravel {ending}", "short")
    for event in SHORT_EVENTS:
        for ending in SHORT_ENDINGS:
            add(items, f"I did not expect {event} {ending}", "short")
    for subject in SHORT_SUBJECTS:
        for outcome in SHORT_OUTCOMES:
            add(items, f"{sentence_case(subject)} was never going to {outcome}", "short")
    return sorted(items)


def build_conversational() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for person in PEOPLE:
        for event in CONVERSATIONAL_EVENTS:
            for followup in FOLLOWUPS:
                add(items, f"{person.capitalize()} {event}, {followup}", "conversational")
            for place in PLACES:
                add(items, f"{person.capitalize()} {event} {place_phrase(place)}, and the whole room felt it", "conversational")
    for opener in CONV_OPENERS:
        for person in PEOPLE:
            for reaction in CONV_REACTIONS:
                add(items, f"{opener} {person} walked in and {reaction}", "conversational")
    for marker in TIME_MARKERS:
        for person in PEOPLE:
            for reaction in CONV_REACTIONS:
                add(items, f"{marker}, {person} said one thing and {reaction}", "conversational")
    return sorted(items)


def build_narrative() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for marker in TIME_MARKERS:
        for scene in NARRATIVE_SCENES:
            for realization in NARRATIVE_REALIZATIONS:
                add(items, f"{marker}, {scene}, and I realized that {realization}", "narrative")
    for person in PEOPLE:
        for place in PLACES:
            for realization in NARRATIVE_REALIZATIONS:
                add(items, f"By the time {person} reached {place}, I realized that {realization}", "narrative")
                add(items, f"{person.capitalize()} left {place} before anyone could admit that {realization}", "narrative")
    return sorted(items)


def build_descriptive() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for subject in DESCRIPTIVE_SUBJECTS:
        for state in DESCRIPTIVE_STATES:
            add(items, f"{subject} {state}", "descriptive")
        for light in DESCRIPTIVE_LIGHTS:
            for detail in DESCRIPTIVE_DETAILS:
                add(items, f"{subject} looked different {light}, and {detail}", "descriptive")
    for line in WEATHER_LINES:
        add(items, line, "descriptive")
    for place in PLACES:
        for state in DESCRIPTIVE_STATES:
            add(items, f"{place.capitalize()} {state}", "descriptive")
        for light in DESCRIPTIVE_LIGHTS:
            for detail in DESCRIPTIVE_DETAILS:
                add(items, f"{sentence_case(place_phrase(place))}, everything felt slightly altered {light}, and {detail}", "descriptive")
    return sorted(items)


def build_technical() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for subject in TECH_SUBJECTS:
        for issue in TECH_ISSUES:
            add(items, f"{subject.capitalize()} {issue}", "technical")
            for fix in TECH_FIXES:
                add(items, f"{subject.capitalize()} {issue} {fix}", "technical")
                add(items, f"We can usually stabilize {subject} when it {issue} {fix}", "technical")
    return sorted(items)


def build_long() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for opener in LONG_OPENERS:
        for body in LONG_BODIES:
            for ending in LONG_ENDINGS:
                add(items, f"{opener} {body}, {ending}", "long")
                add(items, f"{opener} {body}, and I only really understood that later, {ending}", "long")
    return sorted(items)


def build_emotional() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for scene in EMOTION_SCENES:
        for reaction in EMOTION_REACTIONS:
            add(items, f"{sentence_case(scene)}, {reaction}", "emotional")
            for feeling in EMOTION_ALLOWED.get(reaction, EMOTION_FEELINGS):
                add(
                    items,
                    f"{sentence_case(scene)}, {reaction}, and for a moment I felt completely {feeling}",
                    "emotional",
                )
    return sorted(items)


def build_question() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    why_do = [
        "people keep repeating the same mistake",
        "we wait so long to say the obvious thing",
        "small problems turn into major setbacks",
        "teams keep trusting the unstable version",
        "the easiest answer keep sounding convincing",
        "we keep accepting sloppy fixes under pressure",
        "the same bottlenecks show up every single week",
        "people confuse motion with actual progress",
        "we keep delaying the clean rewrite",
        "smart people keep overcomplicating simple fixes",
    ]
    why_did = [
        "that plan become the default",
        "the room get quiet at exactly that moment",
        "nobody challenge the first assumption",
        "everyone act like that was acceptable",
        "the fallback path become the real production path",
        "the cleaner option get ignored",
        "the timeline slip that quickly",
        "we let the old version survive for so long",
        "the handoff break down there",
        "the simple explanation stop sounding sufficient",
    ]
    how_did = [
        "that issue make it all the way to production",
        "the original plan drift this far off course",
        "the meeting turn into a two-hour argument",
        "the noisy version become the accepted baseline",
        "that workaround become permanent",
        "the easiest decision become the hardest one to defend",
        "we miss the cleaner path the first time",
        "the schedule get that crowded",
        "the whole discussion become about the wrong thing",
        "that result still surprise us",
    ]
    what_happens = [
        "a small problem gets ignored for too long",
        "the fallback becomes the main path",
        "nobody owns the final decision",
        "the prompt text is slightly wrong",
        "the reference audio is clean but the alignment is off",
        "a system keeps scaling past the point it was designed for",
        "everyone agrees too quickly",
        "the shortest answer is the correct one",
        "the model sounds better but becomes less stable",
        "a fix lands without a proper evaluation pass",
    ]
    do_you = [
        "that explanation actually holds up",
        "the current baseline is good enough to keep",
        "we would make the same decision again tomorrow",
        "the tradeoff is really worth it",
        "that version is stable enough for release",
        "the rushed option ever saves time in the end",
        "the old workflow can still be defended",
        "the easier path is also the cleaner one",
        "the current failure rate is acceptable",
        "this plan survives real usage without major changes",
    ]
    who = [
        "decided that this was the stable version",
        "approved that fallback as the final answer",
        "signed off on the noisier output",
        "thought this handoff was actually finished",
        "kept the older path alive for this long",
        "benefits when the messy version stays in place",
        "is still defending that original estimate",
        "is actually responsible for the final call",
        "thought this was ready without more evaluation",
        "was supposed to catch that mismatch earlier",
    ]

    suffixes = [
        "",
        " so often",
        " in practice",
        " under pressure",
    ]

    for clause in why_do:
        for suffix in suffixes:
            add(items, f"Why do {clause}{suffix}?", "question")
    for clause in why_did:
        for suffix in ["", " in the first place", " that quickly", " so quietly"]:
            add(items, f"Why did {clause}{suffix}?", "question")
    for clause in how_did:
        for suffix in ["", " this quickly", " without anyone stopping it", " in the first place"]:
            add(items, f"How did {clause}{suffix}?", "question")
    for clause in what_happens:
        for suffix in ["", " in production", " under load", " over time"]:
            add(items, f"What happens when {clause}{suffix}?", "question")
    for clause in do_you:
        for suffix in ["", " anymore", " after seeing the evidence", " under real-world usage"]:
            add(items, f"Do you really think {clause}{suffix}?", "question")
    for clause in who:
        for suffix in ["", " this time", " at the beginning", " when it first slipped"]:
            add(items, f"Who {clause}{suffix}?", "question")
    return sorted(items)


def build_dialogue() -> list[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for start in DIALOGUE_STARTS:
        for quote in DIALOGUE_QUOTES:
            add(items, f"{start} {quote}", "dialogue")
            for beat in DIALOGUE_BEATS:
                add(items, f"{start} {quote} {beat}", "dialogue")
    return sorted(items)


def build_candidates() -> dict[str, list[tuple[str, str]]]:
    return {
        "short": build_short(),
        "conversational": build_conversational(),
        "narrative": build_narrative(),
        "descriptive": build_descriptive(),
        "technical": build_technical(),
        "long": build_long(),
        "emotional": build_emotional(),
        "question": build_question(),
        "dialogue": build_dialogue(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    total_target = sum(TARGET_COUNTS.values())
    if args.count != total_target:
        raise SystemExit(f"This builder is tuned for {total_target} texts, but --count was {args.count}.")

    by_cat = build_candidates()
    rng = random.Random(args.seed)
    selected: list[tuple[str, str]] = []

    for category, needed in TARGET_COUNTS.items():
        pool = by_cat.get(category, [])
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
    for category in TARGET_COUNTS:
        print(f"{category:16s} {TARGET_COUNTS[category]:4d} / {len(by_cat[category]):5d} candidates")


if __name__ == "__main__":
    main()
