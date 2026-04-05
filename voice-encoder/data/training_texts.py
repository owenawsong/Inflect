"""
Paralinguistic training text corpus for Inflect Phase 3.

Covers: laughs, sighs, whispers, crying, gasps, coughs, emotional modifiers,
        fillers, and delivery tags. Each line has BOTH speech AND at least one tag.
Tags are ElevenLabs v3 format.

~100 texts across diverse contexts, tones, speakers, and registers.
Designed to produce maximally varied audio when synthesized across many voices.
"""

TEXTS = [

    # ── LAUGHS (10) — most voice-dependent, keep variety ─────────────────────

    "I can't believe you actually said that. [laughs] That's the funniest thing I've heard all week.",
    "Wait, you tripped over your own feet? [laughs] In front of your boss? Oh no.",
    "He thought the meeting was tomorrow. [chuckles] It was very much today.",
    "So you texted your professor instead of your friend. [laughs harder] What did it say?",
    "The cat just sat there judging me the entire time. [giggles] I felt so seen.",
    "He walked face-first into the glass door. [snorts] I tried not to laugh. I failed.",
    "He called his teacher 'mom' in front of the whole class. [wheezing] He never recovered.",
    "So I'm standing there, fully confident, giving the wrong presentation. [starts laughing] Complete silence.",
    "I told myself I'd sleep early. [laughs] It is currently three in the morning.",
    "She put the cereal in the fridge and the milk in the cabinet. [laughing] And then wondered why it tasted wrong.",

    # ── SIGHS / EXHALES (5) ───────────────────────────────────────────────────

    "I don't know. [sighs] It's just been one of those weeks, you know?",
    "I applied to fourteen jobs this month. [sighs] Two responses. Both rejections.",
    "He said he'd call. [exhales sharply] That was three weeks ago.",
    "I keep trying to explain it but nobody's listening. [frustrated sigh] It's exhausting.",
    "I thought things would be different by now. [exhales] I was wrong.",

    # ── WHISPERS (5) ──────────────────────────────────────────────────────────

    "[whispers] Don't look now but I think that's him over there.",
    "The baby finally went to sleep. [whispers] Don't. Move. A. Muscle.",
    "[whispers] She still doesn't know it was me. And I'd like to keep it that way.",
    "Okay, everyone act normal. [whispers] He's literally right behind you.",
    "[whispers] I've been faking that I understood this the entire time.",

    # ── CRYING (6) ────────────────────────────────────────────────────────────

    "I'm fine. [crying] I promise I'm completely fine.",
    "She just— [crying] sorry. Give me a second.",
    "We had him for twelve years. [crying] He was the best dog.",
    "I got the job. [crying] I can't believe I actually got the job.",
    "He said he was proud of me. [crying] Nobody's ever said that to me before.",
    "I thought I was over it. [crying] Clearly not.",

    # ── GASPS (4) ─────────────────────────────────────────────────────────────

    "[gasps] You got engaged? When? How? Tell me everything right now.",
    "[happy gasp] Oh my god. This is the best thing I've ever tasted.",
    "[inhales deeply] Okay. I'm calm. I'm completely calm. I can handle this.",
    "No way. [gasps] No absolute way. How did you keep this secret for so long?",

    # ── EMOTIONAL MODIFIERS (10) ──────────────────────────────────────────────

    "[excited] Okay so I have the most insane idea and I need you to just hear me out.",
    "[sad] I used to love this place. It's not the same anymore.",
    "[angry] I specifically asked for one thing. One thing. And it still didn't happen.",
    "[curious] Have you ever thought about why we do things the way we do?",
    "[thoughtful] I think the problem is that we're solving the wrong question entirely.",
    "[surprised] I didn't even know you two knew each other.",
    "[sarcastic] Oh yes, because that plan has worked out so perfectly every other time.",
    "[impressed] I've seen a lot of presentations. That was genuinely one of the best.",
    "[warmly] You know, I think about that conversation more than you'd expect.",
    "[dramatically] I have fought armies. I have crossed oceans. I did not come this far to lose.",

    # ── CLEARS THROAT / MUTTERING (4) ────────────────────────────────────────

    "[clears throat] Right. So. Moving on from that particular disaster.",
    "Sorry, give me a second. [clears throat] Okay. Where were we.",
    "[muttering] Okay, okay. Where did I put that. I just had it.",
    "So the thing is— [muttering] no that's not right either. Hang on.",

    # ── MIXED / COMPLEX (14) — multiple tags, most training signal ────────────

    "I didn't sleep last night. [sighs] I kept thinking about what you said. [muttering] Which I know is ridiculous.",
    "Okay so here's the thing. [clears throat] I may have accidentally told him. [whispers] Don't be mad.",
    "She handed me the results and I just— [gasps] I couldn't even speak. [crying] I know. I know.",
    "You're going to laugh at this. [laughs] Actually you're already laughing. This is fine.",
    "I thought it would feel better than this. [sighs] Like, I got what I wanted. [exhales] So why does it feel like nothing?",
    "[excited] Okay so I ran the numbers and— [wheezing] I'm sorry, I'm so excited I can barely talk.",
    "He just looked at me. [laughs] And I looked at him. [laughs harder] Neither of us said anything for a full minute.",
    "I promise I had a plan. [clears throat] The plan did not survive contact with reality. [chuckles]",
    "[sad] I keep starting sentences I don't know how to finish. [sighs] This is one of them.",
    "I tried to explain it calmly. [frustrated sigh] I was not calm. [sighs] I'm working on it.",
    "[whispers] He's been standing there for ten minutes. [gasps] Oh no he's coming over.",
    "Listen. [clears throat] I have something to tell you. [sighs] I don't know how to say this.",
    "Three seconds. [inhales deeply] Two. One. [exhales sharply] Okay. Let's go.",
    "[delighted] Oh— [gasps] wait is that— [laughs] how did you know this is my favorite?",

]

# ── Voice list for ElevenLabs API generation ─────────────────────────────────
# These span a range of ages, genders, accents, and registers.
# Each text will be generated for EVERY voice to get voice-matched paralinguistics.

ELEVENLABS_VOICES = [
    "Rachel",      # Young woman, American, warm/conversational
    "Domi",        # Young woman, American, confident/energetic
    "Bella",       # Young woman, American, soft/expressive
    "Antoni",      # Young man, American, calm/natural
    "Josh",        # Young man, American, deep/confident
    "Arnold",      # Middle-aged man, American, authoritative
    "Sam",         # Young man, American, raspy/dynamic
    "Callum",      # Middle-aged man, British, intense
    "Charlotte",   # Young woman, British/Swedish, seductive
    "Matilda",     # Middle-aged woman, American, warm/authoritative
]

# ── Fish-Speech voices (open source fallback) ─────────────────────────────────
# Run locally, no TOS issues, unlimited generation.
# Pass reference audio clips to get variety.
FISH_SPEECH_REFS = []  # populate with paths to ref clips
