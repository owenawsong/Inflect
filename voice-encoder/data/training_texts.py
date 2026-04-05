"""
Paralinguistic training text corpus for Inflect Phase 3.

Covers: laughs, sighs, whispers, crying, gasps, coughs, emotional modifiers,
        fillers, and delivery tags. Each line has BOTH speech AND at least one tag.
Tags are ElevenLabs v3 format.

~100 texts across diverse contexts, tones, speakers, and registers.
Designed to produce maximally varied audio when synthesized across many voices.
"""

TEXTS = [

    # ── LAUGHS (most important — extremely voice-dependent) ───────────────────

    "I can't believe you actually said that. [laughs] That's the funniest thing I've heard all week.",
    "Wait, you tripped over your own feet? [laughs] In front of your boss? Oh no.",
    "She showed up to the wrong party. [laughs] The look on her face when she realized.",
    "He thought the meeting was tomorrow. [chuckles] It was very much today.",
    "I tried to cook dinner for once. [laughs] The smoke alarm disagreed with my methods.",
    "So you texted your professor instead of your friend. [laughs harder] What did it say?",
    "The cat just sat there judging me the entire time. [giggles] I felt so seen.",
    "He walked face-first into the glass door. [snorts] I tried not to laugh. I failed.",
    "You thought it was a costume party. It was not. [laughs] How long did you stay?",
    "She said 'sounds good' to an email she definitely did not read. [chuckles] Classic.",
    "I locked myself out in my pajamas at six AM. [laughs] The neighbor saw everything.",
    "He called his teacher 'mom' in front of the whole class. [wheezing] He never recovered.",
    "I set a reminder to take my medication. The reminder was from three weeks ago. [laughs]",
    "So I'm standing there, fully confident, giving the wrong presentation. [starts laughing] Complete silence.",
    "We showed up an hour early to the concert. It was the next day's concert. [giggles]",
    "I told him I'd be ready in five minutes. [laughs] It was forty-five minutes.",
    "She's been mispronouncing that word for years and nobody told her. [snorts] I'm not going to either.",
    "He asked me if I'd eaten and I said yes. I had eaten six hours ago. [chuckles] Does that count?",
    "I told myself I'd sleep early. [laughs] It is currently three in the morning.",
    "She put the cereal in the fridge and the milk in the cabinet. [laughing] And then wondered why it tasted wrong.",

    # ── SIGHS / EXHALES ───────────────────────────────────────────────────────

    "I don't know. [sighs] It's just been one of those weeks, you know?",
    "We've been over this so many times. [exhales] I genuinely don't know what else to say.",
    "I applied to fourteen jobs this month. [sighs] Two responses. Both rejections.",
    "He said he'd call. [exhales sharply] That was three weeks ago.",
    "I keep trying to explain it but nobody's listening. [frustrated sigh] It's exhausting.",
    "We were so close. [sighs] One more step and it would've worked.",
    "I cleaned the whole apartment. [sighs] They didn't notice.",
    "It rained on the one day I forgot my umbrella. [exhales] Of course it did.",
    "I've been sitting here for two hours trying to start this. [sighs] One sentence.",
    "I thought things would be different by now. [exhales] I was wrong.",

    # ── WHISPERS ──────────────────────────────────────────────────────────────

    "[whispers] Don't look now but I think that's him over there.",
    "Okay so the thing is— [whispers] I don't think we're supposed to be in here.",
    "[whispers] She still doesn't know it was me. And I'd like to keep it that way.",
    "The baby finally went to sleep. [whispers] Don't. Move. A. Muscle.",
    "[whispers] I found out something about the new manager. Meet me at lunch.",
    "Everything's fine. [whispers] Nothing is fine. But I'm managing.",
    "[whispers] There's someone standing just outside the door and I don't know who it is.",
    "Okay, everyone act normal. [whispers] He's literally right behind you.",
    "[whispers] I've been faking that I understood this the entire time.",

    # ── CRYING ────────────────────────────────────────────────────────────────

    "I'm fine. [crying] I promise I'm completely fine.",
    "She just— [crying] sorry. Give me a second.",
    "We had him for twelve years. [crying] He was the best dog.",
    "I didn't expect it to hit me like this. [crying] I don't know why I'm so emotional.",
    "They played our song at the end. [crying] I wasn't ready for that at all.",
    "I got the job. [crying] I can't believe I actually got the job.",
    "He said he was proud of me. [crying] Nobody's ever said that to me before.",
    "I thought I was over it. [crying] Clearly not.",

    # ── GASPS / SURPRISE ──────────────────────────────────────────────────────

    "[gasps] You got engaged? When? How? Tell me everything right now.",
    "Wait— [gasps] is that actually real? How much did that cost?",
    "[happy gasp] Oh my god. This is the best thing I've ever tasted.",
    "And then she just— [gasps] she actually said that to his face?",
    "[inhales deeply] Okay. I'm calm. I'm completely calm. I can handle this.",
    "[gasps] I completely forgot that was today. What time is it?",
    "No way. [gasps] No absolute way. How did you keep this secret for so long?",

    # ── EMOTIONAL DELIVERY MODIFIERS ─────────────────────────────────────────

    "[excited] Okay so I have the most insane idea and I need you to just hear me out.",
    "This is going to sound crazy but [excited] I think it might actually work.",
    "[sad] I used to love this place. It's not the same anymore.",
    "I remember when this felt effortless. [sad] Now it just feels like work.",
    "[angry] I specifically asked for one thing. One thing. And it still didn't happen.",
    "I've asked nicely. I've asked twice. [angry] I'm done asking.",
    "[curious] Have you ever thought about why we do things the way we do?",
    "Wait, [curious] if that's true then what does that mean for everything else we assumed?",
    "[thoughtful] I think the problem is that we're solving the wrong question entirely.",
    "Maybe it's not about being ready. [thoughtful] Maybe you just start anyway.",
    "[surprised] I didn't even know you two knew each other.",
    "You've had that opinion this whole time? [surprised] I had no idea.",
    "[sarcastic] Oh yes, because that plan has worked out so perfectly every other time.",
    "Sure. [sarcastic] I'm sure everything will just fall into place on its own.",
    "[impressed] I've seen a lot of presentations. That was genuinely one of the best.",
    "I didn't think it was possible in that timeframe. [impressed] Apparently I was wrong.",
    "[delighted] Oh this is perfect. This is exactly what I was hoping for.",
    "You remembered! [delighted] I can't believe you actually remembered.",
    "[amazed] I've been doing this for twenty years and I've never seen anything like it.",
    "She said it in three languages. Fluently. [amazed] She's seventeen years old.",
    "[warmly] You know, I think about that conversation more than you'd expect.",

    # ── DRAMATIC / PERFORMANCE ────────────────────────────────────────────────

    "[dramatically] And so it ends. As it always was going to end. In silence.",
    "After all this time. [dramatically] After everything we've been through. It comes down to this.",
    "[dramatically] I have fought armies. I have crossed oceans. I did not come this far to lose.",

    # ── CLEARS THROAT / TRANSITION ────────────────────────────────────────────

    "[clears throat] Right. So. Moving on from that particular disaster.",
    "Anyway. [clears throat] What I was trying to say before all of that.",
    "[clears throat] I'd like to say a few words, if that's alright with everyone.",
    "Sorry, give me a second. [clears throat] Okay. Where were we.",

    # ── FILLERS / THINKING ────────────────────────────────────────────────────

    "[muttering] Okay, okay. Where did I put that. I just had it.",
    "So the thing is— [muttering] no that's not right either. Hang on.",
    "I was going to say— [muttering] it was something about the timing. I forget.",
    "One second. [muttering] Seven plus nine is... carry the one... okay yeah sixteen.",
    "I'm not saying I'm lost but [muttering] none of these streets make any sense.",

    # ── WOO / EXCITEMENT ─────────────────────────────────────────────────────

    "We won! After everything, we actually won! [woo] I can't believe it!",
    "[woo] That's what I'm talking about! Let's go!",
    "It's finally done. Three years of work. [woo] We're done!",

    # ── MIXED / COMPLEX (multiple tags, varied flow) ──────────────────────────

    "I didn't sleep last night. [sighs] I kept thinking about what you said. [muttering] Which I know is ridiculous.",
    "Okay so here's the thing. [clears throat] I may have accidentally told him. [whispers] Don't be mad.",
    "She handed me the results and I just— [gasps] I couldn't even speak. [crying] I know. I know.",
    "You're going to laugh at this. [laughs] Actually you're already laughing. This is fine.",
    "[curious] What if we've been thinking about this backwards? [thoughtful] Like, what if the constraint is actually the answer?",
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
