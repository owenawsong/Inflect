# Inflect — Paralinguistic Tag Vocabulary
# Maps raw tag strings (from manifest) to clean canonical names and integer IDs

# Raw tag → canonical tag mapping
# Groups similar/variant tags into a single canonical form
TAG_ALIASES = {
    # Laughter family
    "laughs":           "laughs",
    "laughing":         "laughs",
    "starts laughing":  "laughs",
    "laughs harder":    "laughs_hard",
    "chuckles":         "chuckles",
    "giggles":          "giggles",
    "wheezing":         "wheezing",
    "snorts":           "snorts",

    # Breath/air family
    "sighs":            "sighs",
    "frustrated sigh":  "sighs",
    "exhales":          "exhales",
    "exhales sharply":  "exhales",
    "inhales deeply":   "inhales",
    "gasps":            "gasps",
    "happy gasp":       "gasps",

    # Sadness/distress family
    "crying":           "crying",

    # Throat family
    "clears throat":    "clears_throat",
    "muttering":        "muttering",

    # Whisper
    "whispers":         "whispers",

    # Emotion-as-sound (these affect delivery, Para Module generates short sound cues)
    "excited":          "excited",
    "angry":            "angry",
    "sad":              "sad",
    "curious":          "curious",
    "dramatically":     "dramatically",
    "impressed":        "impressed",
    "sarcastic":        "sarcastic",
    "surprised":        "surprised",
    "delighted":        "delighted",
    "warmly":           "warmly",
    "thoughtful":       "thoughtful",
}

# Canonical tag → integer ID
TAGS = [
    # Sound events (Para Module generates actual sounds for these)
    "laughs",           # 0
    "laughs_hard",      # 1
    "chuckles",         # 2
    "giggles",          # 3
    "wheezing",         # 4
    "snorts",           # 5
    "sighs",            # 6
    "exhales",          # 7
    "inhales",          # 8
    "gasps",            # 9
    "crying",           # 10
    "clears_throat",    # 11
    "muttering",        # 12
    "whispers",         # 13

    # Emotion delivery cues (shorter sound cues, affect prosody)
    "excited",          # 14
    "angry",            # 15
    "sad",              # 16
    "curious",          # 17
    "dramatically",     # 18
    "impressed",        # 19
    "sarcastic",        # 20
    "surprised",        # 21
    "delighted",        # 22
    "warmly",           # 23
    "thoughtful",       # 24
]

TAG_TO_ID = {tag: i for i, tag in enumerate(TAGS)}
ID_TO_TAG = {i: tag for i, tag in enumerate(TAGS)}
NUM_TAGS = len(TAGS)


def normalize_tag(raw_tag: str) -> str | None:
    """Normalize a raw tag string from the manifest to canonical form.
    Returns None if the tag is not in our vocabulary."""
    raw = raw_tag.strip().lower()
    return TAG_ALIASES.get(raw, None)


def tag_to_id(raw_tag: str) -> int | None:
    """Convert raw tag string to integer ID. Returns None if unknown."""
    canonical = normalize_tag(raw_tag)
    if canonical is None:
        return None
    return TAG_TO_ID.get(canonical, None)
