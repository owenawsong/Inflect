"""
Inflect Phase 4 — LLM Text Preprocessor

Converts natural language text → structured Kokoro input with prosody hints.

Input:  "wow that's crazy!!! [laugh] i can't believe it..."
Output: {
  "segments": [
    {"text": "wow that's crazy", "emotion": "excited", "speed": 1.1},
    {"tag": "laugh", "intensity": 0.9},
    {"text": "i can't believe it", "emotion": "amazed", "speed": 1.0}
  ]
}

Uses Qwen3-0.6B (tiny, runs on RTX 3060) to:
1. Parse paralinguistic tags `[laugh]`, `[sigh]`, etc.
2. Detect emotion from punctuation + text ("!!!" = excited)
3. Suggest prosody (speed changes, pauses)
4. Break into natural speech chunks

Runtime: ~2s per sentence on RTX 3060.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE    = Path(r"C:\Users\Owen\Inflect-New\voice-encoder")
OUT_DIR = BASE / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion detection rules (before LLM, for speed)
EMOTION_RULES = {
    "excited": r"!{2,}|wow|amazing|crazy|incredible",
    "sad": r"\.\.\.|sigh|unfortunately|sad|depressed",
    "surprised": r"\?{2,}|what|really|no way",
    "angry": r"!{3,}|hate|terrible|awful",
    "calm": r"\.|hm|mm|hmm",
}

# Paralinguistic tags and their defaults
NVV_TAGS = {
    "laugh": {"intensity": 0.7, "duration": 1.0},
    "cough": {"intensity": 0.8, "duration": 0.8},
    "sigh": {"intensity": 0.6, "duration": 0.7},
    "sneeze": {"intensity": 0.9, "duration": 0.6},
    "breath": {"intensity": 0.5, "duration": 0.5},
    "sniff": {"intensity": 0.6, "duration": 0.4},
    "throat": {"intensity": 0.7, "duration": 0.5},
    "pause": {"duration": 0.5},
}


class TextPreprocessor:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.model = None
        self.tokenizer = None

        if use_llm:
            self.load_qwen()

    def load_qwen(self):
        """Load Qwen3-0.6B for advanced text understanding."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print("Loading Qwen3-0.6B...")
            model_id = "Qwen/Qwen2.5-0.5B"  # smallest, ~300MB
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=DEVICE,
            )
            self.model.eval()
            print("Qwen loaded.")
        except ImportError:
            print("WARNING: transformers not installed. Using rule-based preprocessing only.")
            self.use_llm = False

    def detect_emotion(self, text: str) -> str:
        """Detect emotion from text patterns."""
        import re
        text_lower = text.lower()
        for emotion, pattern in EMOTION_RULES.items():
            if re.search(pattern, text_lower):
                return emotion
        return "neutral"

    def parse_tags(self, text: str) -> list[dict]:
        """Extract [tag] markers and their positions."""
        import re
        matches = []
        for match in re.finditer(r"\[(\w+)\]", text):
            tag = match.group(1).lower()
            if tag in NVV_TAGS:
                matches.append({
                    "tag": tag,
                    "start": match.start(),
                    "end": match.end(),
                    "config": NVV_TAGS[tag].copy(),
                })
        return matches

    def break_into_chunks(self, text: str) -> list[str]:
        """Break text into natural speech units (sentences, clauses)."""
        import re
        # Split on sentence boundaries (keep trailing punctuation groups together)
        chunks = re.split(r'(?<=[.!?])\s+', text.strip())
        return [c.strip() for c in chunks if c.strip()]

    def estimate_speed(self, text: str) -> float:
        """Estimate speech speed based on text characteristics."""
        # More punctuation = more emotion = faster
        punct_count = sum(1 for c in text if c in "!?")
        if punct_count >= 2:
            return 1.15
        elif punct_count == 1:
            return 1.05
        else:
            return 1.0

    def process(self, text: str) -> dict:
        """Main preprocessing: text → structured segments preserving tag positions."""
        import re

        # Split text into alternating speech/tag parts while preserving order
        pattern = r'\[(\w+)\]'
        parts = re.split(pattern, text)

        segments = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Speech segment — further split into sentence chunks
                part = part.strip()
                if not part:
                    continue
                for chunk in self.break_into_chunks(part):
                    if chunk.strip():
                        emotion = self.detect_emotion(chunk)
                        speed   = self.estimate_speed(chunk)
                        segments.append({
                            "type":    "speech",
                            "text":    chunk.strip(),
                            "emotion": emotion,
                            "speed":   speed,
                        })
            else:
                # Tag
                tag = part.lower()
                if tag in NVV_TAGS:
                    segments.append({
                        "type": "tag",
                        "tag":  tag,
                        **NVV_TAGS[tag].copy(),
                    })

        return {
            "segments":  segments,
            "raw_text":  text,
            "uses_llm":  self.use_llm,
        }

    def format_for_synthesis(self, processed: dict) -> str:
        """Convert processed output → format for inflect_tts.py."""
        parts = []
        for seg in processed["segments"]:
            if seg["type"] == "speech":
                parts.append(seg["text"])
            elif seg["type"] == "tag":
                parts.append(f"[{seg['tag']}]")
        return " ".join(parts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Text to preprocess")
    parser.add_argument("--use-llm", action="store_true", help="Use Qwen for advanced understanding")
    parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    preprocessor = TextPreprocessor(use_llm=args.use_llm)
    result = preprocessor.process(args.text)

    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print("Processed segments:")
        for i, seg in enumerate(result["segments"]):
            if seg["type"] == "speech":
                print(f"  {i}. SPEECH: \"{seg['text'][:60]}\"")
                print(f"      emotion={seg['emotion']}, speed={seg['speed']:.2f}")
            elif seg["type"] == "tag":
                print(f"  {i}. TAG: [{seg['tag']}] (intensity={seg.get('intensity', '?')})")

        # Show synthesizable format
        synth_text = preprocessor.format_for_synthesis(result)
        print(f"\nFor synthesis:")
        print(f"  python inflect_tts.py \"{synth_text}\"")


if __name__ == "__main__":
    main()
