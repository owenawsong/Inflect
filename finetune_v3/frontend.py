"""Language-explicit phonemes for native178; no English fallback or IPA guessing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping
import unicodedata


class FrontendError(ValueError):
    pass


def _local_json(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise FrontendError("Linked vocabulary/phone-map paths are not supported")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FrontendError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise FrontendError("Nonfinite JSON: " + value)
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique, parse_constant=nonfinite)


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise FrontendError("Expected Unicode text")
    if any(unicodedata.category(c).startswith("C") and c not in "\r\n\t" for c in text):
        raise FrontendError("Unsupported Unicode control/format character")
    result = " ".join(unicodedata.normalize("NFC", text).split())
    if not result:
        raise FrontendError("Empty text/phones after normalization")
    return result


def load_vocab(vocab, *, base_vocab=None) -> dict[str, int]:
    """Accept an explicit mapping or local JSON; never assign or recycle IDs."""
    if isinstance(vocab, (str, Path)):
        vocab = _local_json(vocab)
    if isinstance(vocab, Mapping) and "vocab" in vocab:
        if base_vocab is None:
            base_vocab = vocab.get("base_vocab")
        vocab = vocab["vocab"]
    if not isinstance(vocab, Mapping) or not vocab:
        raise FrontendError("Explicit native178 vocabulary mapping required")
    result = dict(vocab)
    if not all(isinstance(k, str) and len(k) == 1 and type(v) is int and 0 < v < 178
               for k, v in result.items()):
        raise FrontendError("Vocabulary needs single codepoints mapped to IDs 1..177; zero is boundary-only")
    if len(set(result.values())) != len(result):
        raise FrontendError("Vocabulary aliases cannot silently share an embedding ID")
    if base_vocab is not None:
        base = load_vocab(base_vocab)
        if any(result.get(phone) != ident for phone, ident in base.items()):
            raise FrontendError("Explicit vocabulary extension cannot change existing mapped IDs")
        if any(ident in set(base.values()) for phone, ident in result.items() if phone not in base):
            raise FrontendError("New phones must use previously unused native178 IDs")
    return result


def phoneme_ids(phones: str, vocab) -> list[int]:
    mapping = load_vocab(vocab)
    if not isinstance(phones, str) or not 1 <= len(phones) <= 510:
        raise FrontendError("Final phone sequence must contain 1..510 codepoints")
    missing = sorted(set(phones) - set(mapping))
    if missing:
        raise FrontendError("Unknown phones; no dropping, IPA conversion, or ID assignment: " + repr(missing))
    return [0, *(mapping[p] for p in phones), 0]


def _make_espeak(language):
    from phonemizer.backend import EspeakBackend
    if not EspeakBackend.is_available():
        if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
            raise FrontendError("Configured eSpeak library is unavailable")
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
        espeakng_loader.make_library_available()
        espeakng_loader.load_library()
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
    if language not in EspeakBackend.supported_languages():
        raise FrontendError("Unsupported explicit eSpeak language: " + language)
    return EspeakBackend(language=language, preserve_punctuation=True, with_stress=True,
                         language_switch="keep-flags", words_mismatch="warn")


class Frontend:
    """Explicit eSpeak or caller phonemes, plus an optional audited codepoint map.

    Mapping schema: {"language": "fr-fr", "name": "...", "replacements": {"x": "y"}}.
    Maps are language-specific and one-codepoint to one-codepoint only. Original
    output and normalized pre-map phones are retained for coverage inspection.
    Final mapped codepoints are not normalized again, including for alignment.
    """
    def __init__(self, language, frontend, vocab, *, phone_mapping=None, base_vocab=None):
        if not isinstance(language, str) or not language or language != language.strip():
            raise FrontendError("Explicit nonempty language required; there is no default language")
        if frontend not in ("prephonemized", "espeak"):
            raise FrontendError("frontend must be prephonemized or espeak")
        self.language, self.mode = language, frontend
        spec = _local_json(vocab) if isinstance(vocab, (str, Path)) else vocab
        self.vocab = load_vocab(spec, base_vocab=base_vocab)
        embedded_base = None
        if isinstance(spec, Mapping) and "vocab" in spec:
            embedded_base = spec.get("base_vocab")
        checked_base = base_vocab if base_vocab is not None else embedded_base
        self.base_vocab_sha256 = canonical_sha(load_vocab(checked_base)) if checked_base is not None else None
        if isinstance(phone_mapping, (str, Path)):
            phone_mapping = _local_json(phone_mapping)
        if phone_mapping is not None:
            if (not isinstance(phone_mapping, dict) or phone_mapping.get("language") != language
                    or not isinstance(phone_mapping.get("name"), str) or not phone_mapping["name"]
                    or not isinstance(phone_mapping.get("replacements"), dict)):
                raise FrontendError("Phone map needs exact language, name and replacements")
            if not all(isinstance(k, str) and len(k) == 1 and isinstance(v, str) and len(v) == 1
                       for k, v in phone_mapping["replacements"].items()):
                raise FrontendError("No deletions, expansions or multicodepoint replacement rules")
        self.phone_mapping = json.loads(json.dumps(phone_mapping, ensure_ascii=False))
        self.backend = None

    def metadata(self) -> dict:
        return dict(schema="inflect-v3-frontend-v1", mode=self.mode, language=self.language,
                    normalization="NFC and whitespace collapse; no English number expansion",
                    phone_mapping=self.phone_mapping, vocab=self.vocab.copy(),
                    vocab_sha256=canonical_sha(self.vocab),
                    base_vocab_sha256=self.base_vocab_sha256,
                    english_fallback=False, unknown_phone_policy="error",
                    boundary_id=0, embedding_rows=178,
                    espeak_version=str(self.backend.version()) if self.backend is not None else None)

    def encode(self, text, *, phones=None, audit_only=False) -> dict:
        if type(audit_only) is not bool:
            raise FrontendError('audit_only must be boolean')
        normalized_text = normalize_text(text)
        if self.mode == "prephonemized":
            if phones is None:
                raise FrontendError("Every prephonemized row needs explicit phones")
            original = phones
        else:
            if phones is not None:
                raise FrontendError("Do not mix caller phones with eSpeak mode")
            try:
                if self.backend is None:
                    self.backend = _make_espeak(self.language)
                from phonemizer.separator import Separator
                result = self.backend.phonemize([normalized_text],
                    separator=Separator(phone="", word=" ", syllable=""), strip=True, njobs=1)
                if len(result) != 1 or not isinstance(result[0], str):
                    raise FrontendError("Unexpected eSpeak result")
                original = result[0]
            except (ImportError, RuntimeError, OSError) as exc:
                raise FrontendError("Explicit eSpeak frontend failed; no fallback: " + str(exc)) from exc
            if re.search(r"\([a-zA-Z][a-zA-Z0-9-]*\)", original):
                raise FrontendError("eSpeak language-switch/fallback flags are not permitted")
        normalized = normalize_text(original)
        rules = self.phone_mapping["replacements"] if self.phone_mapping else {}
        final = "".join(rules.get(c, c) for c in normalized)
        unknown = sorted(set(final) - set(self.vocab))
        if not 1 <= len(final) <= 510:
            raise FrontendError('Final phone sequence must contain 1..510 codepoints')
        ids = None if audit_only else phoneme_ids(final, self.vocab)
        return dict(text=normalized_text, original_phones=original, normalized_phones=normalized,
                    phones=final, phonemes=final, phone_ids=ids,
                    coverage=dict(unknown_phones=unknown, audit_only=audit_only, final_codepoints=len(final),
                                  mapped_codepoints=sum(rules.get(c, c) != c for c in normalized)))

    __call__ = encode
