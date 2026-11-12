import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune_v3.frontend import Frontend, FrontendError, load_vocab, normalize_text, phoneme_ids


def test_language_neutral_unicode_and_no_english_expansion():
    assert normalize_text("  Cafe\u0301\t42\nDr. ") == "Caf\u00e9 42 Dr."
    with pytest.raises(FrontendError, match="control"):
        normalize_text("hidden\u202etext")


def test_prephonemized_preserves_original_and_wraps_boundaries():
    front = Frontend("fr-fr", "prephonemized", {"a": 1, " ": 2, "b": 3})
    result = front.encode("Bonjour 42.", phones="  a\tb  ")
    assert result["original_phones"] == "  a\tb  "
    assert result["normalized_phones"] == result["phones"] == "a b"
    assert result["phone_ids"] == [0, 1, 2, 3, 0]
    assert front.metadata()["english_fallback"] is False


@pytest.mark.parametrize("language,mode", [("", "prephonemized"), ("fr-fr", "english"), (None, "espeak")])
def test_explicit_language_and_mode_required(language, mode):
    with pytest.raises(FrontendError):
        Frontend(language, mode, {"a": 1})


def test_unknown_phones_are_not_dropped_or_converted():
    with pytest.raises(FrontendError, match="Unknown phones"):
        Frontend("fr-fr", "prephonemized", {"a": 1}).encode("Bonjour", phones="a\u0281")
    with pytest.raises(FrontendError):
        phoneme_ids("", {"a": 1})


def test_explicit_phone_map_retains_source_and_does_not_recurse():
    rules = {"language": "fr-fr", "name": "user-reviewed-codepoints", "replacements": {"x": "a", "a": "b"}}
    front = Frontend("fr-fr", "prephonemized", {"a": 1, "b": 2}, phone_mapping=rules)
    result = front.encode("Bonjour", phones="xa")
    assert result["normalized_phones"] == "xa"
    assert result["phones"] == "ab"
    assert result["coverage"]["mapped_codepoints"] == 2
    assert front.metadata()["phone_mapping"] == rules


@pytest.mark.parametrize("rules", [
    {"language": "en-us", "name": "wrong-language", "replacements": {"x": "a"}},
    {"language": "fr-fr", "name": "drop", "replacements": {"x": ""}},
    {"language": "fr-fr", "name": "expand", "replacements": {"x": "ab"}},
])
def test_no_implicit_lossy_or_cross_language_mapping(rules):
    with pytest.raises(FrontendError):
        Frontend("fr-fr", "prephonemized", {"a": 1}, phone_mapping=rules)


def test_reserved_id_extension_preserves_base(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"a": 6, "b": 100}), encoding="utf-8")
    assert load_vocab(path, base_vocab={"a": 6}) == {"a": 6, "b": 100}
    with pytest.raises(FrontendError, match="existing"):
        load_vocab({"a": 7, "b": 100}, base_vocab={"a": 6})
    for bad in ({"a": 0}, {"a": 178}, {"a": True}, {"ab": 1}, {"a": 1, "b": 1}):
        with pytest.raises(FrontendError):
            load_vocab(bad)


def test_wrapped_vocab_file_preserves_base_identity_and_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "extension.json"
    path.write_text(json.dumps(dict(vocab={"a": 6, "\u0250": 100}, base_vocab={"a": 6})), encoding="utf-8")
    front = Frontend("fr-fr", "prephonemized", path)
    assert front.metadata()["base_vocab_sha256"] is not None
    assert front.encode("test", phones="\u0250")["phone_ids"] == [0, 100, 0]
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(FrontendError, match="Duplicate"):
        load_vocab(path)


def test_espeak_switch_flags_are_hard_error_not_removed():
    front = Frontend("fr-fr", "espeak", {"a": 1})
    front.backend = SimpleNamespace(phonemize=lambda *a, **k: ["(en)a(fr)"])
    with pytest.raises(FrontendError, match="switch"):
        front.encode("word")


def test_espeak_failure_never_uses_english_or_given_phones(monkeypatch):
    import finetune_v3.frontend as module
    calls = []
    def fail(language):
        calls.append(language)
        raise RuntimeError("unavailable")
    monkeypatch.setattr(module, "_make_espeak", fail)
    front = Frontend("zz-invalid", "espeak", {"a": 1})
    with pytest.raises(FrontendError, match="no fallback"):
        front.encode("hello")
    assert calls == ["zz-invalid"]
    with pytest.raises(FrontendError, match="mix"):
        front.encode("hello", phones="a")


def test_real_local_espeak_has_explicit_language_and_coverage():
    from finetune_v3.frontend import _make_espeak
    from phonemizer.separator import Separator
    backend = _make_espeak("fr-fr")
    phones = backend.phonemize(["Bonjour."], separator=Separator(phone="", word=" ", syllable=""), strip=True, njobs=1)[0]
    vocabulary = {c: i + 1 for i, c in enumerate(sorted(set(normalize_text(phones))))}
    front = Frontend("fr-fr", "espeak", vocabulary)
    result = front.encode("Bonjour.")
    assert result["phones"] == normalize_text(phones)
    assert result["phone_ids"][0] == result["phone_ids"][-1] == 0
    assert front.metadata()["language"] == "fr-fr"
