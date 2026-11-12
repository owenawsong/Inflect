import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from finetune_v3.data import audit, load_prepared, prepare

VOCAB = {"a": 1, "b": 2, " ": 3}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path, count=4, samples=2400, rate=24000):
    root = tmp_path / "raw"
    root.mkdir()
    rows = []
    for i in range(count):
        wave = (.1 * np.sin(np.arange(samples) * (i + 1) * .13)).astype(np.float32)
        wave[-600:] = 0
        sf.write(root / f"{i}.wav", wave, rate, subtype="FLOAT")
        rows.append(dict(id=f"id-{i}", audio=f"{i}.wav", text=f"Authored sentence number {i}.",
                         phones="a", phone_durations=[1, 2, 1], group=f"session-{i}",
                         source={"description": "Local authored fixture"}, consent={"declaration": "fixture author"},
                         alignment_source={"kind": "human", "language": "fr-fr", "source": "authored test alignment"}))
    manifest = tmp_path / "input.jsonl"
    write_rows(manifest, rows)
    return manifest, root, rows


def run_prepare(tmp_path, manifest, root, **kwargs):
    return prepare(manifest, root, tmp_path / "prepared", "fr-fr", "prephonemized", VOCAB, 42, **kwargs)


def test_prepare_audit_and_training_schema(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    path = run_prepare(tmp_path, manifest, root)
    assert path.name == "dataset.json"
    data = load_prepared(path)
    assert len(data["rows"]) == 4 and data["train"] and data["validation"]
    for row in data["rows"]:
        assert row["phone_ids"] == [0, 1, 0]
        assert row["phone_durations"] == row["durations"] == [1, 2, 1]
        assert row["phones"] == row["phonemes"] == "a"
        assert row["sample_count"] == sf.info(path.parent / row["audio"]).frames == 2400
        assert not Path(row["audio"]).is_absolute()
        assert "not independently verified" in row["metadata_verification"]
    assert audit(path)["passed"] is True
    assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}


def test_split_is_manifest_order_independent(tmp_path):
    manifest, root, rows = fixture(tmp_path, count=8)
    a = load_prepared(run_prepare(tmp_path, manifest, root))
    write_rows(manifest, list(reversed(rows)))
    b = load_prepared(prepare(manifest, root, tmp_path / "second", "fr-fr", "prephonemized", VOCAB, 42))
    assert [(r["id"], r["split"], r["component_id"]) for r in a["rows"]] == [(r["id"], r["split"], r["component_id"]) for r in b["rows"]]


def test_transitive_group_transcript_and_audio_leakage(tmp_path):
    manifest, root, rows = fixture(tmp_path, count=5)
    rows[1]["group"] = rows[0]["group"]
    rows[2]["text"] = rows[1]["text"].upper()
    rows[3]["audio"] = rows[2]["audio"]
    write_rows(manifest, rows)
    data = load_prepared(run_prepare(tmp_path, manifest, root))
    assert len({r["component_id"] for r in data["rows"][:4]}) == 1
    assert len({r["split"] for r in data["rows"][:4]}) == 1
    assert data["rows"][4]["split"] != data["rows"][0]["split"]


def test_explicit_leaking_split_rejected_before_any_output(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    for i, row in enumerate(rows):
        row["split"] = "validation" if i == 0 else "train"
    rows[1]["text"] = rows[0]["text"]
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="leakage"):
        run_prepare(tmp_path, manifest, root)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("change,match", [
    ({"phone_durations": None}, "durations"),
    ({"phone_durations": [2]}, "boundaries"),
    ({"phone_durations": [0, 3, 1]}, "boundaries"),
    ({"phone_durations": [True, 2, 1]}, "boundaries"),
    ({"phone_durations": [1, 1, 1]}, "totals"),
    ({"phones": "unknown"}, "Unknown phones"),
    ({"group": ""}, "Empty"),
    ({"source": {}}, "source"),
    ({"consent": {}}, "consent"),
    ({"audio": "../outside.wav"}, "escapes"),
    ({"audio": "C:/outside.wav"}, "Unsafe"),
    ({"alignment_source": {"kind": "uniform", "language": "fr-fr", "source": "guess"}}, "forbidden"),
    ({"alignment_source": {"kind": "human", "language": "en-us", "source": "English base"}}, "language"),
    ({"alignment_phones": "b"}, "bind"),
])
def test_strict_rejections(tmp_path, change, match):
    manifest, root, rows = fixture(tmp_path)
    rows[0].update(change)
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match=match):
        run_prepare(tmp_path, manifest, root)
    assert not (tmp_path / "prepared").exists()


def test_duplicate_ids_and_single_component_rejected(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    rows[1]["id"] = rows[0]["id"].upper()
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        run_prepare(tmp_path, manifest, root)
    rows[1]["id"] = "new-id"
    for row in rows:
        row["group"] = "same-recording"
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="independent"):
        run_prepare(tmp_path, manifest, root)


def test_explicit_clock_trim_only_silent_tail(tmp_path):
    manifest, root, rows = fixture(tmp_path, samples=2499)
    with pytest.raises(ValueError, match="off the"):
        run_prepare(tmp_path, manifest, root)
    with pytest.raises(ValueError, match="declared"):
        run_prepare(tmp_path, manifest, root, clock_tolerance_samples=100)
    for row in rows:
        row["trailing_silence_samples"] = 600
    write_rows(manifest, rows)
    data = load_prepared(run_prepare(tmp_path, manifest, root, clock_tolerance_samples=100))
    assert all(r["audio_preparation"]["trimmed_tail_samples"] == 99 for r in data["rows"])


def test_non_silent_tail_not_cut(tmp_path):
    manifest, root, rows = fixture(tmp_path, samples=2499)
    for row in rows:
        row["trailing_silence_samples"] = 600
    sf.write(root / "0.wav", np.full(2499, .1, dtype=np.float32), 24000, subtype="FLOAT")
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="non-silent"):
        run_prepare(tmp_path, manifest, root, clock_tolerance_samples=100)


def test_explicit_clock_padding_transparent(tmp_path):
    manifest, root, rows = fixture(tmp_path, samples=2301)
    data = load_prepared(run_prepare(tmp_path, manifest, root, clock_tolerance_samples=100))
    assert all(r["audio_preparation"]["padded_tail_samples"] == 99 for r in data["rows"])


def test_resample_requires_opt_in_and_retains_source_clock(tmp_path):
    manifest, root, rows = fixture(tmp_path, samples=4800, rate=48000)
    with pytest.raises(ValueError, match="resampling"):
        run_prepare(tmp_path, manifest, root)
    data = load_prepared(run_prepare(tmp_path, manifest, root, allow_resample=True))
    assert all(r["sample_count"] == 2400 and r["audio_preparation"]["source_sample_rate"] == 48000 for r in data["rows"])


def test_timed_alignment_quantizes_real_intervals(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    for row in rows:
        del row["phone_durations"]
        row["alignment"] = [dict(token_id=0, start=0, end=.025),
                            dict(token_id=1, start=.025, end=.075),
                            dict(token_id=0, start=.075, end=.1)]
    write_rows(manifest, rows)
    data = load_prepared(run_prepare(tmp_path, manifest, root))
    assert all(r["phone_durations"] == [1, 2, 1] for r in data["rows"])


def test_timed_alignment_gap_wrong_token_and_incomplete_audio_fail(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    row = rows[0]
    row.pop("phone_durations")
    row["alignment"] = [dict(token_id=0, start=0, end=.025), dict(token_id=1, start=.03, end=.075), dict(token_id=0, start=.075, end=.1)]
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="contiguous"):
        run_prepare(tmp_path, manifest, root)
    row["alignment"][1]["start"] = .025
    row["alignment"][1]["token_id"] = 2
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="token IDs"):
        run_prepare(tmp_path, manifest, root)
    row["alignment"][1]["token_id"] = 1
    row["alignment"][-1]["end"] = .09
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="complete"):
        run_prepare(tmp_path, manifest, root)


def test_output_no_overwrite_and_cannot_be_inside_audio_root(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    run_prepare(tmp_path, manifest, root)
    with pytest.raises(ValueError, match="New output"):
        run_prepare(tmp_path, manifest, root)
    with pytest.raises(ValueError, match="disjoint"):
        prepare(manifest, root, root / "prepared", "fr-fr", "prephonemized", VOCAB, 42)


def test_tampered_audio_rejected_by_load_prepared(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    path = run_prepare(tmp_path, manifest, root)
    data = load_prepared(path)
    audio = path.parent / data["rows"][0]["audio"]
    with audio.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="audio hash"):
        load_prepared(path)


def test_symlink_source_and_parent_rejected(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    link = root / "linked.wav"
    try:
        link.symlink_to(root / "0.wav")
    except OSError:
        pytest.skip("OS does not permit test symlinks")
    rows[0]["audio"] = "linked.wav"
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="Symlinks"):
        run_prepare(tmp_path, manifest, root)


def test_mapping_requires_final_label_binding_and_keeps_original(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    for row in rows:
        row["phones"] = "x"
    rules = dict(language="fr-fr", name="explicit-test-map", replacements={"x": "a"})
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="bind"):
        run_prepare(tmp_path, manifest, root, phone_mapping=rules)
    for row in rows:
        row["alignment_phones"] = "a"
    write_rows(manifest, rows)
    data = load_prepared(run_prepare(tmp_path, manifest, root, phone_mapping=rules))
    assert all(r["original_phones"] == "x" and r["phones"] == "a" for r in data["rows"])


def test_non_ascii_prepared_roundtrip_never_uses_windows_default_encoding(tmp_path, monkeypatch):
    manifest, root, rows = fixture(tmp_path)
    phones = "\u0250\u0281\u026a"
    vocabulary = {phone: i + 1 for i, phone in enumerate(phones)}
    for i, row in enumerate(rows):
        row.update(text=f"\u00c9nonc\u00e9 num\u00e9ro {i}.", phones=phones, phone_durations=[1, 1, 1, 1, 1])
        wave = (.1 * np.sin(np.arange(3000) * (i + 1) * .13)).astype(np.float32)
        sf.write(root / row["audio"], wave, 24000, subtype="FLOAT")
    write_rows(manifest, rows)
    original_read = Path.read_text
    def explicit_only(path, *args, **kwargs):
        if not args and kwargs.get("encoding") not in ("utf-8", "utf-8-sig"):
            raise AssertionError("Locale-dependent text read: " + str(path))
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", explicit_only)
    path = prepare(manifest, root, tmp_path / "non-ascii", "fr-fr", "prephonemized", vocabulary, 42)
    data = load_prepared(path)
    assert data["metadata"]["vocab"] == vocabulary
    assert all(row["phones"] == phones and row["text"].startswith("\u00c9nonc\u00e9") for row in data["rows"])


def test_phonemes_duration_aliases_and_source_hash(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    for row in rows:
        row["phonemes"] = row.pop("phones")
        row["durations"] = row.pop("phone_durations")
        row["audio_sha256"] = hashlib.sha256((root / row["audio"]).read_bytes()).hexdigest()
    write_rows(manifest, rows)
    assert audit(run_prepare(tmp_path, manifest, root))["passed"]


def test_wrong_input_audio_hash_rejected(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    rows[0]["audio_sha256"] = "0" * 64
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="Input audio hash"):
        run_prepare(tmp_path, manifest, root)


def test_case_variant_groups_stay_in_same_component(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    rows[0]["group"] = "Speaker A"
    rows[1]["group"] = "SPEAKER A"
    write_rows(manifest, rows)
    data = load_prepared(run_prepare(tmp_path, manifest, root))
    assert data["rows"][0]["component_id"] == data["rows"][1]["component_id"]


def test_prepared_index_requires_all_file_hashes(tmp_path):
    manifest, root, rows = fixture(tmp_path)
    path = run_prepare(tmp_path, manifest, root)
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptor["files"].pop("vocab.json")
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError, match="hash all"):
        load_prepared(path)
