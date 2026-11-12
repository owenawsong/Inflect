"""Strict, local, leakage-safe native24 dataset preparation and readback audit.

prepare(manifest, audioroot, output, language, frontend, vocab, splitseed) returns
output/dataset.json. Inputs are JSONL with id/audio/text/group/source/consent and
real boundary-inclusive alignment labels. Never generates supervision labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import unicodedata

import numpy as np
import soundfile as sf

from .frontend import Frontend, canonical_sha, load_vocab, normalize_text, phoneme_ids

SAMPLE_RATE = 24000
HOP = 600
SCHEMA = "inflect-v3-prepared-dataset-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _no_links(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, "is_junction") and part.is_junction()),
                "Symlinks/junctions are not accepted: " + str(part))
    return path


def _asset(root, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name, "Unsafe relative audio path")
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and ".." not in rel.parts, "Audio path escapes its root")
    path = _no_links(root.joinpath(*rel.parts))
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), "Missing/outside audio asset")
    return path


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))


def read_jsonl(path):
    path = _no_links(path)
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        require(bool(line.strip()), f"Blank JSONL row at line {number}")
        value = _json(line)
        require(isinstance(value, dict), f"JSONL row {number} must be an object")
        rows.append(value)
    require(rows, "Empty dataset")
    return rows


def _dump(path, value, *, lines=False):
    _no_links(path.parent)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        if lines:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
        else:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n")


def _audio(path, *, allow_resample, tolerance, trailing_silence_samples):
    info = sf.info(path)
    require(info.channels == 1 and info.frames > 0 and info.samplerate > 0, "Audio must be nonempty mono")
    require(info.frames / info.samplerate <= 30.025, "Audio exceeds native 1200-frame duration cap")
    require(info.samplerate == SAMPLE_RATE or allow_resample, "Audio must be 24 kHz; resampling requires explicit opt-in")
    wave, rate = sf.read(path, dtype="float32", always_2d=False)
    require(wave.ndim == 1 and np.isfinite(wave).all() and np.abs(wave).max() < 1,
            "Audio must be finite, mono, below full scale; no gain/clipping repair")
    raw_pcm_sha = hashlib.sha256(wave.astype("<f4", copy=False).tobytes()).hexdigest()
    source_count = len(wave)
    resample_rounding = 0
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        gcd = math.gcd(rate, SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // gcd, rate // gcd).astype(np.float32)
        expected = (source_count * SAMPLE_RATE * 2 + rate) // (2 * rate)
        resample_rounding = len(wave) - expected
        require(0 <= resample_rounding <= 1, "Unexpected resample length")
        wave = wave[:expected]
    native_count = len(wave)
    target = ((native_count + HOP // 2) // HOP) * HOP
    delta = target - native_count
    require(abs(delta) <= tolerance, "Audio is off the 600-sample clock; set explicit clock_tolerance_samples (0..300)")
    require(2 * HOP <= target <= 1200 * HOP, "Audio outside 2..1200 native frames")
    if delta < 0:
        require(type(trailing_silence_samples) is int and -delta <= trailing_silence_samples <= native_count,
                "Trimming needs caller-declared trailing_silence_samples at 24 kHz; no incomplete-word cuts")
        tail = wave[target:]
        require(float(np.max(np.abs(tail))) <= 10 ** (-55 / 20), "Refuse to cut a non-silent audio tail")
        wave = wave[:target]
    elif delta > 0:
        wave = np.pad(wave, (0, delta))
    require(np.isfinite(wave).all() and np.abs(wave).max() < 1, "Resampling created unsafe audio")
    return wave, dict(source_sample_rate=rate, source_sample_count=source_count,
        source_pcm_sha256=raw_pcm_sha, native_samples_before_clock_adjustment=native_count,
        resampled=rate != SAMPLE_RATE, resample_method="scipy.signal.resample_poly" if rate != SAMPLE_RATE else None,
        resample_rounding_trim_samples=resample_rounding, trimmed_tail_samples=max(-delta, 0),
        padded_tail_samples=max(delta, 0), clock_tolerance_samples=tolerance,
        trailing_silence_samples=trailing_silence_samples, sample_rate=SAMPLE_RATE, sample_count=target)


def _durations(row, encoded, audio, language):
    source = row.get("alignment_source")
    require(isinstance(source, dict) and source.get("kind") in
            ("human", "external_forced_alignment", "dataset_alignment"),
            "Explicit alignment_source kind required; uniform/base-predicted durations are forbidden")
    require(source.get("language") == language and isinstance(source.get("source"), str) and source["source"].strip(),
            "Alignment provenance needs matching language and nonempty source")
    require(source.get("base_predicted", False) is False, "English/base predictions are not duration labels")
    bound_phones = row.get("alignment_phones", row.get("phones", row.get("phonemes")))
    require(isinstance(bound_phones, str) and normalize_text(bound_phones) == encoded["phones"],
            "Alignment must explicitly bind to the final, mapped phoneme sequence")
    ids = encoded["phone_ids"]
    durations = row.get("phone_durations", row.get("durations"))
    if "phone_durations" in row and "durations" in row:
        require(row["phone_durations"] == row["durations"], "Conflicting duration aliases")
    alignment = row.get("alignment")
    if alignment is not None:
        require(isinstance(alignment, list) and len(alignment) == len(ids), "Timed alignment must include all boundary-wrapped tokens")
        stops = [0]
        previous = 0.0
        for index, (item, ident) in enumerate(zip(alignment, ids)):
            require(isinstance(item, dict) and type(item.get("token_id")) is int and item["token_id"] == ident,
                    "Timed alignment token IDs differ from final phones/boundaries")
            start, end = item.get("start"), item.get("end")
            require(type(start) in (int, float) and type(end) in (int, float) and math.isfinite(start) and math.isfinite(end)
                    and 0 <= start < end and abs(start - previous) * SAMPLE_RATE <= .5 + 1e-7,
                    "Timed alignment must be contiguous, finite and start at zero")
            previous = end
            stops.append(math.floor(end * SAMPLE_RATE / HOP + .5))
        raw_seconds = audio["native_samples_before_clock_adjustment"] / SAMPLE_RATE
        require(abs(previous - raw_seconds) * SAMPLE_RATE <= audio["clock_tolerance_samples"] + 1e-6,
                "Timed alignment does not cover the complete source audio")
        timed = [b - a for a, b in zip(stops, stops[1:])]
        require(durations is None or durations == timed, "Timed alignment and duration labels disagree")
        durations = timed
    require(isinstance(durations, list) and len(durations) == len(ids)
            and all(type(x) is int and x >= 1 for x in durations),
            "Real integer frame durations must cover each token including both zero boundaries")
    require(sum(durations) * HOP == audio["sample_count"], "Duration totals do not match audio on the native 600-sample clock")
    return durations, source


def _components(rows, seed, validation_fraction):
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for index, row in enumerate(rows):
        transcript = re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", row["text"]).casefold()).strip()
        require(bool(transcript), "Transcript needs a non-punctuation content key")
        keys = [("group", unicodedata.normalize("NFKC", row["group"]).casefold()), ("transcript", transcript),
                ("source_pcm", row["audio_preparation"]["source_sample_rate"], row["audio_preparation"]["source_pcm_sha256"]),
                ("prepared_pcm", row["prepared_pcm_sha256"])]
        if row["source"].get("recording_id") is not None:
            require(isinstance(row["source"]["recording_id"], str) and row["source"]["recording_id"], "Invalid source recording_id")
            keys.append(("recording", normalize_text(row["source"]["recording_id"]).casefold()))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            seen[key] = index
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    require(len(groups) >= 2, "At least two independent connected components needed for train/validation")
    components = {}
    for members in groups.values():
        ident = canonical_sha(sorted(rows[i]["id"] for i in members))
        components[ident] = members
    has_splits = any("requested_split" in row for row in rows)
    if has_splits:
        require(all(row.get("requested_split") in ("train", "validation") for row in rows), "Supply every explicit split as train or validation")
        assignments = {}
        for ident, members in components.items():
            splits = {rows[i]["requested_split"] for i in members}
            require(len(splits) == 1, "Train/validation leakage through duplicate/group/transcript component")
            assignments[ident] = next(iter(splits))
        require(set(assignments.values()) == {"train", "validation"}, "Both splits need independent components")
    else:
        ranked = sorted(components, key=lambda ident: hashlib.sha256(f"{seed}:{ident}".encode()).hexdigest())
        count = min(len(ranked) - 1, max(1, math.floor(len(ranked) * validation_fraction + .5)))
        assignments = {ident: "validation" if ident in set(ranked[:count]) else "train" for ident in ranked}
    for ident, members in components.items():
        for i in members:
            rows[i]["component_id"], rows[i]["split"] = ident, assignments[ident]
            rows[i].pop("requested_split", None)
    return len(components)


def prepare(manifest, audioroot, output, language, frontend, vocab, splitseed, *,
            validation_fraction=.1, clock_tolerance_samples=0, allow_resample=False,
            phone_mapping=None, base_vocab=None) -> Path:
    """Prepare a NEW directory; failed preflight writes nothing.

    Off-clock audio requires explicit tolerance <=300 samples. Trimming also
    needs row.trailing_silence_samples and a below -55 dBFS removed tail.
    Labels bind to row.alignment_phones (or row.phones when already final).
    """
    require(type(splitseed) is int and 0 <= splitseed < 2**32, "splitseed must be uint32")
    require(type(validation_fraction) in (float, int) and 0 < validation_fraction < 1, "validation_fraction must lie in (0,1)")
    require(type(clock_tolerance_samples) is int and 0 <= clock_tolerance_samples <= 300, "Clock tolerance must be 0..300 samples")
    require(type(allow_resample) is bool, "Explicit boolean resampling policy required")
    manifest = _no_links(manifest).resolve(strict=True)
    root = _no_links(audioroot).resolve(strict=True)
    destination = _no_links(output).absolute()
    require(root.is_dir() and destination.parent.is_dir() and not destination.exists(), "New output directory with existing parent required")
    require(not destination.resolve().is_relative_to(root) and not root.is_relative_to(destination.resolve()), "Output and raw audio root must be disjoint")
    front = Frontend(language, frontend, vocab, phone_mapping=phone_mapping, base_vocab=base_vocab)
    source_manifest_sha = sha256(manifest)
    source_rows = read_jsonl(manifest)
    rows, seen_ids = [], set()
    for raw in source_rows:
        require(isinstance(raw.get("id"), str), "Every row requires a string id")
        ident = normalize_text(raw["id"])
        require(len(ident) <= 200 and ident.casefold() not in seen_ids, "Duplicate/oversized normalized id")
        seen_ids.add(ident.casefold())
        require(isinstance(raw.get("group"), str), "Every row requires a leakage group")
        group = normalize_text(raw["group"])
        require(raw.get("language", language) == language, "Row language differs from the explicit dataset language")
        for key in ("source", "consent"):
            require(isinstance(raw.get(key), dict) and raw[key], "Nonempty user-supplied " + key + " metadata required")
        encoded = front.encode(raw.get("text"), phones=raw.get("phones", raw.get("phonemes")))
        if "phones" in raw and "phonemes" in raw:
            require(raw["phones"] == raw["phonemes"], "Conflicting phone aliases")
        path = _asset(root, raw.get("audio"))
        source_sha = sha256(path)
        for key in ("audio_sha256", "sha256"):
            if key in raw:
                require(raw[key] == source_sha, "Input audio hash differs from supplied " + key)
        wave, details = _audio(path, allow_resample=allow_resample, tolerance=clock_tolerance_samples,
                               trailing_silence_samples=raw.get("trailing_silence_samples"))
        durations, label_source = _durations(raw, encoded, details, language)
        row = dict(encoded, id=ident, raw_text=raw["text"], language=language, group=group,
                   audio="audio/" + hashlib.sha256(ident.encode()).hexdigest() + ".wav",
                   sample_count=len(wave), sample_rate=SAMPLE_RATE,
                   phone_durations=durations, durations=durations, source=raw["source"], consent=raw["consent"],
                   metadata_verification="User-supplied source/consent assertions; not independently verified rights",
                   alignment_source=label_source, audio_preparation=details,
                   source_audio=raw["audio"], source_audio_sha256=source_sha,
                   prepared_pcm_sha256=hashlib.sha256(wave.astype("<f4", copy=False).tobytes()).hexdigest())
        if "alignment" in raw:
            row["original_alignment"] = raw["alignment"]
        for key in ("voice", "speaker_id"):
            if key in raw:
                row[key] = raw[key]
        if "split" in raw:
            row["requested_split"] = raw["split"]
        rows.append(row)
    component_count = _components(rows, splitseed, validation_fraction)
    require(sha256(manifest) == source_manifest_sha, "Source manifest changed during preparation")
    rows.sort(key=lambda row: row["id"])
    _no_links(destination.parent)
    destination.mkdir(exist_ok=False)
    (destination / "audio").mkdir()
    for row in rows:
        source = _asset(root, row["source_audio"])
        require(sha256(source) == row["source_audio_sha256"], "Source audio changed after preflight")
        wave, details = _audio(source, allow_resample=allow_resample, tolerance=clock_tolerance_samples,
                               trailing_silence_samples=row["audio_preparation"]["trailing_silence_samples"])
        require(details == row["audio_preparation"] and hashlib.sha256(wave.astype("<f4", copy=False).tobytes()).hexdigest()
                == row["prepared_pcm_sha256"], "Audio changed during preparation")
        path = destination / row["audio"]
        _no_links(path.parent)
        with path.open("xb") as stream:
            sf.write(stream, wave, SAMPLE_RATE, format="WAV", subtype="FLOAT")
        row["audio_sha256"] = sha256(path)
    _dump(destination / "dataset.jsonl", rows, lines=True)
    _dump(destination / "train.jsonl", [r for r in rows if r["split"] == "train"], lines=True)
    _dump(destination / "validation.jsonl", [r for r in rows if r["split"] == "validation"], lines=True)
    _dump(destination / "vocab.json", front.vocab)
    metadata = dict(schema=SCHEMA, language=language, sample_rate=SAMPLE_RATE, samples_per_frame=HOP,
        frontend=front.metadata(), vocab=front.vocab, vocab_sha256=canonical_sha(front.vocab),
        source_manifest_sha256=source_manifest_sha, splitseed=splitseed,
        validation_fraction=validation_fraction, connected_components=component_count,
        rows="dataset.jsonl", train="train.jsonl", validation="validation.jsonl",
        counts=dict(Counter(r["split"] for r in rows)),
        files={name: sha256(destination / name) for name in ("dataset.jsonl", "train.jsonl", "validation.jsonl", "vocab.json")})
    _dump(destination / "dataset.json", metadata)
    result = audit(destination / "dataset.json")
    _dump(destination / "audit.json", result)
    return destination / "dataset.json"


def load_prepared(path) -> dict:
    """Hash-check index, split lists, audio, vocab, clocks and leakage before use."""
    path = _no_links(path)
    if path.is_dir():
        path = path / "dataset.json"
    metadata = _json(path.read_text(encoding="utf-8-sig"))
    root = path.resolve().parent
    require(metadata.get("schema") == SCHEMA and metadata.get("sample_rate") == SAMPLE_RATE
            and metadata.get("samples_per_frame") == HOP, "Wrong prepared dataset schema/clock")
    vocab = load_vocab(metadata["vocab"])
    require(canonical_sha(vocab) == metadata["vocab_sha256"], "Vocabulary digest differs")
    require(metadata["frontend"].get("vocab") == vocab
            and metadata["frontend"].get("vocab_sha256") == metadata["vocab_sha256"]
            and metadata["frontend"].get("language") == metadata["language"]
            and metadata["frontend"].get("mode") in ("prephonemized", "espeak")
            and metadata["frontend"].get("english_fallback") is False,
            "Frontend metadata differs from dataset vocabulary/language policy")
    require(set(metadata.get("files", {})) == {"dataset.jsonl", "train.jsonl", "validation.jsonl", "vocab.json"},
            "Prepared index must hash all row/split/vocabulary files")
    for name, digest in metadata["files"].items():
        require(sha256(_asset(root, name)) == digest, "Prepared file hash mismatch: " + name)
    require(_json(_asset(root, "vocab.json").read_text(encoding="utf-8-sig")) == vocab, "Prepared vocabulary differs")
    rows = read_jsonl(_asset(root, metadata["rows"]))
    train = read_jsonl(_asset(root, metadata["train"]))
    validation = read_jsonl(_asset(root, metadata["validation"]))
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate prepared id")
    for row in rows:
        for key in ("source", "consent", "alignment_source"):
            require(isinstance(row.get(key), dict) and row[key], "Missing prepared provenance: " + key)
        require(row["language"] == metadata["language"] and row["sample_rate"] == SAMPLE_RATE, "Prepared language/rate differs")
        require(row["phones"] == row["phonemes"] and phoneme_ids(row["phones"], vocab) == row["phone_ids"], "Prepared phone IDs differ")
        durations = row["phone_durations"]
        require(durations == row["durations"] and len(durations) == len(row["phone_ids"])
                and all(type(d) is int and d >= 1 for d in durations)
                and sum(durations) * HOP == row["sample_count"], "Prepared durations/clock differ")
        require(2 * HOP <= row["sample_count"] <= 1200 * HOP, "Prepared duration exceeds model cap")
        _durations(dict(row, alignment_phones=row["phones"]), row, row["audio_preparation"], row["language"])
        audio = _asset(root, row["audio"])
        require(sha256(audio) == row["audio_sha256"], "Prepared audio hash mismatch")
        wave, rate = sf.read(audio, dtype="float32")
        require(rate == SAMPLE_RATE and wave.ndim == 1 and len(wave) == row["sample_count"]
                and np.isfinite(wave).all() and np.abs(wave).max() < 1
                and hashlib.sha256(wave.astype("<f4", copy=False).tobytes()).hexdigest() == row["prepared_pcm_sha256"], "Prepared audio content/clock differs")
    require(train == [r for r in rows if r["split"] == "train"]
            and validation == [r for r in rows if r["split"] == "validation"]
            and len(train) + len(validation) == len(rows), "Split lists differ or omit rows")
    require(metadata["counts"] == dict(Counter(r["split"] for r in rows)), "Prepared split counts differ")
    checked = [dict(r, requested_split=r["split"]) for r in rows]
    count = _components(checked, metadata["splitseed"], metadata["validation_fraction"])
    require(all(a["component_id"] == b["component_id"] for a, b in zip(checked, rows))
            and count == metadata["connected_components"], "Prepared leakage components differ")
    return dict(metadata=metadata, rows=rows, train=train, validation=validation)


def audit(path) -> dict:
    data = load_prepared(path)
    rows = data["rows"]
    return dict(schema="inflect-v3-data-audit-v1", passed=True, clips=len(rows),
                counts=dict(Counter(r["split"] for r in rows)),
                sample_rate=SAMPLE_RATE, samples_per_frame=HOP,
                audio_seconds=sum(r["sample_count"] for r in rows) / SAMPLE_RATE,
                connected_components=data["metadata"]["connected_components"],
                unknown_phones=[], supervised_alignments=True, generated_duration_labels=False,
                source_and_consent="Recorded user assertions, not independent rights verification")
