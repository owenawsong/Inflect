"""Public CLI negative/delegation contracts without loading a Mini model."""
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from finetune_v3 import cli, data, frontend, model, training


@pytest.mark.parametrize("argv", [[], ["unknown-command"], ["prepare"],
    ["train", "--model", "x", "--dataset", "y", "--output", "z", "--device", "auto"],
    ["smoke-dataset", "--model", "x", "--output", "y"],
    ["say", "--model", "x", "--output", "out.wav"],
    ["say", "--model", "x", "--output", "out.wav", "--text", "hello", "--phonemes", "a"],
    ["coverage", "--manifest", "x", "--vocab", "v", "--output", "o"],
])
def test_parser_rejects_ambiguous_or_unacknowledged_requests(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_prepare_delegates_explicit_frontend_and_split_seed(tmp_path, monkeypatch, capsys):
    seen = []
    def prepare(*args, **kwargs):
        seen.append((args, kwargs))
        return tmp_path / "prepared/dataset.json"
    monkeypatch.setattr(data, "prepare", prepare)
    assert cli.main(["prepare", "--manifest", "manifest.jsonl", "--audio-root", "raw",
        "--output", "prepared", "--vocab", "vocab.json", "--language", "fr-fr",
        "--frontend", "prephonemized", "--split-seed", "19", "--validation-fraction", ".25"]) == 0
    assert seen == [(("manifest.jsonl", "raw", "prepared", "fr-fr", "prephonemized", "vocab.json", 19),
                     {"validation_fraction": .25, "phone_mapping": None, "base_vocab": None,
                      "allow_resample": False, "clock_tolerance_samples": 0})]
    assert "dataset.json" in capsys.readouterr().out


def test_prepare_forwards_explicit_mapping_vocab_resample_and_clock_flags(tmp_path, monkeypatch):
    seen = []
    def prepare(*args, **kwargs):
        seen.append((args, kwargs))
        return tmp_path / "prepared/dataset.json"
    monkeypatch.setattr(data, "prepare", prepare)
    assert cli.main(["prepare", "--manifest", "labels.jsonl", "--audio-root", "raw",
        "--output", "prepared", "--vocab", "extended-vocab.json", "--language", "fr-fr",
        "--frontend", "espeak", "--phone-mapping", "fr-map.json", "--base-vocab", "base-vocab.json",
        "--allow-resample", "--clock-tolerance-samples", "120"]) == 0
    args, kwargs = seen[0]
    assert args == ("labels.jsonl", "raw", "prepared", "fr-fr", "espeak", "extended-vocab.json", 2026)
    assert kwargs == {"validation_fraction": .1, "phone_mapping": "fr-map.json",
                      "base_vocab": "base-vocab.json", "allow_resample": True, "clock_tolerance_samples": 120}
    assert type(kwargs["allow_resample"]) is bool
    assert type(kwargs["clock_tolerance_samples"]) is int


@pytest.mark.parametrize("error", [ValueError("alignment refused"), FileNotFoundError("missing source"), FileExistsError("preserve output")])
def test_runtime_errors_have_nonzero_exit_and_no_success_json(monkeypatch, capsys, error):
    def audit(path):
        raise error
    monkeypatch.setattr(data, "audit", audit)
    assert cli.main(["audit", "--dataset", "not-opened"]) == 2
    captured = capsys.readouterr()
    assert "Error: " in captured.err and captured.out == ""


def test_train_forwards_groups_resume_and_migration_without_loading_model(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(training, "train", lambda options: seen.append(options) or Path("fake-run"))
    assert cli.main(["train", "--model", "base", "--dataset", "dataset", "--output", "run",
        "--groups", "duration", "speaker", "--resume", "step.pt", "--vocab-migration", "migration.json",
        "--steps", "12", "--device", "cpu"]) == 0
    options = seen[0]
    assert options.groups == ("duration", "speaker") and options.resume == "step.pt"
    assert options.vocab_migration == "migration.json" and options.steps == 12 and options.device == "cpu"
    assert json.loads(capsys.readouterr().out)["run"] == "fake-run"


@pytest.fixture
def say_fixture(tmp_path, monkeypatch):
    root = tmp_path / "export"
    root.mkdir()
    config = {"target_voice": "cassidy", "frontend": {"mode": "prephonemized", "language": "fr-fr"},
              "vocabulary": {"a": 1}, "automatic_english_fallback": False}
    (root / "frontend.json").write_text(json.dumps(config), encoding="utf-8")
    calls = []
    def synthesize(phones, voice, *, seed):
        calls.append((phones, voice, seed))
        return {"audio": torch.zeros(1200), "sample_rate": 24000, "frames": 2}
    monkeypatch.setattr(model, "load_public_runtime", lambda path: SimpleNamespace(synthesize_phonemes=synthesize))
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    args = Namespace(model=str(root), voice=None, phonemes="a", text=None,
                     output=str(tmp_path / "sample.wav"), seed=42)
    return root, config, args, calls


def test_say_never_silently_applies_english_frontend(say_fixture):
    root, config, args, calls = say_fixture
    args.text, args.phonemes = "bonjour", None
    with pytest.raises(ValueError, match="no implicit English frontend"):
        cli.say(args)
    assert calls == [] and not Path(args.output).exists()


def test_say_rejects_unqualified_voice_slot(say_fixture):
    root, config, args, calls = say_fixture
    args.voice = "hope"
    with pytest.raises(ValueError, match="not qualified"):
        cli.say(args)
    assert calls == []


@pytest.mark.parametrize("where", ["existing", "inside", "missing-parent"])
def test_say_output_admission_precedes_synthesis(say_fixture, where, tmp_path):
    root, config, args, calls = say_fixture
    if where == "existing":
        path = Path(args.output)
        path.write_bytes(b"preserve me")
    elif where == "inside":
        args.output = str(root / "forbidden.wav")
    else:
        args.output = str(tmp_path / "absent/sample.wav")
    with pytest.raises(ValueError, match="new output"):
        cli.say(args)
    assert calls == []
    if where == "existing":
        assert path.read_bytes() == b"preserve me"


def test_say_explicit_phonemes_seed_and_default_target_voice(say_fixture):
    root, config, args, calls = say_fixture
    args.seed = 73
    result = cli.say(args)
    assert calls == [("a", "cassidy", 73)]
    assert result["sample_rate"] == 24000 and Path(args.output).is_file()


def test_coverage_reports_unknowns_without_inventing_alignment(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "foreign", "text": "Bonjour", "phones": "a\u0250\u0250"}) + "\n", encoding="utf-8")
    vocabulary = tmp_path / "vocab.json"
    vocabulary.write_text('{"a": 1}', encoding="utf-8")
    output = tmp_path / "coverage.json"
    assert cli.main(["coverage", "--manifest", str(manifest), "--vocab", str(vocabulary),
        "--language", "fr-fr", "--frontend", "prephonemized", "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["unknown_codepoints"] == {"\u0250": 2}
    assert result["status"] == "explicit-vocabulary-migration-required"
    assert result["alignment_generated"] is False and result["language_quality_approved"] is False
    assert result["examples"][0]["phone_ids"] is None
    assert result["frontend"]["phone_mapping"] is None
    assert result["frontend"]["base_vocab_sha256"] is None


def test_coverage_flags_apply_mapping_and_validate_base_vocab(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"mapped","text":"Bonjour","phones":"x"}\n', encoding="utf-8")
    vocabulary = tmp_path / "vocab.json"
    vocabulary.write_text('{"a":1,"b":100}', encoding="utf-8")
    base = tmp_path / "base.json"
    base.write_text('{"a":1}', encoding="utf-8")
    mapping = tmp_path / "map.json"
    rules = {"language": "fr-fr", "name": "explicit-test-mapping", "replacements": {"x": "a"}}
    mapping.write_text(json.dumps(rules), encoding="utf-8")
    output = tmp_path / "coverage.json"
    assert cli.main(["coverage", "--manifest", str(manifest), "--vocab", str(vocabulary),
        "--language", "fr-fr", "--frontend", "prephonemized", "--output", str(output),
        "--phone-mapping", str(mapping), "--base-vocab", str(base)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "coverage-passed" and result["unknown_codepoints"] == {}
    assert result["frontend"]["phone_mapping"] == rules
    assert result["frontend"]["base_vocab_sha256"] == frontend.canonical_sha({"a": 1})
    assert result["examples"][0]["original_phones"] == "x"
    assert result["examples"][0]["phones"] == "a"


def test_coverage_will_not_overwrite_existing_report(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"text":"test","phones":"a"}\n', encoding="utf-8")
    vocabulary = tmp_path / "vocab.json"
    vocabulary.write_text('{"a":1}', encoding="utf-8")
    output = tmp_path / "coverage.json"
    output.write_bytes(b"original evidence")
    assert cli.main(["coverage", "--manifest", str(manifest), "--vocab", str(vocabulary),
        "--language", "fr-fr", "--frontend", "prephonemized", "--output", str(output)]) == 2
    assert output.read_bytes() == b"original evidence"


def test_export_rejects_checkpoint_from_another_run_before_model_load(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "run-contract.json").write_text("{}", encoding="utf-8")
    foreign = tmp_path / "foreign.pt"
    foreign.write_bytes(b"tiny fixture, not a model")
    def forbidden(*args, **kwargs):
        pytest.fail("Model loaded before checkpoint admission")
    monkeypatch.setattr(model, "load_mini", forbidden)
    with pytest.raises(ValueError, match="belonging to this run"):
        cli.export_run(root, foreign, tmp_path / "export")


@pytest.fixture
def export_fixture(tmp_path):
    root = tmp_path / "bound-run"
    (root / "checkpoints").mkdir(parents=True)
    point = root / "checkpoints/step-0000007.pt"
    point.write_bytes(b"fake checkpoint")
    contract = {"model": "not-a-real-model", "groups": ["duration"], "voice": "cassidy",
                "dataset_sha256": "pinned-dataset", "frontend": {"mode": "prephonemized", "language": "fr-fr"}}
    (root / "run-contract.json").write_text(json.dumps(contract), encoding="utf-8")
    import hashlib
    state = {"step": 7, "checkpoint_sha256": training.sha256(point),
             "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()}
    point.with_suffix(".controller.json").write_text(json.dumps(state), encoding="utf-8")
    return root, point, contract, state, tmp_path / "output"


@pytest.mark.parametrize("tamper", ["bytes", "contract", "receipt", "missing-receipt"])
def test_export_controller_admission_precedes_model_and_output(export_fixture, monkeypatch, tamper):
    root, point, contract, state, output = export_fixture
    if tamper == "bytes":
        point.write_bytes(b"changed weights")
    elif tamper == "contract":
        contract["voice"] = "hope"
        (root / "run-contract.json").write_text(json.dumps(contract), encoding="utf-8")
    elif tamper == "receipt":
        state["contract_sha256"] = "0" * 64
        point.with_suffix(".controller.json").write_text(json.dumps(state), encoding="utf-8")
    else:
        other = point.with_name("orphan.pt")
        other.write_bytes(point.read_bytes())
        point = other
    def forbidden(*args, **kwargs):
        pytest.fail("Export reached model work before validating receipt")
    monkeypatch.setattr(model, "load_mini", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.export_run(root, point, output)
    assert not output.exists()


def test_export_checks_restored_step_before_writing_output(export_fixture, monkeypatch):
    from finetune_v3 import checkpoint, notices
    root, point, contract, state, output = export_fixture
    monkeypatch.setattr(notices, "source_notices", lambda path: (root, {}))
    monkeypatch.setattr(model, "load_mini", lambda *args, **kwargs: SimpleNamespace(vocab={"a": 1}))
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *args, **kwargs: {"step": 8})
    def forbidden(*args, **kwargs):
        pytest.fail("Export output written before restored step check")
    monkeypatch.setattr(checkpoint, "export_model", forbidden)
    with pytest.raises(ValueError, match="Controller step differs"):
        cli.export_run(root, point, output)
    assert not output.exists()


def test_valid_export_calls_controller_validator_before_model(export_fixture, monkeypatch):
    from finetune_v3 import checkpoint, notices
    root, point, contract, state, output = export_fixture
    order = []
    original_validator = training.validate_controller_checkpoint
    def validate(*args):
        result = original_validator(*args)
        order.append("controller")
        return result
    def load(*args, **kwargs):
        order.append("model")
        return SimpleNamespace(vocab={"a": 1}, eval=lambda: None)
    def export(*args):
        order.append("export")
        output.mkdir()
        (output / "manifest.json").write_text('{"files":{}}', encoding="utf-8")
        return {"fixture": True}
    monkeypatch.setattr(training, "validate_controller_checkpoint", validate)
    monkeypatch.setattr(notices, "source_notices", lambda path: (root, {}))
    monkeypatch.setattr(notices, "retain_notices", lambda *args: {})
    monkeypatch.setattr(model, "load_mini", load)
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *args, **kwargs: {"step": 7})
    monkeypatch.setattr(checkpoint, "export_model", export)
    result = cli.export_run(root, point, output)
    assert order == ["controller", "model", "export"]
    descriptor = json.loads((output / "frontend.json").read_text(encoding="utf-8"))
    inventory = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert descriptor["frontend"] == contract["frontend"]
    assert inventory["files"]["frontend.json"] == result["frontend_sha256"]


@pytest.mark.parametrize("recorded,runtime", [(None, "test-engine"), ("old-engine", "new-engine"), ("test-engine", None)])
def test_say_raw_text_rejects_missing_or_mismatched_espeak_version(say_fixture, monkeypatch, recorded, runtime):
    root, config, args, calls = say_fixture
    config["frontend"] = {"mode": "espeak", "language": "fr-fr", "espeak_version": recorded}
    (root / "frontend.json").write_text(json.dumps(config), encoding="utf-8")
    args.text, args.phonemes = "Bonjour", None
    class FakeFrontend:
        def __init__(self, *args, **kwargs):
            self.encoded = False
        def encode(self, text):
            self.encoded = True
            return {"phones": "a"}
        def metadata(self):
            assert self.encoded, "Query engine version only after lazy frontend initialization"
            return {"espeak_version": runtime}
    monkeypatch.setattr(frontend, "Frontend", FakeFrontend)
    with pytest.raises(ValueError, match="eSpeak engine version differs"):
        cli.say(args)
    assert calls == [] and not Path(args.output).exists()


def test_say_raw_text_accepts_exact_recorded_engine(say_fixture, monkeypatch):
    root, config, args, calls = say_fixture
    config["frontend"] = {"mode": "espeak", "language": "fr-fr", "espeak_version": "test-engine"}
    (root / "frontend.json").write_text(json.dumps(config), encoding="utf-8")
    args.text, args.phonemes = "Bonjour", None
    monkeypatch.setattr(frontend, "Frontend", lambda *args, **kwargs: SimpleNamespace(
        encode=lambda text: {"phones": "a"}, metadata=lambda: {"espeak_version": "test-engine"}))
    assert cli.say(args)["sample_rate"] == 24000
    assert calls == [("a", "cassidy", 42)]
