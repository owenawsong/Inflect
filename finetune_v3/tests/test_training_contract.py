"""Small contract tests: fake model/checkpoints, real admission and controller."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from finetune_v3 import data, training


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))
        self.vocab = {"a": 1}
        self.source_identity = {"weights_sha256": "fake-source-identity"}

    def phoneme_ids(self, phones):
        if any(phone not in self.vocab for phone in phones):
            raise ValueError("Unknown phone")
        return [0, *(self.vocab[p] for p in phones), 0]

    def trainability_report(self):
        return {"fake_trainable_parameters": 1}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    source = tmp_path / "model"
    dataset = tmp_path / "dataset"
    source.mkdir()
    dataset.mkdir()
    metadata = {"vocab": {"a": 1}, "frontend": {"mode": "prephonemized", "language": "fr-fr"}}
    rows = {"train": [{"id": "train-one", "_phones": "a", "_durations": [1, 2, 1], "target": .1}],
            "validation": [{"id": "validation-only", "_phones": "a", "_durations": [1, 2, 1], "target": .4}]}
    state = SimpleNamespace(loads=[], forwards=[], restored=[], metadata=metadata, rows=rows,
                            identity="dataset-content-pin", source=source, dataset=dataset)

    def load_model(path, trainable_groups):
        state.loads.append((str(path), trainable_groups))
        return TinyModel()

    def load_rows(path):
        return state.metadata, state.rows, state.identity

    def loss(model, batch, options, seed):
        state.forwards.append((batch["id"], model.training, seed, torch.is_grad_enabled()))
        value = (model.weight - batch["target"]).square()
        return value, {"duration": float(value.detach()), "spectral": 0.}

    def save(model, path, *, step, optimizer):
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump({"step": step, "weight": float(model.weight.detach())}, stream)

    def restore(model, path, *, optimizer):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        state.restored.append(str(path))
        with torch.no_grad():
            model.weight.fill_(payload["weight"])
        return {"step": payload["step"]}

    monkeypatch.setattr(training, "load_mini", load_model)
    monkeypatch.setattr(training, "load_rows", load_rows)
    monkeypatch.setattr(training, "make_batch", lambda model, row, *args: row)
    monkeypatch.setattr(training, "losses", loss)
    monkeypatch.setattr(training, "save_checkpoint", save)
    monkeypatch.setattr(training, "load_checkpoint", restore)
    monkeypatch.setattr(training.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * 2**30))
    monkeypatch.setattr(training.torch, "set_num_threads", lambda threads: None)
    state.options = training.TrainOptions(str(source), str(dataset), str(tmp_path / "run"),
        steps=1, checkpoint_every=1, validation_every=1, max_frames=8, spectral_weight=0., seed=17)
    return state


@pytest.mark.parametrize("change", [
    {"steps": 0}, {"gradient_accumulation": 0}, {"learning_rate": float("nan")},
    {"learning_rate": .01}, {"max_frames": 1201}, {"threads": 0},
    {"checkpoint_every": 0}, {"validation_every": 0}, {"max_validation_clips": 0},
    {"duration_weight": 0., "spectral_weight": 0.}, {"duration_weight": float("inf")},
    {"voice": "unknown"}, {"device": "unavailable"},
])
def test_invalid_options_refused_before_model_or_run(harness, change):
    with pytest.raises(ValueError):
        training.train(replace(harness.options, **change))
    assert harness.loads == []
    assert not Path(harness.options.output).exists()


@pytest.mark.parametrize("target", ["source", "dataset"])
def test_run_output_cannot_live_inside_inputs(harness, target):
    output = getattr(harness, target) / "do-not-write"
    with pytest.raises(ValueError, match="outside"):
        training.train(replace(harness.options, output=str(output)))
    assert not output.exists() and harness.loads == []


def test_requested_cuda_never_falls_back_to_cpu(harness, monkeypatch):
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no silent CPU fallback"):
        training.train(replace(harness.options, device="cuda"))
    assert harness.loads == []


@pytest.mark.parametrize("failure", ["vocab", "coverage", "frame_cap"])
def test_data_admission_precedes_run_creation(harness, failure):
    if failure == "vocab":
        harness.metadata["vocab"] = {"b": 2}
    elif failure == "coverage":
        harness.rows["validation"][0]["_phones"] = "b"
    else:
        harness.rows["validation"][0]["_durations"] = [1, 8, 1]
    with pytest.raises(ValueError):
        training.train(harness.options)
    assert not Path(harness.options.output).exists()
    assert harness.forwards == []


def test_existing_run_is_never_overwritten(harness):
    root = Path(harness.options.output)
    root.mkdir()
    sentinel = root / "preserved-evidence.txt"
    sentinel.write_bytes(b"do not remove")
    with pytest.raises(FileExistsError):
        training.train(harness.options)
    assert sentinel.read_bytes() == b"do not remove"


def test_low_disk_holds_and_preserves_run_evidence(harness, monkeypatch):
    monkeypatch.setattr(training.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    with pytest.raises(ValueError, match="1.5 GiB"):
        training.train(harness.options)
    root = Path(harness.options.output)
    assert (root / "run-contract.json").is_file()
    assert harness.forwards == []
    assert list((root / "checkpoints").iterdir()) == []


def test_tiny_controller_separates_validation_and_binds_checkpoint(harness):
    options = replace(harness.options, gradient_accumulation=2)
    root = training.train(options)
    assert harness.forwards == [("train-one", True, 17, True), ("train-one", True, 18, True),
                                ("validation-only", False, 17, False)]
    point = root / "checkpoints/step-0000001.pt"
    controller = json.loads(point.with_suffix(".controller.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "run-contract.json").read_text(encoding="utf-8"))
    assert controller["checkpoint_sha256"] == hashlib.sha256(point.read_bytes()).hexdigest()
    assert controller["contract_sha256"] == hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    assert contract["validation_is_not_training_data"] is True
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete" and status["language_quality_approved"] is False
    assert status["public_weights_changed"] is False


def test_resume_accepts_only_allowed_cadence_and_final_step_changes(harness):
    root = training.train(harness.options)
    point = root / "checkpoints/step-0000001.pt"
    resumed = replace(harness.options, resume=str(point), steps=2,
                      checkpoint_every=3, validation_every=3, max_validation_clips=1)
    assert training.train(resumed) == root
    assert harness.restored == [str(point.resolve())]
    assert (root / "checkpoints/step-0000002.pt").is_file()


@pytest.mark.parametrize("change", [{"voice": "hope"}, {"seed": 18},
    {"learning_rate": 2e-5}, {"gradient_accumulation": 2}, {"duration_weight": 2.}])
def test_resume_rejects_changed_training_contract(harness, change):
    root = training.train(harness.options)
    point = root / "checkpoints/step-0000001.pt"
    before = point.read_bytes()
    with pytest.raises(ValueError, match="Resume contract differs"):
        training.train(replace(harness.options, resume=str(point), steps=2, **change))
    assert point.read_bytes() == before and harness.restored == []


def test_resume_rejects_changed_dataset_identity(harness):
    root = training.train(harness.options)
    harness.identity = "changed-content-pin"
    with pytest.raises(ValueError, match="Resume contract differs"):
        training.train(replace(harness.options, resume=str(root / "checkpoints/step-0000001.pt"), steps=2))
    assert harness.restored == []


def test_resume_rejects_foreign_checkpoint_even_with_identical_bytes(harness, tmp_path):
    root = training.train(harness.options)
    foreign = tmp_path / "foreign.pt"
    foreign.write_bytes((root / "checkpoints/step-0000001.pt").read_bytes())
    with pytest.raises(ValueError, match="belong to this run"):
        training.train(replace(harness.options, resume=str(foreign), steps=2))
    assert harness.restored == []


@pytest.mark.parametrize("tamper", ["checkpoint", "contract_hash", "checkpoint_hash", "step"])
def test_resume_rejects_controller_checkpoint_tampering(harness, tamper):
    root = training.train(harness.options)
    point = root / "checkpoints/step-0000001.pt"
    controller = point.with_suffix(".controller.json")
    if tamper == "checkpoint":
        point.write_bytes(point.read_bytes() + b" ")
    else:
        state = json.loads(controller.read_text(encoding="utf-8"))
        key = {"contract_hash": "contract_sha256", "checkpoint_hash": "checkpoint_sha256", "step": "step"}[tamper]
        state[key] = 9 if tamper == "step" else "0" * 64
        write_json(controller, state)
    with pytest.raises(ValueError, match="identity mismatch|Controller step differs"):
        training.train(replace(harness.options, resume=str(point), steps=2))
    assert not (root / "checkpoints/step-0000002.pt").exists()


def test_resume_requires_work_beyond_restored_step(harness):
    root = training.train(harness.options)
    with pytest.raises(ValueError, match="exceed the restored step"):
        training.train(replace(harness.options, resume=str(root / "checkpoints/step-0000001.pt")))


def test_make_batch_preserves_explicit_boundary_alignment(tmp_path):
    path = tmp_path / "tiny.wav"
    sf.write(path, np.zeros(2400, dtype=np.float32), 24000, subtype="FLOAT")
    row = {"_phones": "a", "_durations": [1, 2, 1], "_audio_path": str(path)}
    ids, lengths, alignment, sid, durations, target = training.make_batch(TinyModel(), row, "cassidy", "cpu", 4)
    assert ids.tolist() == [[0, 1, 0]] and lengths.tolist() == [3]
    assert alignment.shape == (1, 3, 4)
    assert alignment.argmax(1).tolist() == [[0, 1, 1, 2]]
    assert alignment.sum(2).tolist() == durations.tolist() == [[1., 2., 1.]]
    assert target.shape == (1, 2400) and sid.tolist() == [2]
    with pytest.raises(ValueError, match="never crop silently"):
        training.make_batch(TinyModel(), row, "cassidy", "cpu", 3)


def test_load_rows_requires_audit_before_reading_payload(tmp_path, monkeypatch):
    def rejected(path):
        raise ValueError("auditor refused payload")
    monkeypatch.setattr(data, "load_prepared", rejected)
    with pytest.raises(ValueError, match="auditor refused"):
        training.load_rows(tmp_path)


def test_waveform_loss_refuses_clock_mismatch():
    with pytest.raises(ValueError, match="clocks must match"):
        training.spectral_loss(torch.zeros(1, 1200), torch.zeros(1, 1800))


@pytest.fixture
def controller_fixture(tmp_path):
    run = tmp_path / "controller-run"
    (run / "checkpoints").mkdir(parents=True)
    point = run / "checkpoints/step-0000007.pt"
    point.write_bytes(b"fake checkpoint payload")
    contract = {"voice": "cassidy", "frontend": {"language": "fr-fr", "phone": "\u0281"},
                "dataset_sha256": "fixed-dataset-pin"}
    state = {"step": 7, "checkpoint_sha256": training.sha256(point),
             "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()}
    write_json(point.with_suffix(".controller.json"), state)
    return run, point, contract, state


def test_controller_validation_canonicalizes_contract_key_order(controller_fixture):
    run, point, contract, state = controller_fixture
    reordered = dict(reversed(list(contract.items())))
    assert training.validate_controller_checkpoint(run, point, reordered) == state


@pytest.mark.parametrize("step", [-1, True, 7., "7", None])
def test_controller_validation_requires_nonnegative_integer_step(controller_fixture, step):
    run, point, contract, state = controller_fixture
    state["step"] = step
    write_json(point.with_suffix(".controller.json"), state)
    with pytest.raises(ValueError, match="identity mismatch"):
        training.validate_controller_checkpoint(run, point, contract)


@pytest.mark.parametrize("tamper", ["bytes", "contract", "missing-sidecar", "checkpoint-pin", "contract-pin"])
def test_controller_validation_refuses_unbound_receipts(controller_fixture, tamper):
    run, point, contract, state = controller_fixture
    if tamper == "bytes":
        point.write_bytes(b"replacement checkpoint")
    elif tamper == "contract":
        contract["voice"] = "hope"
    elif tamper == "missing-sidecar":
        # Preserve the existing receipt; choose another file without one.
        other = point.with_name("uncommitted.pt")
        other.write_bytes(point.read_bytes())
        point = other
    else:
        key = "checkpoint_sha256" if tamper == "checkpoint-pin" else "contract_sha256"
        state[key] = "0" * 64
        write_json(point.with_suffix(".controller.json"), state)
    with pytest.raises((ValueError, FileNotFoundError)):
        training.validate_controller_checkpoint(run, point, contract)


@pytest.mark.parametrize("orphan", ["weights", "sidecar", "both"])
def test_checkpoint_collision_preserves_orphan_and_commits_resumable_recovery(harness, orphan):
    root = training.train(harness.options)
    first = root / "checkpoints/step-0000001.pt"
    collision = root / "checkpoints/step-0000002.pt"
    receipt = collision.with_suffix(".controller.json")
    if orphan in ("weights", "both"):
        collision.write_bytes(b"preserved orphan checkpoint")
    if orphan in ("sidecar", "both"):
        receipt.write_bytes(b"preserved orphan controller")
    training.train(replace(harness.options, resume=str(first), steps=2))
    latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    recovery = Path(latest["checkpoint"])
    assert recovery.parent == collision.parent
    assert recovery.name.startswith("step-0000002-recovery-") and recovery.suffix == ".pt"
    assert recovery.is_file() and recovery.with_suffix(".controller.json").is_file()
    if orphan in ("weights", "both"):
        assert collision.read_bytes() == b"preserved orphan checkpoint"
    if orphan in ("sidecar", "both"):
        assert receipt.read_bytes() == b"preserved orphan controller"
    contract = json.loads((root / "run-contract.json").read_text(encoding="utf-8"))
    assert training.validate_controller_checkpoint(root, recovery, contract)["step"] == latest["step"] == 2
    training.train(replace(harness.options, resume=str(recovery), steps=3))
    assert harness.restored == [str(first.resolve()), str(recovery.resolve())]


def test_requested_threads_are_restored_after_native_loader_resets_them(harness, monkeypatch):
    requested = 3
    calls = []
    original_loader = training.load_mini
    original_losses = training.losses
    def set_threads(count):
        calls.append(count)
    def native_loader(*args, **kwargs):
        # Reproduce the published loader's CPU thread policy without loading it.
        training.torch.set_num_threads(2)
        return original_loader(*args, **kwargs)
    def checked_losses(*args, **kwargs):
        assert calls[-1] == requested
        return original_losses(*args, **kwargs)
    monkeypatch.setattr(training.torch, "set_num_threads", set_threads)
    monkeypatch.setattr(training, "load_mini", native_loader)
    monkeypatch.setattr(training, "losses", checked_losses)
    training.train(replace(harness.options, threads=requested))
    assert calls == [requested, 2, requested]
