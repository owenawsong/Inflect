"""CPU integration tests against an existing local public Mini model-core."""

import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from finetune_v3 import export_model, load_checkpoint, load_mini, load_public_runtime, save_checkpoint
from finetune_v3.model import COUNTS, GROUPS, PUBLIC_WEIGHTS_SHA256, VOICES, sha256, validate_inventory


ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("INFLECT_V3_TEST_MODEL", str(
    ROOT / "outputs/events-public-sync-20261111-r001/package/verified-download/core/baseline/models/mini")))
pytestmark = pytest.mark.skipif(not MODEL.is_dir() and 'INFLECT_V3_TEST_MODEL' not in os.environ,
                                reason='Optional model integration: set INFLECT_V3_TEST_MODEL; no download fallback')


@pytest.fixture(scope="module")
def model():
    if not MODEL.is_dir():
        pytest.fail("Local public Mini package missing; set INFLECT_V3_TEST_MODEL (no download fallback)")
    torch.set_num_threads(2)
    return load_mini(MODEL, expected_weights_sha256=PUBLIC_WEIGHTS_SHA256).eval()


@pytest.fixture(autouse=True)
def reset_mode(model):
    model.configure_trainable(())
    model.eval()
    yield
    model.configure_trainable(())
    model.eval()


def inputs(model, *, voice=2):
    ids = torch.tensor([[0, model.vocab["h"], model.vocab["a"], 0]], dtype=torch.long)
    lengths = torch.tensor([ids.shape[1]], dtype=torch.long)
    sid = torch.tensor([voice], dtype=torch.long)
    # Explicit synthetic fixture labels, not a language/dataset alignment policy.
    aln = torch.eye(ids.shape[1], dtype=torch.float32).repeat_interleave(2, dim=1)[None]
    return ids, lengths, aln, sid


def test_exact_public_inventory_and_parameter_counts(model):
    _, identity = validate_inventory(MODEL, PUBLIC_WEIGHTS_SHA256)
    assert identity == model.source_identity
    assert {n: sum(p.numel() for p in getattr(model, n).parameters()) for n in COUNTS} == COUNTS
    assert sum(p.numel() for p in model.parameters()) == 34_310_790
    assert not any(p.requires_grad for p in model.parameters())


def test_trainable_groups_exhaustive_and_source_excitation_excluded(model):
    report = model.configure_trainable(GROUPS)
    assert sum(row["parameters"] for row in report["groups"].values()) == 34_310_790
    assert all(p.requires_grad for p in model.text.parameters())
    assert not any(p.requires_grad for p in model.generator.m_source.parameters())
    assert report["harmonic_excitation_differentiable"] is False
    with pytest.raises(ValueError, match="Invalid trainable"):
        model.configure_trainable(("vits",))


def test_unknown_phones_and_no_silent_id_expansion(model):
    assert model.phoneme_ids("ha") == [0, model.vocab["h"], model.vocab["a"], 0]
    with pytest.raises(ValueError, match="Unknown phonemes"):
        model.phoneme_ids("\u2603")


@pytest.mark.parametrize("voice", VOICES)
def test_all_six_voice_public_waveform_parity(model, voice):
    reference = load_public_runtime(MODEL, expected_weights_sha256=PUBLIC_WEIGHTS_SHA256)
    ids = model.phoneme_ids("h\u0259l\u02c8o\u028a")
    before = torch.random.get_rng_state().clone()
    expected = reference.synthesize_ids(ids, voice=voice, seed=42)
    actual = model.synthesize_ids(ids, voice=voice, seed=42)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert actual["frames"] == expected["frames"]
    assert torch.equal(actual["audio"], expected["audio"])
    assert actual["audio"].numel() == 600 * actual["frames"]
    assert not any(fc._forward_hooks for fc in model.generator.modules())


def test_forward_grad_shapes_and_feature_gradients(model):
    model.configure_trainable(("text_encoder", "duration", "pitch_noise", "speaker"))
    model.train()
    ids, lengths, aln, sid = inputs(model)
    result = model.forward_grad(ids, lengths, aln, sid, seed=17, decode=False)
    assert result["dur"].shape == ids.shape
    assert result["asr"].shape == (1, 512, 8)
    assert result["f0"].shape == result["noise"].shape == (1, 16)
    loss = result["dur"].square().mean() + result["asr"].square().mean()
    loss = loss + result["f0"].square().mean() / 10_000 + result["noise"].square().mean()
    loss.backward()
    nonzero = {model.parameter_group(name) for name, p in model.named_parameters()
               if p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum() > 0)}
    assert {"text_encoder", "duration", "pitch_noise", "speaker"} <= nonzero
    assert all(p.grad is None for p in model.bridge.parameters())
    assert all(p.grad is None for p in model.generator.parameters())


def test_frozen_acoustics_pass_waveform_stft_gradients_and_seed(model):
    model.configure_trainable(("text_encoder", "pitch_noise"))
    model.train()
    args = inputs(model)
    before = torch.random.get_rng_state().clone()
    first = model.forward_grad(*args, seed=42)
    assert torch.equal(torch.random.get_rng_state(), before)
    second = model.forward_grad(*args, seed=42)
    assert torch.equal(first["audio"], second["audio"])
    assert first["audio"].shape == (1, 4800)
    loss = sum(torch.stft(first["audio"], n_fft=n, hop_length=n // 4,
                          window=torch.hann_window(n), return_complex=True).abs().mean()
               for n in (64, 128))
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in model.text.text_encoder.parameters())
    assert all(p.grad is None for p in model.generator.parameters())
    assert all(p.grad is None for p in model.bridge.parameters())


def test_trainable_bridge_and_generator_receive_gradients(model):
    model.configure_trainable(("bridge", "generator"))
    model.train()
    result = model.forward_grad(*inputs(model), seed=42)
    result["audio"].square().mean().backward()
    for module in (model.bridge, model.generator):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
        assert any(bool(g.abs().sum() > 0) for g in gradients)


@pytest.mark.parametrize("bad", ("padding", "soft", "uncovered", "nonmonotonic", "nonfinite", "token", "sid", "batch"))
def test_invalid_training_inputs_fail_closed(model, bad):
    ids, lengths, aln, sid = inputs(model)
    if bad == "padding":
        lengths[0] -= 1
    elif bad == "soft":
        aln[:, 0, 0] = 0.5
    elif bad == "uncovered":
        aln[:, 1] += aln[:, 0]
        aln[:, 0] = 0
    elif bad == "nonmonotonic":
        aln = aln.flip(2)
    elif bad == "nonfinite":
        aln[0, 0, 0] = float("nan")
    elif bad == "token":
        ids[0, 1] = 178
    elif bad == "sid":
        sid[0] = 6
    elif bad == "batch":
        ids, lengths, aln, sid = (v.repeat((2,) + (1,) * (v.ndim - 1)) for v in (ids, lengths, aln, sid))
    with pytest.raises(ValueError):
        model.forward_grad(ids, lengths, aln, sid)


def test_checkpoint_roundtrip_optimizer_and_public_export(model, tmp_path):
    model.configure_trainable(("speaker",))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    # A tiny CPU unit fixture update, not a dataset training run or quality claim.
    original = model.text.speaker_embedding.weight.detach().clone()
    try:
        model.text.speaker_embedding.weight.square().mean().backward()
        optimizer.step()
        expected = model.text.speaker_embedding.weight.detach().clone()
        checkpoint = save_checkpoint(model, tmp_path / "adapt.pt", step=1, optimizer=optimizer)
        with torch.no_grad():
            model.text.speaker_embedding.weight.zero_()
        receipt = load_checkpoint(model, checkpoint, optimizer=optimizer)
        assert receipt["step"] == 1
        assert torch.equal(model.text.speaker_embedding.weight, expected)
        assert optimizer.state
        model.eval()
        ids = model.phoneme_ids("ha")
        expected_audio = {voice: model.synthesize_ids(ids, voice=voice, seed=42)["audio"] for voice in VOICES}
        export = export_model(model, tmp_path / "export")
        assert export["public_loader_strict_readback"]
        deployed = load_public_runtime(tmp_path / "export")
        imported = load_mini(tmp_path / "export").eval()
        for voice in VOICES:
            assert torch.equal(expected_audio[voice], deployed.synthesize_ids(ids, voice=voice, seed=42)["audio"])
            assert torch.equal(expected_audio[voice], imported.synthesize_ids(ids, voice=voice, seed=42)["audio"])
        assert sha256(MODEL / "weights.pt") == PUBLIC_WEIGHTS_SHA256
        assert validate_inventory(MODEL)[1] == model.source_identity
    finally:
        with torch.no_grad():
            model.text.speaker_embedding.weight.copy_(original)


def test_checkpoint_rejects_v2_and_identity_before_mutation(model, tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"format": "inflect_vits_inference_checkpoint_v1", "model": {}}, path)
    before = model.text.speaker_embedding.weight.detach().clone()
    with pytest.raises(ValueError, match="v2 VITS"):
        load_checkpoint(model, path)
    assert torch.equal(before, model.text.speaker_embedding.weight)


def test_fixed_conditioning_cannot_silently_change(model):
    previous = model.vocab["h"]
    try:
        model.vocab["h"] = 177
        with pytest.raises(ValueError, match="Fixed vocabulary"):
            model.inference_payload()
    finally:
        model.vocab["h"] = previous


def test_fixed_conditioning_replacement_rejected(model):
    original = model.vocab
    try:
        model.vocab = {**original, "new": 177}
        with pytest.raises(ValueError, match="replaced"):
            model.inference_payload()
    finally:
        model.vocab = original


def test_bad_optimizer_state_rejected_before_model_mutation(model, tmp_path):
    model.configure_trainable(("speaker",))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    path = save_checkpoint(model, tmp_path / "good.pt", step=0, optimizer=optimizer)
    payload = torch.load(path, weights_only=True)
    # A different valid model tensor must NOT be copied before bad optimizer rejection.
    payload["inference_payload"]["text_state"]["speaker_embedding.weight"].zero_()
    parameter_id = payload["optimizer"]["param_groups"][0]["params"][0]
    payload["optimizer"]["state"][parameter_id] = {
        "step": torch.tensor(1.), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1),
    }
    corrupt = tmp_path / "corrupt.pt"
    torch.save(payload, corrupt)
    before = model.text.speaker_embedding.weight.detach().clone()
    with pytest.raises(ValueError, match="Optimizer moment"):
        load_checkpoint(model, corrupt, optimizer=optimizer)
    assert torch.equal(before, model.text.speaker_embedding.weight)


def test_unsafe_inventory_rejected(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "weights.pt").touch()
    manifest = {"schema": "inflect-native-mini-phonemes-v1", "files": {"weights.pt": "", "../escape.py": ""}}
    # Put the unsafe member first, avoiding an earlier unrelated hash failure.
    manifest["files"] = {"../escape.py": "", "weights.pt": ""}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe inventory"):
        validate_inventory(tmp_path)


def test_imports_restore_existing_modules_and_path(model):
    old = sys.modules.get("student")
    sentinel = ModuleType("student")
    sys.modules["student"] = sentinel
    old_path = sys.path[:]
    try:
        native = load_public_runtime(MODEL)
        assert sys.modules["student"] is sentinel
        assert sys.path == old_path
        assert sum(p.numel() for p in native.text.parameters()) == COUNTS["text"]
    finally:
        if old is None:
            sys.modules.pop("student", None)
        else:
            sys.modules["student"] = old


def test_exports_do_not_overwrite_or_touch_source(model, tmp_path):
    with pytest.raises(ValueError, match="overwrite"):
        export_model(model, MODEL)
    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"keep")
    with pytest.raises(ValueError, match="overwrite"):
        save_checkpoint(model, existing, step=0)
    assert existing.read_bytes() == b"keep"
