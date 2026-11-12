"""Migration, native duration-rounding and resume/export interface regressions."""

from copy import deepcopy
import inspect
import os
from pathlib import Path

import pytest
import torch

from finetune_v3 import export_model, load_checkpoint, load_mini, load_public_runtime, save_checkpoint
from finetune_v3.model import PUBLIC_WEIGHTS_SHA256, sha256


ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("INFLECT_V3_TEST_MODEL", str(
    ROOT / "outputs/events-public-sync-20261111-r001/package/verified-download/core/baseline/models/mini")))
pytestmark = pytest.mark.skipif(not MODEL.is_dir() and 'INFLECT_V3_TEST_MODEL' not in os.environ,
                                reason='Optional model integration: set INFLECT_V3_TEST_MODEL; no download fallback')
EMBEDDINGS = ("bert.embeddings.word_embeddings.weight", "text_encoder.embedding.weight")
PHONE_A, PHONE_B = "\u2603", "\u2602"  # Synthetic test labels, not a language-quality claim.


@pytest.fixture(scope="module")
def reference():
    torch.set_num_threads(2)
    return load_mini(MODEL, expected_weights_sha256=PUBLIC_WEIGHTS_SHA256).eval()


def test_public_native_duration_rounding_is_not_label_preparation(reference):
    alignment = inspect.unwrap(reference.text.infer).__globals__["duration_alignment"]
    # Torch native round uses ties-to-even, then the release clamps to >=1.
    predicted = torch.tensor([[0.5, 1.5, 2.5, 3.5]])
    lengths = torch.tensor([4])
    aln, frames = alignment(predicted, lengths)
    assert torch.equal(aln.sum(2), torch.tensor([[1., 2., 2., 4.]]))
    assert frames.tolist() == [9]
    faster, faster_frames = alignment(predicted, lengths, speed=2.)
    assert torch.equal(faster.sum(2), torch.tensor([[1., 1., 1., 2.]]))
    assert faster_frames.tolist() == [5]
    assert reference.text.get_config()["n_token"] == 178


def test_forward_preserves_explicit_integer_duration_labels(reference):
    ids = torch.tensor([reference.phoneme_ids("ha")])
    lengths = torch.tensor([4])
    supplied = torch.tensor([1, 3, 2, 4])
    path = torch.repeat_interleave(torch.arange(4), supplied)
    aln = torch.nn.functional.one_hot(path, num_classes=4).T[None].float()
    before = aln.clone()
    with torch.no_grad():
        out = reference.forward_grad(ids, lengths, aln, torch.tensor([2]), decode=False)
    assert torch.equal(aln, before)
    assert out["aln"] is aln
    assert torch.equal(out["aln"].sum(2), supplied[None].float())
    assert out["f0"].shape == (1, 2 * int(supplied.sum()))
    assert out["asr"].shape == (1, 512, int(supplied.sum()))


def test_public_torch_stft_window_remains_cpu_and_follows_input(reference):
    # The qualified generator uses TorchSTFT, not the alternate CustomSTFT.
    stft = deepcopy(reference.generator.stft)
    assert type(stft).__name__ == "TorchSTFT"
    assert "window" not in dict(stft.named_buffers())
    waveform = torch.linspace(-0.1, 0.1, 2400)[None]
    expected = type(stft).transform(stft, waveform)
    stft.to("meta")
    assert stft.window.device.type == "cpu"
    actual = type(stft).transform(stft, waveform)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("mapping,copy_from", (
    ({"h": 7}, None),
    ({PHONE_A: 50}, None),
    ({PHONE_A: 178}, None),
    ({PHONE_A: 0}, None),
    ({PHONE_A: True}, None),
    ({PHONE_A: 7, PHONE_B: 7}, None),
    ({"two": 7}, None),
    ({PHONE_A: 7}, {PHONE_A: "missing"}),
))
def test_invalid_migration_is_rejected_before_mutation(mapping, copy_from):
    model = load_mini(MODEL).eval()
    before = model.inference_payload()
    with pytest.raises(ValueError):
        model.migrate_vocab(mapping, copy_from=copy_from)
    assert model.vocab == before["vocab"]
    for key in EMBEDDINGS:
        assert torch.equal(dict(model.text.named_parameters())[key], before["text_state"][key])
    assert not model.vocabulary_migrations


def test_migration_determinism_base_rows_and_export_resume(tmp_path):
    model = load_mini(MODEL, trainable_groups=("speaker",)).eval()
    old_vocab = dict(model.vocab)
    unused = sorted(set(range(1, 178)) - set(old_vocab.values()))
    assert len(old_vocab) == 114 and len(unused) == 63
    original = {name: parameter.detach().clone() for name, parameter in model.text.named_parameters()
                if name in EMBEDDINGS}
    mapping = {PHONE_A: unused[0], PHONE_B: unused[1]}
    keep = [index for index in range(178) if index not in mapping.values()]
    rng_before = torch.random.get_rng_state().clone()
    receipt = model.migrate_vocab(mapping, seed=123, copy_from={PHONE_B: "h"})
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(model.vocab[phone] == index for phone, index in old_vocab.items())
    assert receipt["base_rows_bit_identical"] is True
    assert receipt["factory_rows"] == 178
    assert receipt["source_weights_sha256"] == PUBLIC_WEIGHTS_SHA256
    assert receipt["base_contract_sha256"] != receipt["new_contract_sha256"]
    for key in EMBEDDINGS:
        current = dict(model.text.named_parameters())[key].detach()
        assert torch.equal(original[key][keep], current[keep])
        assert torch.equal(original[key][old_vocab["h"]], current[unused[1]])
        assert not torch.equal(original[key][unused[0]], current[unused[0]])

    restored = load_mini(MODEL, trainable_groups=("speaker",)).eval()
    reordered = {PHONE_B: unused[1], PHONE_A: unused[0]}
    other_receipt = restored.migrate_vocab(reordered, seed=123, copy_from={PHONE_B: "h"})
    assert receipt == other_receipt
    for key in EMBEDDINGS:
        assert torch.equal(dict(model.text.named_parameters())[key], dict(restored.text.named_parameters())[key])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    model.text.speaker_embedding.weight.square().mean().backward()
    optimizer.step()
    checkpoint = save_checkpoint(model, tmp_path / "migrated.pt", step=1, optimizer=optimizer)
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-5)
    assert load_checkpoint(restored, checkpoint, optimizer=restored_optimizer)["step"] == 1
    assert restored_optimizer.state
    assert restored.vocabulary_migrations == model.vocabulary_migrations
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    with pytest.raises(ValueError, match="pre-training"):
        restored.migrate_vocab({"\u2604": unused[2]})

    ids = model.phoneme_ids("h" + PHONE_A + "a")
    before_audio = model.synthesize_ids(ids, voice="cassidy", seed=42)["audio"]
    result = export_model(model, tmp_path / "migrated-export")
    assert result["public_loader_strict_readback"]
    deployed = load_public_runtime(tmp_path / "migrated-export")
    imported = load_mini(tmp_path / "migrated-export").eval()
    assert deployed.vocab == model.vocab == imported.vocab
    assert torch.equal(before_audio, deployed.synthesize_ids(ids, voice="cassidy", seed=42)["audio"])
    assert torch.equal(before_audio, imported.synthesize_ids(ids, voice="cassidy", seed=42)["audio"])
    assert sha256(MODEL / "weights.pt") == PUBLIC_WEIGHTS_SHA256


def test_migration_contract_must_be_replayed_for_resume(tmp_path):
    model = load_mini(MODEL).eval()
    unused = sorted(set(range(1, 178)) - set(model.vocab.values()))
    model.migrate_vocab({PHONE_A: unused[0]}, seed=42)
    checkpoint = save_checkpoint(model, tmp_path / "migrated.pt", step=0)
    base = load_mini(MODEL).eval()
    with pytest.raises(ValueError, match="contract differs"):
        load_checkpoint(base, checkpoint)
    base.migrate_vocab({PHONE_A: unused[0]}, seed=43)
    with pytest.raises(ValueError, match="same explicit vocabulary migration"):
        load_checkpoint(base, checkpoint)


def test_combining_map_native_migration_checkpoint_and_export_keep_exact_ids(tmp_path):
    # Real-base integration: run only after parent admits model/export regression.
    from finetune_v3.frontend import Frontend
    model = load_mini(MODEL).eval()
    unused = min(set(range(1, 178)) - set(model.vocab.values()))
    model.migrate_vocab({'\u0301': unused}, seed=42)
    rules = dict(language='fr-fr', name='explicit-tone-codepoint', replacements={'X': '\u0301'})
    encoded = Frontend('fr-fr', 'prephonemized', model.vocab, phone_mapping=rules).encode('Fixture.', phones='aX')
    expected = [0, model.vocab['a'], unused, 0]
    assert encoded['phones'] == 'a\u0301' and encoded['phone_ids'] == expected
    point = save_checkpoint(model, tmp_path / 'combining.pt', step=0)
    restored = load_mini(MODEL).eval()
    restored.migrate_vocab({'\u0301': unused}, seed=42)
    load_checkpoint(restored, point)
    assert restored.phoneme_ids(encoded['phones']) == expected
    export_model(restored, tmp_path / 'combining-export')
    deployed = load_public_runtime(tmp_path / 'combining-export')
    assert deployed.phoneme_ids(encoded['phones']) == expected
