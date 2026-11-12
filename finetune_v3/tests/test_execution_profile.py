"""Lightweight fake-controller tests; no native weights or GPU are loaded."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import ast

import pytest
from finetune_v3 import training, model
from test_training_contract import harness, controller_fixture


def test_new_profile_recorded_and_same_profile_resume_succeeds(harness):
    root = training.train(harness.options)
    contract = json.loads((root/'run-contract.json').read_text(encoding='utf-8'))
    assert contract['execution_profile'] == training.TRAINING_EXECUTION_PROFILE == 'native24-reflect-cat-v1'
    point = root/'checkpoints/step-0000001.pt'
    training.train(replace(harness.options, resume=str(point), steps=2))
    assert harness.restored == [str(point.resolve())]
    assert json.loads((root/'status.json').read_text(encoding='utf-8'))['step'] == 2


@pytest.mark.parametrize('profile', [None, 'native24-reflect-pad-v0', 'native24-reflect-cat-v2'])
def test_missing_or_altered_profile_rejected_before_any_weights_or_updates(harness, monkeypatch, profile):
    root = training.train(harness.options)
    point = root/'checkpoints/step-0000001.pt'
    path = root/'run-contract.json'
    contract = json.loads(path.read_text(encoding='utf-8'))
    if profile is None:
        contract.pop('execution_profile')
    else:
        contract['execution_profile'] = profile
    path.write_text(json.dumps(contract), encoding='utf-8')
    # Preserve a valid sidecar for that older/different contract; the rejection
    # must be about execution compatibility, not stale controller hashes.
    sidecar = point.with_suffix('.controller.json')
    receipt = json.loads(sidecar.read_text(encoding='utf-8'))
    receipt['contract_sha256'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    sidecar.write_text(json.dumps(receipt), encoding='utf-8')
    assert training.validate_controller_checkpoint(root, point, contract)['step'] == 1
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    forwards, restored = list(harness.forwards), list(harness.restored)
    def forbidden(*args, **kwargs):
        pytest.fail('Attempted native/checkpoint weight load, data read, or optimizer creation before profile rejection')
    for name in ('load_mini', 'load_checkpoint', 'load_rows'):
        monkeypatch.setattr(training, name, forbidden)
    monkeypatch.setattr(training.torch.optim, 'AdamW', forbidden)
    with pytest.raises(ValueError, match='execution profile mismatch or missing'):
        training.train(replace(harness.options, resume=str(point), steps=2))
    assert harness.forwards == forwards and harness.restored == restored
    assert before == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_legacy_sidecar_validation_for_export_remains_allowed(controller_fixture):
    run, point, legacy_contract, state = controller_fixture
    assert 'execution_profile' not in legacy_contract
    assert training.validate_controller_checkpoint(run, point, legacy_contract) == state
    # No inference schema or checkpoint model-template field was introduced.
    assert model.SCHEMA == 'inflect-native-mini-phonemes-v1'


def test_legacy_export_validation_function_is_unchanged():
    original = (Path(os.environ.get('INFLECT_V3_REFERENCE_SOURCE_ROOT', str(Path(__file__).resolve().parents[2])))/'finetune_v3/training.py').read_text(encoding='utf-8')
    current = Path(training.__file__).read_text(encoding='utf-8')
    def segment(text):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'validate_controller_checkpoint')
        return ast.get_source_segment(text, node)
    assert segment(original) == segment(current)
