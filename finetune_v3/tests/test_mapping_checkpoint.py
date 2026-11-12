"""Real tiny CPU checkpoints for final-codepoint vocabulary contracts, no base weights."""
from copy import deepcopy

import pytest
import torch

from finetune_v3.checkpoint import load_checkpoint, save_checkpoint
from finetune_v3.frontend import Frontend, phoneme_ids
from finetune_v3.model import contract_digest


class TinyCheckpointModel(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.model_dir = source
        self.source_identity = {'fixture': 'tiny-only-no-language-claim'}
        self.text = torch.nn.Linear(1, 1)
        self.bridge = torch.nn.Linear(1, 1)
        self.generator = torch.nn.Linear(1, 1)
        self.vocab = {'a': 1, '\u0301': 2}
        self._contract_sha256 = contract_digest({'vocab': self.vocab})
        self.trainable_groups = ()
        self.vocabulary_migrations = [{'phone_to_id': {'\u0301': 2}, 'fixture_only': True}]
        self._vocabulary_locked = False

    def inference_payload(self):
        return {'vocab': deepcopy(self.vocab), **{
            name + '_state': {k: v.detach().clone() for k, v in getattr(self, name).state_dict().items()}
            for name in ('text', 'bridge', 'generator')}}


def test_exact_combining_map_vocabulary_survives_real_tiny_checkpoint(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    original = TinyCheckpointModel(source)
    front = Frontend('fr-fr', 'prephonemized', original.vocab, base_vocab={'a': 1},
                     phone_mapping=dict(language='fr-fr', name='explicit-tone', replacements={'X': '\u0301'}))
    encoded = front.encode('Authored fixture.', phones='aX')
    checkpoint = save_checkpoint(original, tmp_path / 'tiny.pt', step=0)
    restored = TinyCheckpointModel(source)
    load_checkpoint(restored, checkpoint)
    assert restored.vocab == original.vocab and '\u00e1' not in restored.vocab
    assert restored.vocabulary_migrations == original.vocabulary_migrations
    assert encoded['phones'] == 'a\u0301'
    assert phoneme_ids(encoded['phones'], restored.vocab) == [0, 1, 2, 0]
    assert checkpoint.stat().st_size < 100_000
    assert restored._vocabulary_locked is True


def test_nfc_changed_vocabulary_cannot_restore_same_checkpoint(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    original = TinyCheckpointModel(source)
    checkpoint = save_checkpoint(original, tmp_path / 'tiny.pt', step=0)
    restored = TinyCheckpointModel(source)
    restored.vocab = {'\u00e1': 2}
    restored._contract_sha256 = contract_digest({'vocab': restored.vocab})
    with pytest.raises(ValueError, match='contract differs'):
        load_checkpoint(restored, checkpoint)
