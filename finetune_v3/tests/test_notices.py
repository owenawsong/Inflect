"""Notice retention is explicit, complete and content-bound."""
from pathlib import Path

import pytest

from finetune_v3.notices import retain_notices, source_notices
from finetune_v3.model import sha256


def fixture_package(tmp_path):
    package = tmp_path/'package'
    core = package/'core/baseline/models/mini'
    (core/'code').mkdir(parents=True)
    (core/'weights.pt').write_bytes(b'fixture-not-weights')
    for name in ('LICENSE.md', 'LICENSING.md', 'THIRD_PARTY_NOTICES.md',
                 'core/LICENSE.md', 'core/LICENSE_SCOPE.json', 'core/THIRD_PARTY_NOTICES.md',
                 'core/LICENSES/Apache-2.0.txt', 'core/LICENSES/StyleTTS2-MIT.txt'):
        p = package/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('Original notice '+name, encoding='utf-8')
    return package, core


def test_retains_original_texts_and_inventory(tmp_path):
    package, core = fixture_package(tmp_path)
    source, files = source_notices(core)
    assert source == package
    dest = tmp_path/'export'
    dest.mkdir()
    inventory = retain_notices(dest, source, files)
    assert len(files) == 8
    for name, digest in files.items():
        assert (dest/'source-notices'/name).read_bytes() == (package/name).read_bytes()
        assert inventory['source-notices/'+name] == digest
    assert all(sha256(dest/name) == digest for name, digest in inventory.items())
    assert 'not included' in (dest/'THIRD_PARTY_NOTICES.md').read_text()


def test_rejects_changed_or_incomplete_source(tmp_path):
    package, core = fixture_package(tmp_path)
    source, files = source_notices(core)
    (package/'LICENSE.md').write_text('changed', encoding='utf-8')
    dest = tmp_path/'export'
    dest.mkdir()
    with pytest.raises(ValueError, match='changed during export'):
        retain_notices(dest, source, files)
    (package/'core/THIRD_PARTY_NOTICES.md').unlink()
    with pytest.raises(ValueError, match='full source Mini package'):
        source_notices(core)


def test_rejects_missing_runtime_license_text(tmp_path):
    package, core = fixture_package(tmp_path)
    (package/'core/LICENSES/StyleTTS2-MIT.txt').unlink()
    with pytest.raises(ValueError, match='Required source notice missing'):
        source_notices(core)


def test_an_export_can_be_the_next_adaptation_source(tmp_path):
    package, core = fixture_package(tmp_path)
    source, files = source_notices(core)
    first = tmp_path/'first-export'
    (first/'code').mkdir(parents=True)
    (first/'weights.pt').write_bytes(b'fixture-adapted-weight-identity')
    retain_notices(first,source,files)
    retained, second_files = source_notices(first)
    assert retained == first/'source-notices'
    assert second_files == files
