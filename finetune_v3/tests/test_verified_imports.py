"""Tiny source-only import fixtures; no real model or cache is modified."""
import hashlib
import importlib._bootstrap_external as bootstrap
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from finetune_v3.model import SCHEMA, runtime_imports, validate_inventory


def package(tmp_path, sources):
    root = tmp_path / 'model'
    root.mkdir()
    (root / 'weights.pt').write_bytes(b'fixture only; never deserialized')
    files = {'weights.pt': hashlib.sha256((root / 'weights.pt').read_bytes()).hexdigest()}
    for relative, text in sources.items():
        path = root / 'code' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        files['code/' + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps({'schema': SCHEMA, 'files': files}), encoding='utf-8')
    return root


def cached(source, text, kind='timestamp'):
    code = compile(text, str(source), 'exec')
    if kind == 'timestamp':
        data = bootstrap._code_to_timestamp_pyc(code, int(source.stat().st_mtime), source.stat().st_size)
    else:
        data = bootstrap._code_to_hash_pyc(code, importlib.util.source_hash(source.read_bytes()), checked=False)
    path = Path(importlib.util.cache_from_source(str(source)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize('kind', ['timestamp', 'unchecked-hash'])
def test_ignores_valid_poisoned_cache_and_leaves_it_unchanged(tmp_path, kind):
    root = package(tmp_path, {'native_mini_inference.py': "ORIGIN = 'verified-source'\n"})
    source = root / 'code/native_mini_inference.py'
    cache = cached(source, "ORIGIN = 'unlisted-cache'\n", kind)
    previous_bytes = cache.read_bytes()
    # Control: the ordinary importer really would accept this poisoned fixture.
    namespace = {}
    exec(importlib.machinery.SourceFileLoader('cache_control', str(source)).get_code('cache_control'), namespace)
    assert namespace['ORIGIN'] == 'unlisted-cache'
    root, identity = validate_inventory(root)
    with runtime_imports(root, identity['code_sha256']) as runtime:
        assert runtime.ORIGIN == 'verified-source'
        assert Path(runtime.__file__) == source
    assert cache.read_bytes() == previous_bytes


def test_legitimate_existing_cache_is_not_rejected_or_removed(tmp_path):
    text = "ORIGIN = 'verified-source'\n"
    root = package(tmp_path, {'native_mini_inference.py': text})
    cache = cached(root / 'code/native_mini_inference.py', text)
    previous_bytes = cache.read_bytes()
    root, identity = validate_inventory(root)
    with runtime_imports(root, identity['code_sha256']) as runtime:
        assert runtime.ORIGIN == 'verified-source'
    assert cache.read_bytes() == previous_bytes


def test_no_cache_is_created_by_verified_loader(tmp_path):
    root = package(tmp_path, {'native_mini_inference.py': 'VALUE = 7\n'})
    root, identity = validate_inventory(root)
    with runtime_imports(root, identity['code_sha256']) as runtime:
        assert runtime.VALUE == 7
    assert not list((root / 'code').rglob('*.pyc'))


def test_owned_package_submodule_uses_verified_source_not_cache(tmp_path):
    root = package(tmp_path, {'native_mini_inference.py': 'from kokoro import VALUE\n',
                             'kokoro/__init__.py': 'from .modules import VALUE\n',
                             'kokoro/modules.py': "VALUE = 'verified-child'\n"})
    cache = cached(root / 'code/kokoro/modules.py', "VALUE = 'cached-child'\n")
    previous = cache.read_bytes()
    root, identity = validate_inventory(root)
    with runtime_imports(root, identity['code_sha256']) as runtime:
        assert runtime.VALUE == 'verified-child'
    assert cache.read_bytes() == previous


def test_additional_inventoried_helper_is_source_only_and_isolated(tmp_path, monkeypatch):
    root = package(tmp_path, {'native_mini_inference.py': 'from model_helper import VALUE\n',
                             'model_helper.py': "VALUE = 'verified-helper'\n"})
    cache = cached(root / 'code/model_helper.py', "VALUE = 'cached-helper'\n")
    sentinel = ModuleType('model_helper')
    monkeypatch.setitem(sys.modules, 'model_helper', sentinel)
    root, identity = validate_inventory(root)
    with runtime_imports(root, identity['code_sha256']) as runtime:
        assert runtime.VALUE == 'verified-helper'
    assert sys.modules['model_helper'] is sentinel and cache.is_file()


def test_unlisted_code_cache_cannot_shadow_external_dependency(tmp_path, monkeypatch):
    root = package(tmp_path, {'native_mini_inference.py': 'from fixture_dependency import VALUE\n'})
    trusted = tmp_path / 'trusted-installed-fixture'
    trusted.mkdir()
    (trusted / 'fixture_dependency.py').write_text("VALUE = 'external-trusted'\n", encoding='utf-8')
    code = compile("VALUE = 'unlisted-shadow'", '<unlisted-fixture>', 'exec')
    extra = root / 'code/fixture_dependency.pyc'
    extra.write_bytes(bootstrap._code_to_timestamp_pyc(code))
    monkeypatch.syspath_prepend(str(trusted))
    # An existing caller-provided code path must be excluded for the scope too.
    monkeypatch.syspath_prepend(str(root / 'code'))
    monkeypatch.delitem(sys.modules, 'fixture_dependency', raising=False)
    root, identity = validate_inventory(root)
    try:
        with runtime_imports(root, identity['code_sha256']) as runtime:
            assert runtime.VALUE == 'external-trusted'
        assert extra.is_file()
    finally:
        sys.modules.pop('fixture_dependency', None)


def test_unlisted_native_extension_does_not_override_inventoried_source(tmp_path):
    root = package(tmp_path, {'native_mini_inference.py': "VALUE = 'source'\n"})
    extra = root / 'code' / ('native_mini_inference' + importlib.machinery.EXTENSION_SUFFIXES[0])
    extra.write_bytes(b'not a native extension; must never be loaded')
    root, identity = validate_inventory(root)
    with runtime_imports(root, identity['code_sha256']) as runtime:
        assert runtime.VALUE == 'source'
    assert extra.read_bytes().startswith(b'not a native extension')


def test_owned_name_cannot_fall_back_to_unlisted_sourceless_module(tmp_path):
    root = package(tmp_path, {'native_mini_inference.py': 'import native33_text_fast\n'})
    code = compile("ORIGIN = 'unlisted'", '<fixture>', 'exec')
    extra = root / 'code/native33_text_fast.pyc'
    extra.write_bytes(bootstrap._code_to_timestamp_pyc(code))
    root, identity = validate_inventory(root)
    with pytest.raises(ImportError, match='No inventoried runtime source'):
        with runtime_imports(root, identity['code_sha256']):
            pytest.fail('Unlisted bytecode was admitted')
    assert extra.is_file()


@pytest.mark.parametrize('mutate', [False, True])
def test_context_restores_modules_path_and_finders_even_on_reverification_failure(tmp_path, monkeypatch, mutate):
    root = package(tmp_path, {'native_mini_inference.py': 'VALUE = 1\n'})
    root, identity = validate_inventory(root)
    sentinel = ModuleType('native_mini_inference')
    monkeypatch.setitem(sys.modules, 'native_mini_inference', sentinel)
    old_path, old_meta = sys.path[:], sys.meta_path[:]
    if mutate:
        (root / 'code/native_mini_inference.py').write_text('VALUE = 2\n', encoding='utf-8')
        with pytest.raises(ValueError, match='changed after inventory'):
            with runtime_imports(root, identity['code_sha256']):
                pytest.fail('Changed source was executed')
    else:
        with runtime_imports(root, identity['code_sha256']) as runtime:
            assert runtime is not sentinel and runtime.VALUE == 1
    assert sys.modules['native_mini_inference'] is sentinel
    assert sys.path == old_path and sys.meta_path == old_meta
