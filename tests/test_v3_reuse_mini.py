import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/v3_reuse_mini.py'
SPEC = importlib.util.spec_from_file_location('mini_reuse_example', SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

FAKE_API = '''import io, types, wave, time
loads = numeric_loads = preparations = active = peak = 0
def load_numeric(root):
    global numeric_loads
    numeric_loads += 1
    return lambda text: text.replace('2', 'two')
def prepare_text(text, normalizer, mode):
    global preparations
    assert mode == 'conservative'
    preparations += 1
    return {'spoken_text': normalizer(text)}
def load_selected(model, device, profile, runtime, root):
    global loads
    assert (model, device, profile, runtime) == ('mini', 'cpu', 'base', 'canonical')
    loads += 1
    return types.SimpleNamespace(load_numeric=load_numeric, prepare_text=prepare_text), object(), None
def render(prepared, voice, runtime, enhancer, seed, subtype):
    global active, peak
    assert enhancer is None and seed == 42 and subtype == 'PCM_16'
    assert 'two' in prepared['spoken_text']
    active += 1
    peak = max(peak, active)
    time.sleep(.002)
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000)
        audio.writeframes(b'\\0\\0' * 12)
    active -= 1
    return stream.getvalue(), {'sample_rate': 24000}
'''


class ExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v3-reuse-mock-')
        self.root = Path(self.temp.name)
        self.package = self.root / 'package'
        self.core = self.package / 'core'
        self.core.mkdir(parents=True)
        (self.package / 'run.py').write_text('# not imported', encoding='utf-8')
        (self.core / 'synthesize.py').write_text(FAKE_API, encoding='utf-8')
        weights = self.core / 'baseline/models/mini/weights.pt'
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b'fake unit-test weight bytes, never deserialized')
        self.files = {'synthesize.py': M.sha(self.core / 'synthesize.py'),
                      'baseline/models/mini/weights.pt': M.sha(weights)}
        self.patchers = [patch.object(M, '_USED', False),
                         patch.object(M, 'WEIGHTS_SHA256', self.files['baseline/models/mini/weights.pt']),
                         patch.object(sys, 'dont_write_bytecode', True), patch.dict(M.os.environ)]
        for p in self.patchers:
            p.start()
        self.manifest_patch = None
        self.repin()

    def repin(self):
        path = self.core / 'assembly-manifest.json'
        path.write_text(json.dumps({'files': self.files}), encoding='utf-8')
        if self.manifest_patch:
            self.manifest_patch.stop()
        self.manifest_patch = patch.object(M, 'CORE_SHA256', M.sha(path))
        self.manifest_patch.start()

    def tearDown(self):
        self.manifest_patch.stop()
        for p in reversed(self.patchers):
            p.stop()
        self.temp.cleanup()

    def test_two_requests_load_model_and_normalizer_once(self):
        session = M.MiniSession(self.package)
        self.assertEqual(session.render('2 apples.'), session.render('2 pears.', 'hope'))
        self.assertEqual((session.api.loads, session.api.numeric_loads, session.api.preparations), (1, 1, 2))

    def test_render_lock_serializes_requests(self):
        session = M.MiniSession(self.package)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(len(list(pool.map(session.render, ['2 words.'] * 8))), 8)
        self.assertEqual(session.api.peak, 1)

    def test_second_session_refused(self):
        M.MiniSession(self.package)
        with self.assertRaisesRegex(RuntimeError, 'One Mini session'):
            M.MiniSession(self.package)

    def test_manifest_pin_checked_before_code_execution(self):
        (self.core / 'assembly-manifest.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'pinned core'):
            M.MiniSession(self.package)
        self.assertFalse(M._USED)

    def test_source_tamper_rejected_before_import(self):
        (self.core / 'synthesize.py').write_text('raise AssertionError("executed")', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Core integrity failed'):
            M.MiniSession(self.package)

    def test_inventory_traversal_rejected(self):
        self.files['../run.py'] = M.sha(self.package / 'run.py')
        self.repin()
        with self.assertRaisesRegex(ValueError, 'Unsafe core inventory'):
            M.MiniSession(self.package)

    def test_bytecode_cache_rejected(self):
        cache = self.core / '__pycache__'
        cache.mkdir()
        (cache / 'synthesize.cpython-312.pyc').write_bytes(b'not executed')
        with self.assertRaisesRegex(ValueError, 'cache/extension refused'):
            M.MiniSession(self.package)

    def test_unlisted_python_rejected(self):
        (self.core / 'shadow.py').write_text('raise AssertionError', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unlisted package source'):
            M.MiniSession(self.package)

    def test_preloaded_module_collision_rejected(self):
        with patch.dict(sys.modules, {'kokoro': types.ModuleType('kokoro')}):
            with self.assertRaisesRegex(RuntimeError, 'collision'):
                M.MiniSession(self.package)

    def test_markup_rejected_and_official_route_linked(self):
        for text in ('2 words [laugh]', '<speak>2 words</speak>', '2 words]'):
            with self.assertRaisesRegex(ValueError, 'official events CLI'):
                M.speech_text(text)

    def test_voice_and_text_validation(self):
        session = M.MiniSession(self.package)
        with self.assertRaisesRegex(ValueError, 'Unknown preset'):
            session.render('2 words', 'unknown')
        with self.assertRaises(ValueError):
            session.render('')
        self.assertEqual(session.api.preparations, 0)

    def test_cli_new_wavs_and_no_overwrite(self):
        output = self.root / 'out'
        output.mkdir()
        args = ['--package', str(self.package), '--text', '2 words.', '--text', '2 more.', '--output-dir', str(output)]
        with patch('builtins.print'):
            M.main(args)
        self.assertEqual(len(list(output.glob('*.wav'))), 2)
        old = (output / 'speech-001.wav').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            M.main(args)
        self.assertEqual((output / 'speech-001.wav').read_bytes(), old)

    def test_output_inside_entire_package_rejected(self):
        output = self.package / 'not-core'
        output.mkdir()
        with self.assertRaisesRegex(ValueError, 'entire package'):
            M.main(['--package', str(self.package), '--text', '2 words', '--output-dir', str(output)])
        self.assertFalse(M._USED)

    def test_mutable_root_readme_not_pinned_or_executed(self):
        (self.package / 'README.md').write_text('New public card', encoding='utf-8')
        (self.package / 'release-manifest.json').write_text('{}', encoding='utf-8')
        self.assertIsInstance(M.MiniSession(self.package), M.MiniSession)

    def test_linked_package_rejected_without_following_it(self):
        original = Path.is_symlink

        def linked(path):
            return path == self.package or original(path)

        with patch.object(Path, 'is_symlink', linked):
            with self.assertRaisesRegex(ValueError, 'Linked paths'):
                M.MiniSession(self.package)
        self.assertFalse(M._USED)

    def test_cpu_only_cli_rejects_cuda_option_before_loading(self):
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            M.main(['--package', str(self.package), '--text', '2 words',
                    '--output-dir', str(self.root), '--device', 'cuda'])
        self.assertFalse(M._USED)


if __name__ == '__main__':
    unittest.main()
