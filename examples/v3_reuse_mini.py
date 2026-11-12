"""Load one verified bundled Mini runtime per fresh process; CPU/base only.

Not a stable library API, server or streaming interface. Keep package files
immutable throughout the session. Install the package's CPU requirements first.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import types
import wave

CORE_SHA256 = '3df1dd36fe67562448c2e9560ae8d8610cd5a6369fd21f966bc002fab1308f95'
WEIGHTS_SHA256 = '8460e7d54a2fbf7fb54b3c7be7569c493c852de9a2aa1a0fd9fd9a099de581ed'
VOICES = ('hope', 'sophia', 'cassidy', 'james', 'jasper', 'mark')
EVENT_GUIDE = 'https://github.com/owenawsong/Inflect/blob/main/docs/v3/features.md#vocal-event-tags'
_PROCESS_LOCK = threading.Lock()
_USED = False


def plain_path(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('Linked paths are unsupported; use hf download --local-dir')
    return path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def speech_text(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 20000:
        raise ValueError('Supply nonempty speech text, at most 20000 characters')
    if any(char in text for char in '[]<>'):
        raise ValueError('Speech only: markup/event tags are rejected. Use the official events CLI: ' + EVENT_GUIDE)


def verify_core(package):
    package = plain_path(package).resolve(strict=True)
    core = plain_path(package / 'core')
    manifest = plain_path(core / 'assembly-manifest.json')
    if not (package / 'run.py').is_file() or sha(manifest) != CORE_SHA256:
        raise ValueError('Require a complete Mini package with the pinned core assembly')
    files = json.loads(manifest.read_text(encoding='utf-8'))['files']
    if files.get('baseline/models/mini/weights.pt') != WEIGHTS_SHA256 or 'synthesize.py' not in files:
        raise ValueError('Unexpected core weights or public runtime entrypoint')
    for relative, expected in files.items():
        parts = PurePosixPath(relative).parts
        if not parts or relative != '/'.join(parts) or '\\' in relative or ':' in relative or any(p in ('.', '..', '/') for p in parts):
            raise ValueError('Unsafe core inventory path')
        path = plain_path(core.joinpath(*parts))
        if not path.resolve().is_relative_to(core) or not path.is_file() or sha(path) != expected:
            raise ValueError('Core integrity failed: ' + relative)
    # A fresh source-only package prevents valid-looking stale .pyc execution.
    for path in core.rglob('*'):
        plain_path(path)
        relative = path.relative_to(core).as_posix()
        if path.name == '__pycache__' or path.suffix.lower() in ('.pyc', '.pyo', '.pyd', '.so', '.dll'):
            raise ValueError('Package code cache/extension refused; use a fresh --local-dir download')
        if path.suffix == '.py' and relative not in files:
            raise ValueError('Unlisted package source refused: ' + relative)
    names = {PurePosixPath(p).stem for p in files if p.endswith('.py')}
    names.add('kokoro')
    if any(name.split('.')[0] in names or name.startswith(('_release_', '_selected_', '_inflect_rc_')) for name in sys.modules):
        raise RuntimeError('Bundled-module collision; start a fresh Python process')
    return package, core


class MiniSession:
    def __init__(self, package):
        global _USED
        with _PROCESS_LOCK:
            if _USED:
                raise RuntimeError('One Mini session per process; reuse it or start a fresh process')
            self.package, core = verify_core(package)
            _USED = True  # Failed imports also require a fresh process, not a polluted retry.
            sys.dont_write_bytecode = True
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
            self.api = types.ModuleType('_v3_reuse_public_core')
            source = core / 'synthesize.py'
            self.api.__file__ = str(source)
            exec(compile(source.read_bytes(), str(source), 'exec'), self.api.__dict__)
            self.cli, self.runtime, enhancer = self.api.load_selected('mini', 'cpu', 'base', 'canonical', root=core)
            if enhancer is not None:
                raise RuntimeError('Base-only example must not load an enhancer')
            self.normalizer = self.cli.load_numeric(core / 'baseline')
            self.lock = threading.Lock()

    def render(self, text, voice='cassidy'):
        speech_text(text)
        if voice not in VOICES:
            raise ValueError('Unknown preset voice')
        with self.lock:
            prepared = self.cli.prepare_text(text, self.normalizer, 'conservative')
            payload, _ = self.api.render(prepared, voice, self.runtime, enhancer=None, seed=42, subtype='PCM_16')
            with wave.open(io.BytesIO(payload), 'rb') as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000) or audio.getnframes() == 0:
                    raise ValueError('Expected nonempty native 24 kHz mono PCM16 WAV')
            return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True, help='Complete hf download --local-dir package, not a cache snapshot guess')
    parser.add_argument('--voice', choices=VOICES, default='cassidy')
    parser.add_argument('--text', action='append', required=True, help='Repeat for each complete utterance')
    parser.add_argument('--output-dir', type=Path, required=True, help='Existing output directory outside the whole package')
    args = parser.parse_args(argv)
    package = plain_path(args.package).resolve(strict=True)
    output = plain_path(args.output_dir).resolve(strict=True)
    if not output.is_dir() or output.is_relative_to(package):
        raise ValueError('Output directory must exist outside the entire package')
    targets = [output / f'speech-{index:03d}.wav' for index in range(1, len(args.text) + 1)]
    for text, target in zip(args.text, targets):
        speech_text(text)
        if target.exists() or target.is_symlink():
            raise ValueError('Output already exists; choose an empty output directory')
    session = MiniSession(package)
    for text, target in zip(args.text, targets):
        payload = session.render(text, args.voice)
        with target.open('xb') as stream:
            stream.write(payload)
        print(str(target))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('Mini example failed (no automatic retry): ' + str(error))
