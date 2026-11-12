"""Retain the supplied distribution's license and third-party notice texts."""
from pathlib import Path
import shutil

from .model import resolve_model_dir, sha256


def source_notices(model_path):
    """Resolve notices from the full downloaded Mini package, not a guessed grant."""
    model = resolve_model_dir(model_path)
    package = next((p for p in (model/'source-notices', *model.parents)
                    if (p/'LICENSING.md').is_file() and (p/'LICENSE.md').is_file()
                    and (p/'core/THIRD_PARTY_NOTICES.md').is_file()), None)
    if package is None:
        raise ValueError('CLI export requires the full source Mini package with LICENSE.md, '
                         'LICENSING.md and core notices; retain these when downloading')
    names = ['LICENSE.md', 'LICENSING.md', 'THIRD_PARTY_NOTICES.md',
             'core/LICENSE.md', 'core/LICENSE_SCOPE.json', 'core/THIRD_PARTY_NOTICES.md',
             'core/LICENSES/Apache-2.0.txt', 'core/LICENSES/StyleTTS2-MIT.txt']
    names += [p.relative_to(package).as_posix()
              for p in sorted((package/'core/LICENSES').rglob('*'))
              if p.is_file() and p.relative_to(package).as_posix() not in names]
    files = {}
    for name in names:
        path = package/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(package):
            raise ValueError('Required source notice missing or unsafe: '+name)
        files[name] = sha256(path)
    return package, files


def retain_notices(destination, source, files):
    """Preserve original texts and paths; do not relicense third-party material."""
    destination, source = Path(destination), Path(source)
    result = {}
    for name, digest in files.items():
        original = source/name
        if sha256(original) != digest:
            raise ValueError('Source notice changed during export: '+name)
        target = destination/'source-notices'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if sha256(target) != digest:
            raise ValueError('Source notice readback differs: '+name)
        result[target.relative_to(destination).as_posix()] = digest
    for name in ('LICENSE.md', 'LICENSING.md'):
        shutil.copyfile(source/name, destination/name)
        result[name] = sha256(destination/name)
    text = ('# Adapted Model Notices\n\n'
            'Inflect-owned source material retains the grant in [LICENSING.md](LICENSING.md). '
            'Third-party material is not relicensed. Your training data, recordings and '
            'speaker permissions remain your responsibility.\n\n'
            'The original supplied texts are retained verbatim under '
            '[source-notices/](source-notices/), including '
            '[core notices](source-notices/core/THIRD_PARTY_NOTICES.md) and '
            '[license texts](source-notices/core/LICENSES/). '
            'Those texts describe the complete original package, including optional '
            'components not included in this model-core export; historical source paths '
            'and weight hashes are provenance, not hashes for the newly adapted weights.\n\n'
            'This export includes the adapted Mini core and its explicit frontend '
            'configuration, not the original event, enhancer or tuner components.\n')
    (destination/'THIRD_PARTY_NOTICES.md').write_text(text, encoding='utf-8')
    result['THIRD_PARTY_NOTICES.md'] = sha256(destination/'THIRD_PARTY_NOTICES.md')
    return result
