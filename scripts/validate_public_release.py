"""Check the current v3 front door and retained v2 release documentation."""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    'README.md', 'requirements.txt',
    'assets/inflect-micro-v2-cover.png', 'assets/inflect-nano-v2-cover.png',
    'assets/evidence/cpu-throughput.svg', 'docs/ARCHITECTURE.md',
    'docs/DEPLOYMENT.md', 'docs/EVALUATION.md', 'docs/README.md',
    'docs/INFLECT_V2_TECHNICAL_REPORT.md', 'examples/download_and_speak.py',
    'examples/compare_models.py', 'finetune_v3/README.md',
    'finetune_v3/pyproject.toml', 'docs/v3/language-adaptation/index.md',
    'docs/v3/getting-started.md', 'docs/v3/features.md',
)
STALE_PUBLIC_TEXT = ('1.58x real time', '1.58\u00d7 real time',
                     '1.59x real time', '1.59\u00d7 real time')
MARKDOWN_LINK = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
HTML_LINK = re.compile(r'(?:href|src)="([^"]+)"')


def main():
    missing = [name for name in REQUIRED_FILES if not (ROOT/name).is_file()]
    if missing:
        raise SystemExit('Missing public release files:\n- '+'\n- '.join(missing))
    legacy = [ROOT/'docs'/name for name in (
        'DEPLOYMENT.md','EVALUATION.md','README.md','INFLECT_V2_TECHNICAL_REPORT.md',
        'INFLECT_V2_RELEASE_NOTES_20260721.md')]
    public_docs = [ROOT/'README.md', *legacy, *sorted((ROOT/'docs/v3').rglob('*.md')),
                   ROOT/'finetune_v3/README.md',ROOT/'finetune_v3/API_HANDOFF.md']
    violations = []
    for path in public_docs:
        text = path.read_text(encoding='utf-8')
        for stale in STALE_PUBLIC_TEXT:
            if stale in text:
                violations.append(f'{path.relative_to(ROOT)} contains {stale!r}')
        # This old hardware attribution was incorrect for the managed v2 probe.
        # It is valid in the separately measured v3 CPU protocols.
        if path in legacy and 'AMD Ryzen 9 3900X' in text:
            violations.append(f'{path.relative_to(ROOT)} contains stale v2 CPU attribution')
        if re.search(r'\b[A-Za-z]:\\',text):
            violations.append(f'{path.relative_to(ROOT)} contains an absolute Windows path')
        for target in MARKDOWN_LINK.findall(text)+HTML_LINK.findall(text):
            target = target.strip().split('#',1)[0]
            if not target or '://' in target or target.startswith(('mailto:','#')):
                continue
            if not (path.parent/target).resolve().exists():
                violations.append(f'{path.relative_to(ROOT)} links to missing local target {target!r}')
    if violations:
        raise SystemExit('Public release inconsistencies:\n- '+'\n- '.join(violations))
    print(f'Validated {len(REQUIRED_FILES)} release files and {len(public_docs)} docs.')


if __name__ == '__main__':
    main()
