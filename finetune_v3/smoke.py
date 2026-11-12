"""Self-generated mechanical fixture; never a foreign-language quality test."""
import json
from pathlib import Path

import soundfile as sf
import torch

from .model import load_mini
from .data import prepare

EXAMPLES = (
    ('Hello.', 'h\u0259l\u02c8O.'),
    ('Good morning.', '\u0261\u028ad m\u02c8\u0254\u0279n\u026a\u014b.'),
    ('A quiet day.', '\u0250 kw\u02c8a\u026a\u0259t d\u02c8A.'),
    ('Please wait.', 'pl\u02c8iz w\u02c8At.'),
    ('We are here.', 'wi \u0251\u0279 h\u02c8\u026a\u0279.'),
    ('See you soon.', 's\u02c8i ju s\u02c8un.'),
)


def make_smoke_dataset(model_path, output, *, frontend='prephonemized'):
    if frontend not in ('prephonemized', 'espeak'):
        raise ValueError('Unknown mechanical-fixture frontend')
    model = load_mini(model_path)
    model.eval()
    root = Path(output).absolute()
    if root.resolve().is_relative_to(model.model_dir) or root.exists() or not root.parent.is_dir():
        raise ValueError('Choose a new fixture directory outside the source model with existing parent')
    root.mkdir()
    raw = root/'raw'
    raw.mkdir()
    from .frontend import Frontend
    front = Frontend('en-us', frontend, model.vocab)
    rows=[]
    for index,(text,phones) in enumerate(EXAMPLES):
        if frontend == 'espeak':
            phones = front.encode(text)['phones']
        ids = model.phoneme_ids(phones)
        with torch.no_grad():
            features=model.text.infer(torch.tensor([ids]),torch.tensor([len(ids)]),torch.tensor([2]))
            durations=features['aln'][0].sum(1).long().tolist()
            audio=model.synthesize_ids(ids,'cassidy',seed=42)['audio']
        name=f'fixture-{index:02d}'
        with (raw/(name+'.wav')).open('xb') as stream:
            sf.write(stream,audio.numpy(),24000,format='WAV',subtype='FLOAT')
        row = dict(id=name,audio=name+'.wav',text=text,
            alignment_phones=phones,phone_durations=durations,group=name,language='en-us',
            source={'kind':'self-generated mechanical fixture','language_quality_evidence':False},
            consent={'assertion':'Generated locally from the supplied model; no reference recording used.'},
            alignment_source={'kind':'dataset_alignment','language':'en-us','base_predicted':False,
                'source':'Exact duration clock used to GENERATE this fixture. Not an estimated alignment for a separate recording.'})
        if frontend == 'prephonemized':
            row['phones'] = phones
        rows.append(row)
    manifest=raw/'metadata.jsonl'
    manifest.write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in rows),encoding='utf-8')
    vocabulary=root/'vocab.json'
    vocabulary.write_text(json.dumps(model.vocab,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    index=prepare(manifest,raw,root/'prepared','en-us',frontend,vocabulary,2026,validation_fraction=.34)
    scope=dict(schema='inflect-v3-mechanical-smoke-fixture-v1',self_generated=True,clips=6,
        language_quality_evidence=False,foreign_language_alignment_evidence=False,
        purpose='Check preparation/training/resume/export/inference plumbing only',
        do_not_use='Never substitute these English predicted clocks for alignment labels on a separate recording or new language.',
        source_weights_sha256=model.source_identity['weights_sha256'],frontend=frontend,prepared=str(index))
    (root/'SMOKE-SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n',encoding='utf-8')
    return scope
