"""Public command boundary for local Mini v3 adaptation."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(description='Mini v3 aligned language/fixed-voice adaptation. Not the v2 trainer.')
    sub = root.add_subparsers(dest='command', required=True)
    p = sub.add_parser('smoke-dataset', help='Generate a mechanical fixture, not language-quality evidence')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--acknowledge-mechanical-only', action='store_true', required=True)
    p.add_argument('--frontend', choices=['prephonemized','espeak'], default='prephonemized')
    p = sub.add_parser('vocab', help='Extract the verified base vocabulary before preparing data')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('coverage', help='Audit language/frontend phones before collecting or aligning audio')
    p.add_argument('--manifest', required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--language', required=True)
    p.add_argument('--frontend', choices=['prephonemized','espeak'], default='espeak')
    p.add_argument('--phone-mapping')
    p.add_argument('--base-vocab')
    p.add_argument('--output', required=True)
    p = sub.add_parser('prepare', help='Prepare explicitly aligned recordings; never guess durations')
    p.add_argument('--manifest', required=True)
    p.add_argument('--audio-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--language', required=True)
    p.add_argument('--frontend', choices=['prephonemized','espeak'], default='prephonemized')
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--validation-fraction', type=float, default=.1)
    p.add_argument('--phone-mapping')
    p.add_argument('--base-vocab')
    p.add_argument('--allow-resample', action='store_true')
    p.add_argument('--clock-tolerance-samples', type=int, default=0)
    p = sub.add_parser('audit', help='Verify prepared data, coverage, alignment and split integrity')
    p.add_argument('--dataset', required=True)
    p = sub.add_parser('train', help='Train only explicitly selected modules; preserve source weights')
    for name in ['model','dataset','output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--voice', default='cassidy', choices=['cassidy','hope','sophia','james','jasper','mark'])
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--groups', nargs='+', default=['text_encoder','duration','pitch_noise','speaker'])
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--gradient-accumulation', type=int, default=1)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--validation-every', type=int, default=100)
    p.add_argument('--max-validation-clips', type=int, default=4)
    p.add_argument('--max-frames', type=int, default=480)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--duration-weight', type=float, default=1.)
    p.add_argument('--spectral-weight', type=float, default=1.)
    p.add_argument('--resume')
    p.add_argument('--vocab-migration')
    p = sub.add_parser('export', help='Export a separate model core with its explicit frontend contract')
    p.add_argument('--run', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('say', help='Generate with an exported adapted model; no implicit English G2P')
    p.add_argument('--model', required=True)
    p.add_argument('--voice')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--phonemes')
    group.add_argument('--text')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    return root


def export_run(run, checkpoint, output):
    from .model import load_mini, sha256
    from .checkpoint import load_checkpoint, export_model
    from .training import atomic_json, validate_controller_checkpoint
    run = Path(run).resolve(strict=True)
    config = json.loads((run/'run-contract.json').read_text(encoding='utf-8'))
    point = Path(checkpoint).resolve(strict=True)
    if point.parent != run/'checkpoints':
        raise ValueError('Choose a checkpoint belonging to this run')
    controller = validate_controller_checkpoint(run, point, config)
    from .notices import source_notices, retain_notices
    notice_root, notice_files = source_notices(config['model'])
    model = load_mini(config['model'], trainable_groups=config['groups'])
    migration = config.get('vocabulary_migration')
    if migration:
        model.migrate_vocab(migration['phone_to_id'], seed=migration.get('seed',42), copy_from=migration.get('copy_from'))
    restored = load_checkpoint(model, point)
    if restored['step'] != controller['step']:
        raise ValueError('Controller step differs from checkpoint')
    model.eval()
    result = export_model(model, output)
    dest = Path(output)
    frontend = config.get('frontend')
    atomic_json(dest/'frontend.json',dict(frontend=frontend, target_voice=config['voice'], automatic_english_fallback=False,
        dataset_sha256=config['dataset_sha256'], phoneme_input_supported=True,
        vocabulary=model.vocab, requires_matching_training_frontend=True))
    manifest = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    manifest['files'].update(retain_notices(dest, notice_root, notice_files))
    manifest['files']['frontend.json'] = sha256(dest/'frontend.json')
    atomic_json(dest/'manifest.json',manifest)
    result['frontend_sha256'] = sha256(dest/'frontend.json')
    result['manifest_sha256'] = sha256(dest/'manifest.json')
    result['source_notices_retained'] = True
    return result


def say(args):
    import soundfile as sf
    import torch
    from .model import load_public_runtime
    torch.set_num_threads(2)
    root = Path(args.model).resolve(strict=True)
    model = load_public_runtime(root)
    config = json.loads((root/'frontend.json').read_text(encoding='utf-8'))
    target_voice = config.get('target_voice')
    voice = args.voice or target_voice
    if target_voice and voice != target_voice:
        raise ValueError('This adaptation was configured for '+target_voice+'; other preset slots are not qualified')
    phones = args.phonemes
    if args.text:
        frontend = config['frontend']
        if not isinstance(frontend,dict) or frontend.get('mode') != 'espeak':
            raise ValueError('This export requires explicit --phonemes; there is no implicit English frontend')
        from .frontend import Frontend
        front = Frontend(frontend['language'],'espeak',config['vocabulary'],phone_mapping=frontend.get('phone_mapping'))
        phones = front.encode(args.text)['phones']
        version = frontend.get('espeak_version')
        if not version or front.metadata().get('espeak_version') != version:
            raise ValueError('eSpeak engine version differs from training; use the matching engine or explicit --phonemes')
    if voice not in ('hope','sophia','cassidy','james','jasper','mark'):
        raise ValueError('Unknown preset voice slot')
    dest = Path(args.output).absolute()
    if dest.exists() or dest.is_symlink() or not dest.parent.is_dir() or dest.resolve().is_relative_to(root):
        raise ValueError('Choose a new output file outside the exported model, in an existing directory')
    result = model.synthesize_phonemes(phones,voice,seed=args.seed)
    with dest.open('xb') as stream:
        sf.write(stream,result['audio'].numpy(),result['sample_rate'],format='WAV',subtype='PCM_16')
    return dict(output=str(dest),sample_rate=result['sample_rate'],frames=result['frames'])


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'smoke-dataset':
            import torch
            from .smoke import make_smoke_dataset
            torch.set_num_threads(2)
            result = make_smoke_dataset(args.model,args.output,frontend=args.frontend)
        elif args.command == 'vocab':
            import torch
            from .model import validate_inventory
            root,identity = validate_inventory(args.model)
            payload = torch.load(root/'weights.pt',map_location='cpu',weights_only=True)
            output = Path(args.output).absolute()
            if output.exists() or not output.parent.is_dir() or output.resolve().is_relative_to(root):
                raise ValueError('Choose a new vocabulary JSON path outside the model')
            with output.open('x',encoding='utf-8') as stream:
                json.dump(payload['vocab'],stream,ensure_ascii=False,indent=2)
            result = dict(vocab=str(output),mapped_symbols=len(payload['vocab']),embedding_rows=178,
                unused_ids=sorted(set(range(1,178))-set(payload['vocab'].values())),source_identity=identity)
        elif args.command == 'coverage':
            from collections import Counter
            from .data import read_jsonl
            from .frontend import Frontend
            front = Frontend(args.language,args.frontend,args.vocab,
                             phone_mapping=args.phone_mapping,base_vocab=args.base_vocab)
            rows = read_jsonl(Path(args.manifest))
            missing = Counter()
            examples=[]
            for row in rows:
                encoded = front.encode(row['text'],phones=row.get('phones',row.get('phonemes')),audit_only=True)
                missing.update(c for c in encoded['phones'] if c not in front.vocab)
                if len(examples)<12:
                    examples.append(dict(id=row.get('id'),**encoded))
            result = dict(language=args.language,rows=len(rows),frontend=front.metadata(),
                unknown_codepoints=dict(missing),examples=examples,
                status='coverage-passed' if not missing else 'explicit-vocabulary-migration-required',
                language_quality_approved=False,alignment_generated=False)
            output = Path(args.output)
            if output.exists() or not output.parent.is_dir():
                raise ValueError('Choose a new coverage report path with existing parent')
            with output.open('x',encoding='utf-8') as stream:
                json.dump(result,stream,ensure_ascii=False,indent=2)
        elif args.command == 'train':
            from .training import TrainOptions, train
            options = vars(args).copy()
            del options['command']
            options['groups'] = tuple(options['groups'])
            result = dict(run=str(train(TrainOptions(**options))))
        elif args.command == 'export':
            result = export_run(args.run,args.checkpoint,args.output)
        elif args.command == 'say':
            result = say(args)
        elif args.command == 'prepare':
            from .data import prepare
            result = prepare(args.manifest,args.audio_root,args.output,args.language,args.frontend,args.vocab,
                args.split_seed,validation_fraction=args.validation_fraction,
                phone_mapping=args.phone_mapping,base_vocab=args.base_vocab,
                allow_resample=args.allow_resample,clock_tolerance_samples=args.clock_tolerance_samples)
        else:
            from .data import audit
            result = audit(args.dataset)
        print(json.dumps(result,default=str,allow_nan=False))
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        print('Error: '+str(error),file=sys.stderr)
        return 2
    return 0
