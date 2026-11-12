"""Explicit-alignment Mini adaptation with bounded, resumable training state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .model import VOICES, load_mini, sha256
from .checkpoint import save_checkpoint, load_checkpoint, export_model


TRAINING_EXECUTION_PROFILE = 'native24-reflect-cat-v1'


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    pending = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex + '.pending')
    owned = False
    try:
        with pending.open('x', encoding='utf-8', newline='\n') as stream:
            owned = True
            stream.write(payload)
        # Windows readers can briefly deny replacement; never relax other errors.
        delays = (.01, .02, .04, .08, .16, .25, .25)
        for attempt in range(8):
            try:
                os.replace(pending, path)
                break
            except OSError as error:
                if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                    raise
                time.sleep(delays[attempt])
    finally:
        if owned:
            pending.unlink(missing_ok=True)


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_controller_checkpoint(run, point, contract):
    """Bind exported/resumed weights to their original data/frontend/voice contract."""
    run, point = Path(run).resolve(strict=True), Path(point).resolve(strict=True)
    if point.parent != (run/'checkpoints').resolve(strict=True):
        raise ValueError('Checkpoint must belong to this run')
    state = json.loads(point.with_suffix('.controller.json').read_text(encoding='utf-8'))
    expected = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (state.get('contract_sha256') != expected or state.get('checkpoint_sha256') != sha256(point)
            or type(state.get('step')) is not int or state['step'] < 0):
        raise ValueError('Controller/checkpoint identity mismatch')
    return state


@dataclass
class TrainOptions:
    model: str
    dataset: str
    output: str
    voice: str = 'cassidy'
    device: str = 'cpu'
    groups: tuple[str, ...] = ('text_encoder', 'duration', 'pitch_noise', 'speaker')
    steps: int = 100
    learning_rate: float = 1e-5
    gradient_accumulation: int = 1
    checkpoint_every: int = 100
    seed: int = 2026
    threads: int = 2
    duration_weight: float = 1.
    spectral_weight: float = 1.
    max_frames: int = 480
    validation_every: int = 100
    max_validation_clips: int = 4
    resume: str | None = None
    vocab_migration: str | None = None

    def validate(self):
        if self.voice not in VOICES or self.device not in ('cpu', 'cuda'):
            raise ValueError('Use a preset voice and cpu/cuda device')
        if not 1 <= self.steps <= 1_000_000 or not 1 <= self.gradient_accumulation <= 128:
            raise ValueError('Steps/accumulation outside bounded range')
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1e-3:
            raise ValueError('Learning rate must be finite and in (0, 1e-3]')
        if not 2 <= self.max_frames <= 1200 or not 1 <= self.threads <= 32:
            raise ValueError('Invalid frame/thread bounds')
        if any(type(x) is not int or x < 1 for x in (self.checkpoint_every, self.validation_every, self.max_validation_clips)):
            raise ValueError('Positive checkpoint/validation limits required')
        if any(not math.isfinite(x) or x < 0 for x in (self.duration_weight, self.spectral_weight)) or self.duration_weight + self.spectral_weight == 0:
            raise ValueError('Finite nonnegative loss weights with nonzero total required')


def load_rows(dataset):
    """Consume an audited prepared dataset, never private corpus layouts."""
    from .data import load_prepared
    audited = load_prepared(dataset)
    root = Path(dataset).resolve(strict=True)
    if root.is_file():
        root = root.parent
    metadata = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    result = {}
    digest = hashlib.sha256((root / 'dataset.json').read_bytes())
    for split in ('train', 'validation'):
        path = root / (split + '.jsonl')
        raw = path.read_bytes()
        digest.update(raw)
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        if not rows:
            raise ValueError('Both train and validation splits must be nonempty')
        for row in rows:
            rel = Path(row['audio'])
            audio = root / rel
            if rel.is_absolute() or '..' in rel.parts or audio.is_symlink() or not audio.resolve().is_relative_to(root):
                raise ValueError('Unsafe prepared audio path')
            expected = row.get('audio_sha256', row.get('sha256'))
            if not expected or sha256(audio) != expected:
                raise ValueError('Prepared audio hash mismatch')
            phones = row.get('phonemes', row.get('phones'))
            durations = row.get('phone_durations', row.get('durations'))
            if not isinstance(phones, str) or not phones or len(phones) > 510:
                raise ValueError('Prepared phonemes required')
            if not isinstance(durations, list) or len(durations) != len(phones) + 2 or any(type(d) is not int or d < 1 for d in durations):
                raise ValueError('Positive integer phone durations including both boundaries required')
            if not 2 <= sum(durations) <= 1200:
                raise ValueError('Invalid aligned utterance length')
            info = sf.info(audio)
            if info.samplerate != 24000 or info.channels != 1 or info.frames != 600 * sum(durations):
                raise ValueError('Prepared audio must match the alignment clock exactly')
            row['_phones'], row['_durations'], row['_audio_path'] = phones, durations, str(audio)
        result[split] = rows
    train_audio = {r.get('audio_sha256', r.get('sha256')) for r in result['train']}
    if train_audio & {r.get('audio_sha256', r.get('sha256')) for r in result['validation']}:
        raise ValueError('Duplicate audio leaked into validation')
    train_text = {' '.join(r['text'].split()).casefold() for r in result['train']}
    if train_text & {' '.join(r['text'].split()).casefold() for r in result['validation']}:
        raise ValueError('Duplicate transcript leaked into validation')
    return metadata, result, digest.hexdigest()


def make_batch(model, row, voice, device, max_frames):
    ids = model.phoneme_ids(row['_phones'])
    counts = row['_durations']
    if sum(counts) > max_frames:
        raise ValueError('Utterance exceeds configured frame cap; use shorter complete clips, never crop silently')
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    sid = torch.tensor([VOICES.index(voice)], dtype=torch.long, device=device)
    counts_t = torch.tensor(counts, dtype=torch.long, device=device)
    path = torch.repeat_interleave(torch.arange(len(ids), device=device), counts_t)
    aln = F.one_hot(path, num_classes=len(ids)).T[None].float()
    audio, rate = sf.read(row['_audio_path'], dtype='float32')
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) >= 1:
        raise ValueError('Nonfinite or clipped training audio')
    target = torch.from_numpy(audio.copy()).to(device)[None]
    return tokens, lengths, aln, sid, counts_t.float()[None], target


def _reflect_center_pad(waveform, padding):
    """STFT's reflect-center values via explicit slice/flip/cat paths."""
    if type(padding) is not int or padding < 0 or waveform.ndim not in (1, 2):
        raise ValueError('Expected 1D/2D waveform and nonnegative integer padding')
    if waveform.shape[-1] <= padding:
        raise ValueError('Reflection padding must be smaller than waveform length')
    if padding == 0:
        return waveform
    return torch.cat((waveform[..., 1:padding+1].flip(-1), waveform,
                      waveform[..., -padding-1:-1].flip(-1)), dim=-1)


def spectral_loss(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError('Waveform and target clocks must match')
    total = predicted.new_zeros(())
    for size in (256, 512, 1024):
        window = torch.hann_window(size, device=predicted.device)
        a = torch.stft(_reflect_center_pad(predicted.float(), size//2), size,
                       hop_length=size//4, window=window, center=False, return_complex=True).abs()
        b = torch.stft(_reflect_center_pad(target.float(), size//2), size,
                       hop_length=size//4, window=window, center=False, return_complex=True).abs()
        total = total + F.l1_loss(torch.log(a.clamp_min(1e-5)), torch.log(b.clamp_min(1e-5)))
        total = total + torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-5)
    return total / 3


def losses(model, batch, options, seed):
    ids, lengths, aln, sid, durations, target = batch
    out = model.forward_grad(ids, lengths, aln, sid, seed=seed, decode=options.spectral_weight > 0)
    duration = F.smooth_l1_loss(torch.log1p(out['dur']), torch.log1p(durations))
    spectral = spectral_loss(out['audio'], target) if options.spectral_weight else duration.new_zeros(())
    loss = options.duration_weight * duration + options.spectral_weight * spectral
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite objective; preserve the failed run for diagnosis')
    return loss, dict(duration=float(duration.detach()), spectral=float(spectral.detach()))


def train(options: TrainOptions):
    options.validate()
    root = Path(options.output).absolute()
    model_path, dataset_path = Path(options.model).resolve(strict=True), Path(options.dataset).resolve(strict=True)
    if dataset_path.is_file():
        dataset_path = dataset_path.parent
    if root.resolve().is_relative_to(model_path) or root.resolve().is_relative_to(dataset_path):
        raise ValueError('Run output must be outside source model and prepared dataset')
    resume_contract = None
    if options.resume:
        if not root.is_dir():
            raise ValueError('Resume requires the existing run directory')
        resume_contract = json.loads((root/'run-contract.json').read_text(encoding='utf-8'))
        if not isinstance(resume_contract, dict) or resume_contract.get('execution_profile') != TRAINING_EXECUTION_PROFILE:
            raise ValueError('Training execution profile mismatch or missing; start a new run instead of cross-profile resume')
    metadata, rows, dataset_sha = load_rows(dataset_path)
    torch.set_num_threads(options.threads)
    if options.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable; no silent CPU fallback')
    device = torch.device(options.device)
    model = load_mini(model_path, trainable_groups=options.groups)
    # The published inference loader sets its own thread count while importing.
    torch.set_num_threads(options.threads)
    migration = None
    if options.vocab_migration:
        migration = json.loads(Path(options.vocab_migration).read_text(encoding='utf-8'))
        model.migrate_vocab(migration['phone_to_id'], seed=migration.get('seed',42), copy_from=migration.get('copy_from'))
    model = model.to(device)
    if metadata.get('vocab') != model.vocab:
        raise ValueError('Prepared vocabulary differs from model; supply its explicit vocabulary migration')
    # Refuse bad data/coverage before creating a run or taking an optimizer step.
    for split in rows.values():
        for row in split:
            model.phoneme_ids(row['_phones'])
            if sum(row['_durations']) > options.max_frames:
                raise ValueError('Prepared clip exceeds frame cap')
    config = asdict(options)
    config.update(model=str(model_path),dataset=str(dataset_path))
    contract = {k:v for k,v in config.items() if k not in ('steps','resume','output','checkpoint_every','validation_every','max_validation_clips')}
    contract.update(execution_profile=TRAINING_EXECUTION_PROFILE,
                    dataset_sha256=dataset_sha, source_identity=model.source_identity,
                    frontend=metadata.get('frontend'), vocabulary_migration=migration,
                    validation_is_not_training_data=True)
    contract = json.loads(json.dumps(contract))
    contract_sha = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if options.resume:
        if resume_contract != contract:
            raise ValueError('Resume contract differs: source/data/voice/groups/loss/settings changed')
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root/'checkpoints').mkdir()
        atomic_json(root/'run-contract.json', contract)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=options.learning_rate, weight_decay=0.)
    step = 0
    if options.resume:
        path = Path(options.resume).resolve(strict=True)
        state = validate_controller_checkpoint(root, path, contract)
        step = load_checkpoint(model, path, optimizer=optimizer)['step']
        if state['step'] != step:
            raise ValueError('Controller step differs')
    if step >= options.steps:
        raise ValueError('Requested final step must exceed the restored step')
    start = time.monotonic()
    model.train()
    def checkpoint(current):
        destination = root/'checkpoints'/f'step-{current:07d}.pt'
        # A crash may leave a weight file without its committed controller receipt.
        # Preserve that evidence and use a new name when replaying the step.
        if destination.exists() or destination.with_suffix('.controller.json').exists():
            destination = destination.with_name(destination.stem+'-recovery-'+uuid.uuid4().hex+'.pt')
        save_checkpoint(model, destination, step=current, optimizer=optimizer)
        atomic_json(destination.with_suffix('.controller.json'),dict(step=current,
            contract_sha256=contract_sha, checkpoint_sha256=sha256(destination),
            scheduling='Counter-based deterministic shuffle and per-microbatch seeds; no global RNG dependency'))
        atomic_json(root/'latest.json',dict(step=current, checkpoint=str(destination), sha256=sha256(destination)))
    while step < options.steps:
        if shutil.disk_usage(root).free < 1536 * 2**20:
            raise ValueError('Less than 1.5 GiB free; preserve this run and reclaim storage before resuming')
        optimizer.zero_grad(set_to_none=True)
        values = []
        for micro in range(options.gradient_accumulation):
            index = step * options.gradient_accumulation + micro
            epoch, offset = divmod(index, len(rows['train']))
            order = list(range(len(rows['train'])))
            random.Random(options.seed + epoch).shuffle(order)
            row = rows['train'][order[offset]]
            batch = make_batch(model, row, options.voice, device, options.max_frames)
            loss, report = losses(model,batch,options,(options.seed+index) % (2**32))
            (loss/options.gradient_accumulation).backward()
            values.append(report)
        parameters = [p for p in model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        elapsed = time.monotonic()-start
        report = dict(utc=utc(), step=step, final_step=options.steps,
            duration=sum(x['duration'] for x in values)/len(values), spectral=sum(x['spectral'] for x in values)/len(values),
            gradient_norm=float(norm), elapsed_seconds=elapsed,
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
        if step % options.validation_every == 0 or step == options.steps:
            model.eval()
            vals = []
            with torch.no_grad():
                for row in rows['validation'][:options.max_validation_clips]:
                    batch = make_batch(model,row,options.voice,device,options.max_frames)
                    value,_ = losses(model,batch,options,options.seed)
                    vals.append(float(value))
            report['validation_loss'] = sum(vals)/len(vals)
            report['validation_clips'] = len(vals)
            report['validation_mode'] = 'reference-aligned reconstruction; not text-only intelligibility'
            model.train()
        with (root/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(report,allow_nan=False)+'\n')
        atomic_json(root/'status.json',dict(state='training',**report))
        print(json.dumps(report),flush=True)
        if step % options.checkpoint_every == 0 or step == options.steps:
            checkpoint(step)
    model.eval()
    atomic_json(root/'status.json',dict(state='complete',utc=utc(),step=step,
        elapsed_seconds=time.monotonic()-start, trainability=model.trainability_report(),
        public_weights_changed=False, language_quality_approved=False))
    return root
