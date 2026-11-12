"""Strict local factories for the published native24 Mini architecture."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import sys
import threading
from typing import Iterator

import torch
from torch import nn


SCHEMA = "inflect-native-mini-phonemes-v1"
VOICES = ("hope", "sophia", "cassidy", "james", "jasper", "mark")
PUBLIC_WEIGHTS_SHA256 = "8460e7d54a2fbf7fb54b3c7be7569c493c852de9a2aa1a0fd9fd9a099de581ed"
COUNTS = {"text": 11_914_548, "bridge": 2_728_732, "generator": 19_667_510}
GROUPS = ("text_encoder", "duration", "pitch_noise", "speaker", "bridge", "generator")
_ROOTS = (
    "native_mini_inference", "native33_text_model", "native26_text_model",
    "native33_text_fast", "inflect_kokoro_student27", "native27_nativegen_bridge",
    "qualify_native27_native_generator_budget", "student_decoder", "student", "kokoro",
)
_IMPORT_LOCK = threading.RLock()
_RNG_LOCK = threading.RLock()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_model_dir(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    for relative in ("", "models/mini", "baseline/models/mini", "core/baseline/models/mini"):
        candidate = root / relative
        if (candidate / "weights.pt").is_file() and (candidate / "code").is_dir():
            return candidate
    raise ValueError("Supply a local Mini v3 package or models/mini directory; no downloads occur")


def validate_inventory(path: str | Path, expected_weights_sha256: str | None = None) -> tuple[Path, dict]:
    root = resolve_model_dir(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == SCHEMA, "Not a Mini v3 model-core manifest")
    entries = manifest.get("files")
    require(isinstance(entries, dict) and "weights.pt" in entries, "Missing model inventory")
    for name, digest in entries.items():
        require(isinstance(name, str) and "\\" not in name, "Unsafe inventory path")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ":" not in name and ".." not in relative.parts,
                "Unsafe inventory path")
        file = root / name
        require(file.resolve().is_relative_to(root) and file.is_file() and not file.is_symlink(),
                "Inventory file missing or outside model directory: " + name)
        require(isinstance(digest, str) and sha256(file) == digest, "Inventory hash mismatch: " + name)
    require("code/native_mini_inference.py" in entries, "Inference factory missing from inventory")
    actual_code = {p.relative_to(root).as_posix() for p in (root / "code").rglob("*.py")}
    require(actual_code <= set(entries), "Uninventoried Python source in model directory")
    if expected_weights_sha256 is not None:
        require(entries["weights.pt"] == expected_weights_sha256, "Expected weight pin differs")
    identity = {"manifest_sha256": sha256(root / "manifest.json"),
                "weights_sha256": entries["weights.pt"],
                "code_sha256": {k: v for k, v in entries.items() if k.startswith("code/")}}
    return root, identity


def _owned_module(name: str, roots=_ROOTS) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


class _VerifiedSourceLoader:
    def __init__(self, root: Path, source: Path, digest: str):
        self.root, self.source, self.digest = root, source, digest

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        require(self.source.resolve().is_relative_to(self.root) and self.source.is_file()
                and not self.source.is_symlink(), "Unsafe runtime source: " + str(self.source))
        # Compile the same bytes that were verified; never consult or write pyc caches.
        data = self.source.read_bytes()
        require(hashlib.sha256(data).hexdigest() == self.digest,
                "Runtime source changed after inventory validation: " + str(self.source))
        exec(compile(data, str(self.source), "exec", dont_inherit=True), module.__dict__)


class _VerifiedSourceFinder:
    def __init__(self, root: Path, files: dict[str, str]):
        self.root, self.files = root, dict(files)
        self.roots = set(_ROOTS) | {name.removeprefix("code/").split("/", 1)[0].removesuffix(".py")
                                    for name in files if name.startswith("code/") and name.endswith(".py")}

    def find_spec(self, fullname, path=None, target=None):
        if not _owned_module(fullname, self.roots):
            return None
        base = "code/" + fullname.replace(".", "/")
        package = base + "/__init__.py"
        relative = package if package in self.files else base + ".py"
        # An owned name must not fall through to bytecode, native extensions or sys.path.
        if relative not in self.files:
            raise ImportError("No inventoried runtime source for " + fullname)
        source = self.root / relative
        return importlib.util.spec_from_file_location(
            fullname, source, loader=_VerifiedSourceLoader(self.root, source, self.files[relative]),
            submodule_search_locations=[str(source.parent)] if relative == package else None)


@contextmanager
def runtime_imports(root: Path, files: dict[str, str]) -> Iterator[object]:
    """Temporarily isolate the release's legacy absolute imports, then restore them."""
    with _IMPORT_LOCK:
        finder = _VerifiedSourceFinder(root, files)
        previous = {name: module for name, module in sys.modules.items() if _owned_module(name, finder.roots)}
        old_path = sys.path[:]
        old_meta_path = sys.meta_path[:]
        for name in previous:
            del sys.modules[name]
        # Owned sources resolve through the finder, not a general-purpose code/ path
        # that could let unlisted bytecode shadow an ordinary installed dependency.
        code_root = (root / "code").resolve()
        sys.path[:] = [entry for entry in old_path if Path(entry or ".").resolve() != code_root]
        sys.meta_path.insert(0, finder)
        try:
            runtime = importlib.import_module("native_mini_inference")
            yield runtime
        finally:
            for name in list(sys.modules):
                if _owned_module(name, finder.roots):
                    del sys.modules[name]
            sys.modules.update(previous)
            sys.path[:] = old_path
            sys.meta_path[:] = old_meta_path


def _load_native(path: str | Path, expected_weights_sha256: str | None = None):
    root, identity = validate_inventory(path, expected_weights_sha256)
    require(str(torch.get_default_device()) == "cpu", "Construct on CPU before explicit device movement")
    with runtime_imports(root, identity["code_sha256"]) as runtime, torch.random.fork_rng(devices=[]):
        native = runtime.load_export(root, device="cpu", fast=False)
    require(native.text_kind == "native33", "Only the released native33 Mini architecture is supported")
    for name, count in COUNTS.items():
        require(sum(p.numel() for p in getattr(native, name).parameters()) == count,
                "Unexpected parameter count: " + name)
    return root, identity, native


def load_public_runtime(path: str | Path, *, expected_weights_sha256: str | None = None):
    """Load the unchanged public inference class for independent parity checks."""
    return _load_native(path, expected_weights_sha256)[2]


def _clone_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return deepcopy(value)


def contract_digest(value) -> str:
    """Hash nested metadata and fixed conditioning without pickle-container dependence."""
    digest = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            header = json.dumps([str(tensor.dtype), list(tensor.shape)]).encode()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(b"tensor" + len(header).to_bytes(8, "big") + header)
            digest.update(len(raw).to_bytes(8, "big") + raw)
        elif isinstance(item, dict):
            digest.update(b"dict" + len(item).to_bytes(8, "big"))
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + len(item).to_bytes(8, "big"))
            for child in item:
                add(child)
        else:
            encoded = json.dumps(item, sort_keys=True, allow_nan=False).encode()
            digest.update(type(item).__name__.encode() + len(encoded).to_bytes(8, "big") + encoded)

    add(value)
    return digest.hexdigest()


class MiniV3Bridge(nn.Module):
    """Trainable v3 modules with unchanged full-utterance clocks and conditioning.

    No predicted alignment is used as a supervised duration label. Caller supplies
    hard monotonic alignment at 600 waveform samples per frame. Batch size is one:
    padding unequal utterances changes the released instance-normalized acoustics.
    """

    def __init__(self, root: Path, identity: dict, native: nn.Module):
        super().__init__()
        self.text, self.bridge, self.generator = native.text, native.bridge, native.generator
        self.model_dir, self.source_identity = root, deepcopy(identity)
        self.styles, self.vocab = _clone_cpu(native.styles), deepcopy(native.vocab)
        self.max_frames = native.max_frames
        self._native_type = type(native)
        self._template = {
            "schema": SCHEMA, "text_kind": native.text_kind,
            "text_config": deepcopy(self.text.get_config()),
            "generator_config": deepcopy(self._native_type.__init__.__globals__["GENERATOR_CONFIG"]),
            "styles": self.styles, "vocab": self.vocab, "max_frames": self.max_frames,
            "sample_rate": 24000,
            "choices_sha256": self._native_type.__init__.__globals__["CHOICES_SHA"],
        }
        self._contract_sha256 = contract_digest(self._template)
        self._render_lock = threading.RLock()
        self._vocabulary_locked = False
        self.vocabulary_migrations: list[dict] = []
        self.trainable_groups: tuple[str, ...] = ()
        self.configure_trainable(())

    @staticmethod
    def parameter_group(name: str) -> str:
        if name.startswith("bridge."):
            return "bridge"
        if name.startswith("generator."):
            return "generator"
        require(name.startswith("text."), "Unknown parameter namespace: " + name)
        name = name.removeprefix("text.")
        if name.startswith("speaker_embedding."):
            return "speaker"
        if name.startswith(("text_encoder.", "asr_proj.")):
            return "text_encoder"
        if name.startswith(("bert.", "bert_encoder.", "predictor.text_encoder.",
                            "predictor.lstm.", "predictor.duration_proj.")):
            return "duration"
        require(name.startswith("predictor."), "Unclassified text parameter: " + name)
        return "pitch_noise"

    def configure_trainable(self, groups) -> dict:
        require(not isinstance(groups, str), "Supply a sequence of trainable group names")
        names = tuple(groups)
        require(len(set(names)) == len(names) and set(names) <= set(GROUPS), "Invalid trainable groups")
        self.trainable_groups = tuple(name for name in GROUPS if name in names)
        for name, parameter in self.named_parameters():
            # The released generator computes its harmonic excitation under no_grad.
            active = self.parameter_group(name) in names and not name.startswith("generator.m_source.")
            parameter.requires_grad_(active)
            parameter.grad = None
        self.train(self.training)
        return self.trainability_report()

    def train(self, mode: bool = True):
        super().train(mode)
        for name in ("text", "bridge", "generator"):
            module = getattr(self, name)
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        return self

    def trainability_report(self) -> dict:
        groups = {name: {"parameters": 0, "trainable": 0} for name in GROUPS}
        for name, parameter in self.named_parameters():
            row = groups[self.parameter_group(name)]
            row["parameters"] += parameter.numel()
            row["trainable"] += parameter.numel() * int(parameter.requires_grad)
        return {"core_parameters": sum(p.numel() for p in self.parameters()), "groups": groups,
                "fixed_conditioning_trainable": False,
                "duration_requires_explicit_supervision": True,
                "harmonic_excitation_differentiable": False,
                "duration_group_includes_shared_pitch_context": True}

    def phoneme_ids(self, phonemes: str) -> list[int]:
        require(isinstance(phonemes, str) and 1 <= len(phonemes) <= 510, "Phoneme length must be 1..510")
        unknown = sorted(set(phonemes) - set(self.vocab))
        require(not unknown, "Unknown phonemes; no dropping or automatic ID reuse: " + repr(unknown))
        return [0, *(self.vocab[phone] for phone in phonemes), 0]

    def migrate_vocab(self, phone_to_unused_id: dict[str, int], *, seed: int = 42,
                      copy_from: dict[str, str] | None = None) -> dict:
        """Explicit CPU, pre-training reuse of currently unmapped native178 IDs.

        Call BEFORE constructing an optimizer or calling forward_grad. Example:
        model.migrate_vocab({new_phone: unused_id}, seed=42,
                            copy_from={new_phone: explicitly_related_old_phone})

        With no copy_from entry, only the new ID's two embedding rows receive
        deterministic N(0, width**-0.5) values. A copy_from entry copies the exact
        old phone row in both embeddings, with no implicit linguistic inference.
        All other rows, the 178-row factories and original mappings are preserved.
        This does not certify a frontend's phoneme coverage or language quality.
        Returned receipt binds both contracts and rows to the warm-start hashes.
        Reconstruct the SAME migration when restoring a training checkpoint.
        """
        require(not self._vocabulary_locked and not any(p.grad is not None for p in self.parameters()),
                "Vocabulary migration is pre-training only; use a fresh warm start")
        require(all(p.device.type == "cpu" for p in self.parameters()), "Migrate vocabulary on CPU before device movement")
        require(type(seed) is int and 0 <= seed < 2**32, "seed must be uint32")
        require(isinstance(phone_to_unused_id, dict) and bool(phone_to_unused_id), "Explicit nonempty phone-to-ID map required")
        mapping = dict(phone_to_unused_id)
        require(all(isinstance(phone, str) and len(phone) == 1 and phone.isprintable()
                    and not phone.isspace() and phone not in self.vocab for phone in mapping),
                "Add only new single-character phones; never replace existing mappings")
        used = set(self.vocab.values())
        require(all(type(index) is int and 1 <= index < 178 and index not in used for index in mapping.values())
                and len(set(mapping.values())) == len(mapping), "Use distinct IDs absent from the current vocabulary, within 1..177")
        require(copy_from is None or isinstance(copy_from, dict), "copy_from must explicitly map new phones to existing phones")
        sources = dict(copy_from or {})
        require(set(sources) <= set(mapping)
                and all(isinstance(phone, str) and phone in self.vocab for phone in sources.values()),
                "copy_from targets must be existing phones, not chained additions")
        self._check_contract()
        parameters = dict(self.text.named_parameters())
        keys = ("bert.embeddings.word_embeddings.weight", "text_encoder.embedding.weight")
        require(all(key in parameters and parameters[key].ndim == 2 and parameters[key].shape[0] == 178
                    and parameters[key].dtype == torch.float32 for key in keys), "Exact native178 FP32 embeddings required")
        changed_ids = set(mapping.values())
        retained = torch.tensor([index for index in range(178) if index not in changed_ids], dtype=torch.long)
        original_rows = {key: parameters[key].detach().clone() for key in keys}
        retained_hashes = {key: contract_digest(value[retained]) for key, value in original_rows.items()}
        new_rows = {}
        for key in keys:
            for phone, index in mapping.items():
                if phone in sources:
                    row = original_rows[key][self.vocab[sources[phone]]].clone()
                else:
                    identity = json.dumps([seed, phone, index, key], ensure_ascii=True).encode()
                    local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
                    generator = torch.Generator(device="cpu").manual_seed(local_seed)
                    row = torch.randn(parameters[key].shape[1], generator=generator) * parameters[key].shape[1] ** -0.5
                new_rows[key, index] = row
        previous_contract = self._contract_sha256
        with torch.no_grad():
            for (key, index), row in new_rows.items():
                parameters[key][index].copy_(row)
        require(all(contract_digest(parameters[key].detach()[retained]) == retained_hashes[key] for key in keys),
                "Embedding rows outside the explicit new IDs changed")
        self.vocab = {**self.vocab, **mapping}
        self._template = {**self._template, "vocab": self.vocab}
        self._contract_sha256 = contract_digest(self._template)
        receipt = {
            "schema": "inflect-mini-v3-unused-id-migration-v1", "version": len(self.vocabulary_migrations) + 1,
            "source_weights_sha256": self.source_identity["weights_sha256"],
            "source_manifest_sha256": self.source_identity["manifest_sha256"],
            "base_contract_sha256": previous_contract, "new_contract_sha256": self._contract_sha256,
            "phone_to_id": mapping, "copy_from": sources, "seed": seed,
            "initializer": "per-phone-and-embedding-seeded-normal-or-explicit-old-row-copy-v1",
            "retained_rows_sha256": retained_hashes,
            "new_rows_sha256": {key: {str(index): contract_digest(row) for (name, index), row in new_rows.items() if name == key}
                                for key in keys},
            "base_rows_bit_identical": True, "factory_rows": 178,
            "frontend_coverage_validated": False, "language_quality_claimed": False,
        }
        self.vocabulary_migrations.append(deepcopy(receipt))
        return receipt

    def _inputs(self, ids, lengths, sid):
        require(isinstance(ids, torch.Tensor) and ids.dtype == torch.long and ids.ndim == 2
                and ids.shape[0] == 1 and 3 <= ids.shape[1] <= 512, "ids must be LongTensor [1,T], 3<=T<=512")
        require(isinstance(lengths, torch.Tensor) and lengths.dtype == torch.long
                and lengths.shape == (1,) and int(lengths[0]) == ids.shape[1], "No padded utterances; lengths must be [T]")
        require(isinstance(sid, torch.Tensor) and sid.dtype == torch.long and sid.shape == (1,)
                and 0 <= int(sid[0]) < 6, "sid must be LongTensor [1] in 0..5")
        device = next(self.text.parameters()).device
        require(all(t.device == device for t in (ids, lengths, sid)), "Inputs and model must share a device")
        require(int(ids[0, 0]) == int(ids[0, -1]) == 0
                and bool(((ids[:, 1:-1] > 0) & (ids[:, 1:-1] < 178)).all()),
                "Exact boundary-wrapped native178 IDs required")
        supported = torch.tensor(sorted(set(self.vocab.values())), dtype=torch.long, device=device)
        require(bool(torch.isin(ids[:, 1:-1], supported).all()), "Unmapped phone ID; supply an explicit vocabulary migration first")

    def _alignment(self, ids, aln):
        require(isinstance(aln, torch.Tensor) and aln.dtype == next(self.text.parameters()).dtype
                and aln.device == ids.device and aln.ndim == 3 and aln.shape[:2] == ids.shape
                and 2 <= aln.shape[2] <= self.max_frames, "aln must be model-dtype [1,T,F] within frame cap")
        require(not aln.requires_grad and bool(torch.isfinite(aln).all())
                and bool(((aln == 0) | (aln == 1)).all()) and bool((aln.sum(1) == 1).all()),
                "Supply a fixed hard one-hot alignment, not soft or predicted labels")
        path = aln.argmax(1)[0]
        require(bool((path[1:] >= path[:-1]).all()) and bool((aln.sum(2) >= 1).all()),
                "Alignment must be monotonic and cover every token, including boundaries")

    @contextmanager
    def _seed(self, seed: int):
        require(type(seed) is int and 0 <= seed < 2**32, "seed must be uint32")
        device = next(self.text.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with _RNG_LOCK, torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            yield

    def _decode(self, ids, sid, asr, f0, noise):
        hidden = self.bridge.forward_hidden(asr, f0, noise, sid)
        index = int(sid[0])
        donor = self._native_type.__init__.__globals__["DONORS"][index]
        style = (self.styles["banks"][donor][ids.shape[1] - 3].to(asr.device)
                 + self.styles["offsets"][index].to(asr.device) * self.styles["strengths"][index])[None]
        # Preserve exactly the public voice-specific affine hooks, with cleanup.
        with self._native_type._affines(self, VOICES[index]):
            audio = self.generator(hidden, style, f0).reshape(1, -1)
        require(audio.shape == (1, asr.shape[2] * 600) and bool(torch.isfinite(audio).all()),
                "Native audio clock or finite-value check failed")
        return hidden, audio

    def forward_grad(self, ids, lengths, aln, sid, *, seed: int = 42, decode: bool = True) -> dict:
        """Differentiable teacher-aligned forward; does not detach learned features.

        Duration output [1,T]; f0/noise [1,2F]; asr [1,512,F]; audio [1,600F].
        F0 remains differentiable through the bridge, not the harmonic-source path.
        Frozen modules still pass gradients to their inputs. Seed also fixes any
        training dropout and is caller-controlled, not an implicit training policy.
        """
        self._inputs(ids, lengths, sid)
        self._alignment(ids, aln)
        self._vocabulary_locked = True
        with self._render_lock, self._seed(seed):
            dur, f0, noise, asr_tok = self.text(ids, lengths, aln=aln, speaker_ids=sid)
            asr = asr_tok @ aln
            result = {"dur": dur, "f0": f0, "noise": noise, "asr": asr,
                      "asr_tok": asr_tok, "aln": aln, "sample_rate": 24000}
            if decode:
                result["hidden"], result["audio"] = self._decode(ids, sid, asr, f0, noise)
            return result

    def forward(self, ids, lengths, aln, sid, *, seed: int = 42, decode: bool = True):
        return self.forward_grad(ids, lengths, aln, sid, seed=seed, decode=decode)

    @torch.no_grad()
    def synthesize_ids(self, ids: list[int], voice: str = "hope", *, seed: int = 42, speed: float = 1.) -> dict:
        require(not any(module.training for module in self.modules()), "Call eval() before synthesis")
        require(voice in VOICES, "Unknown voice")
        require(type(speed) in (int, float) and math.isfinite(speed) and speed > 0, "Invalid speed")
        require(isinstance(ids, list) and all(type(i) is int for i in ids), "Supply integer ID list")
        device = next(self.text.parameters()).device
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
        sid = torch.tensor([VOICES.index(voice)], dtype=torch.long, device=device)
        self._inputs(tokens, lengths, sid)
        with self._render_lock:
            dur = self.text(tokens, lengths, speaker_ids=sid)[0]
            require(bool(torch.isfinite(dur).all()), "Invalid predicted duration")
            frames = int((dur / speed).round().clamp(min=1).sum())
            require(2 <= frames <= self.max_frames, "Predicted duration exceeds frame cap")
            features = self.text.infer(tokens, lengths, sid, speed=speed)
            require(int(features["frame_lengths"][0]) == frames, "Predicted inference clock differs")
            with self._seed(seed):
                _, audio = self._decode(tokens, sid, features["asr"], features["F0"], features["N"])
            require(bool(audio.abs().max() < 1), "Clipping rejected; no gain/filter fallback")
            return {"audio": audio.reshape(-1).cpu(), "frames": frames, "sample_rate": 24000, "voice": voice}

    def _check_contract(self) -> dict:
        require(contract_digest(self._template) == self._contract_sha256,
                "Fixed vocabulary, factory metadata or voice conditioning was changed")
        current = {**self._template, "styles": self.styles, "vocab": self.vocab,
                   "max_frames": self.max_frames, "text_config": self.text.get_config()}
        require(contract_digest(current) == self._contract_sha256,
                "Fixed vocabulary, factory metadata or voice conditioning was replaced")
        return current

    def inference_payload(self) -> dict:
        current = self._check_contract()
        payload = _clone_cpu(current)
        for name in COUNTS:
            state = _clone_cpu(getattr(self, name).state_dict())
            require(all(bool(torch.isfinite(t).all()) for t in state.values()), "Nonfinite model state")
            payload[name + "_state"] = state
        return payload


def load_mini(path: str | Path, *, trainable_groups=(), expected_weights_sha256: str | None = None) -> MiniV3Bridge:
    """Strict CPU warm start. No architecture migration, downloads or optimizer reset."""
    root, identity, native = _load_native(path, expected_weights_sha256)
    result = MiniV3Bridge(root, identity, native)
    result.configure_trainable(trainable_groups)
    return result
