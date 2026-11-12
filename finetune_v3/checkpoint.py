"""Strict adaptation checkpoints and public-loader-compatible model-core exports."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import torch

from .model import COUNTS, SCHEMA, MiniV3Bridge, _clone_cpu, contract_digest, require, sha256


TRAINING_FORMAT = "inflect-mini-v3-adaptation-v1"


def _new_destination(model: MiniV3Bridge, path: str | Path) -> Path:
    destination = Path(path).absolute()
    require(not destination.exists(), "Refusing to overwrite an existing output")
    require(not destination.resolve().is_relative_to(model.model_dir), "Do not write into the source model")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _optimizer_names(model: MiniV3Bridge, optimizer) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result, seen = [], set()
    for group in optimizer.param_groups:
        current = []
        for parameter in group["params"]:
            require(id(parameter) in names and parameter.requires_grad, "Optimizer includes foreign or frozen parameters")
            name = names[id(parameter)]
            require(name not in seen, "Optimizer contains duplicate parameters")
            seen.add(name)
            current.append(name)
        result.append(current)
    require(seen == {name for name, p in model.named_parameters() if p.requires_grad},
            "Optimizer must contain exactly all selected trainable parameters")
    return result


def _validate_payload(model: MiniV3Bridge, payload: dict) -> None:
    expected = model.inference_payload()
    require(isinstance(payload, dict) and set(payload) == set(expected), "Inference payload fields differ")
    fixed = {key: value for key, value in payload.items() if key not in {n + "_state" for n in COUNTS}}
    require(contract_digest(fixed) == model._contract_sha256, "Vocabulary/config/conditioning differs")
    for part in COUNTS:
        key = part + "_state"
        state = payload[key]
        require(isinstance(state, dict) and set(state) == set(expected[key]), "Checkpoint state keys differ: " + part)
        for name, value in state.items():
            target = expected[key][name]
            require(torch.is_tensor(value) and value.device.type == "cpu"
                    and value.shape == target.shape and value.dtype == target.dtype
                    and bool(torch.isfinite(value).all()), "Checkpoint tensor differs: " + part + "." + name)


def save_checkpoint(model: MiniV3Bridge, path: str | Path, *, step: int, optimizer=None) -> Path:
    """Save model/optimizer identity, not a controller/scheduler/data-loader resume."""
    require(type(step) is int and step >= 0, "step must be a nonnegative integer")
    destination = _new_destination(model, path)
    payload = {
        "format": TRAINING_FORMAT, "step": step,
        "source_identity": model.source_identity,
        "contract_sha256": model._contract_sha256,
        "trainable_groups": list(model.trainable_groups),
        "vocabulary_migrations": deepcopy(model.vocabulary_migrations),
        "inference_payload": model.inference_payload(),
        "optimizer_type": None if optimizer is None else type(optimizer).__module__ + "." + type(optimizer).__qualname__,
        "optimizer_parameter_names": None if optimizer is None else _optimizer_names(model, optimizer),
        "optimizer": None if optimizer is None else _clone_cpu(optimizer.state_dict()),
        "controller_resume_included": False,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".mini-checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard linking atomically fails if a destination appeared during the write.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(model: MiniV3Bridge, path: str | Path, *, optimizer=None) -> dict:
    """Validate identity and all tensors before copying any checkpoint weights."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    require(isinstance(payload, dict) and payload.get("format") == TRAINING_FORMAT,
            "Not a Mini v3 adaptation checkpoint; v2 VITS checkpoints are incompatible")
    require(payload.get("source_identity") == model.source_identity, "Warm-start source identity differs")
    require(payload.get("contract_sha256") == model._contract_sha256, "Model contract differs")
    require(payload.get("vocabulary_migrations", []) == model.vocabulary_migrations,
            "Reconstruct the same explicit vocabulary migration before restoring")
    require(payload.get("trainable_groups") == list(model.trainable_groups),
            "Configure the same trainable groups before loading")
    require(type(payload.get("step")) is int and payload["step"] >= 0, "Invalid checkpoint step")
    _validate_payload(model, payload.get("inference_payload"))
    if optimizer is not None:
        require(payload.get("optimizer_type") == type(optimizer).__module__ + "." + type(optimizer).__qualname__,
                "Optimizer class differs")
        require(payload.get("optimizer_parameter_names") == _optimizer_names(model, optimizer),
                "Optimizer parameter identity/order differs")
        require(isinstance(payload.get("optimizer"), dict), "Checkpoint has no optimizer state")
        # Validate optimizer structure on an independent copy before touching the
        # live model. Torch load_state_dict alone does not check moment shapes.
        candidate = deepcopy(optimizer)
        candidate.load_state_dict(payload["optimizer"])
        for parameter, state in candidate.state.items():
            for key, tensor in state.items():
                if torch.is_tensor(tensor):
                    require(bool(torch.isfinite(tensor).all()), "Nonfinite optimizer state")
                    if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
                        require(tensor.shape == parameter.shape and tensor.dtype == parameter.dtype,
                                "Optimizer moment shape/dtype differs")
        del candidate
    for part in COUNTS:
        getattr(model, part).load_state_dict(payload["inference_payload"][part + "_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    model._vocabulary_locked = True
    return {"step": payload["step"], "checkpoint_sha256": sha256(path), "controller_resume_included": False}


def export_model(model: MiniV3Bridge, path: str | Path) -> dict:
    """Write a NEW standalone model-core directory accepted by load_export.

    This is not an assembled full raw-text release: the parent integrates the
    unchanged frontend, license/notices and outer package inventory separately.
    """
    destination = _new_destination(model, path)
    # Rehash the source so changed code can never silently enter an export.
    from .model import validate_inventory

    _, identity = validate_inventory(model.model_dir)
    require(identity == model.source_identity, "Source package changed after loading")
    payload = model.inference_payload()
    _validate_payload(model, payload)
    temporary = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".mini-export-"))
    try:
        for relative, digest in model.source_identity["code_sha256"].items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model.model_dir / relative, target)
            require(sha256(target) == digest, "Source code changed while copying")
        torch.save(payload, temporary / "weights.pt")
        files = {relative: digest for relative, digest in model.source_identity["code_sha256"].items()}
        files["weights.pt"] = sha256(temporary / "weights.pt")
        manifest = {
            "schema": SCHEMA, "files": files,
            "receipt": {"core_parameters": sum(COUNTS.values()), "parameters": COUNTS,
                        "finetune_v3": {"format": TRAINING_FORMAT,
                                        "source_weights_sha256": identity["weights_sha256"],
                                        "source_manifest_sha256": identity["manifest_sha256"],
                                        "contract_sha256": model._contract_sha256,
                                        "trainable_groups": list(model.trainable_groups),
                                        "vocabulary_migrations": deepcopy(model.vocabulary_migrations),
                                        "quality_or_publication_approval": False}},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Same exact release loader validates the exported state/configuration.
        from .model import load_public_runtime

        reference = load_public_runtime(temporary)
        for part in COUNTS:
            exported = getattr(reference, part).state_dict()
            require(all(torch.equal(tensor, exported[key]) for key, tensor in payload[part + "_state"].items()),
                    "Public-loader tensor readback differs: " + part)
        del reference
        require(not destination.exists(), "Output appeared while export was being validated")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"path": str(destination), "weights_sha256": files["weights.pt"],
            "manifest_sha256": sha256(destination / "manifest.json"),
            "public_loader_strict_readback": True, "assembled_release": False}
