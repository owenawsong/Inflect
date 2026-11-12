"""Atomic JSON policy contracts without importing Torch/model code."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
import time
import pytest

P = Path(__file__).resolve().parent
SOURCE = P.parent / "training.py"


def load_helper(path, *, os_module=os, time_module=time, uuid_module=uuid):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "atomic_json")
    namespace = dict(Path=Path, json=json, os=os_module, time=time_module, uuid=uuid_module)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["atomic_json"]


loader = SimpleNamespace(load=load_helper)


def error(code):
    value = OSError("synthetic sharing contract")
    value.winerror = code
    return value


def helper(replace, sleeps, platform="nt", nonce=uuid):
    return loader.load(SOURCE, os_module=SimpleNamespace(name=platform, getpid=os.getpid, replace=replace),
        time_module=SimpleNamespace(sleep=sleeps.append), uuid_module=nonce)


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_sharing_then_UTF8_success(tmp_path, code):
    target = tmp_path / "status.json"
    target.write_bytes(b"old")
    calls, sleeps, committed = [], [], []
    def replace(src, dest):
        calls.append(str(src))
        if len(calls) < 3: raise error(code)
        committed.append((Path(dest), Path(src).read_bytes()))
    value = {"text": "Za\u017c\u00f3\u0142\u0107 g\u0119\u015bl\u0105 ja\u017a\u0144 \u2192", "step": 55}
    assert helper(replace, sleeps)(target, value) is None
    assert committed == [(target, (json.dumps(value, indent=2, allow_nan=False)+"\n").encode("utf-8"))]
    assert json.loads(committed[0][1].decode("utf-8")) == value
    assert target.read_bytes() == b"old"
    assert sleeps == [.01, .02] and len(set(calls)) == 1
    assert not list(tmp_path.glob("*.pending"))


@pytest.mark.parametrize("code", [5, 32, 33])
def test_exhaustion_old_target_preserved_cleanup(tmp_path, code):
    target = tmp_path / "status.json"
    target.write_bytes(b"original")
    calls, sleeps = [], []
    def replace(src, dest):
        calls.append(str(src))
        raise error(code)
    with pytest.raises(OSError): helper(replace, sleeps)(target, {"new": True})
    assert len(calls) == 8 and len(sleeps) == 7 and sum(sleeps) == pytest.approx(.81)
    assert target.read_bytes() == b"original" and not list(tmp_path.glob("*.pending"))


@pytest.mark.parametrize("code,platform", [(2,"nt"),(6,"nt"),(13,"nt"),(87,"nt"),(None,"nt"),(5,"posix")])
def test_other_error_or_OS_immediate(tmp_path, code, platform):
    calls, sleeps = [], []
    def replace(src, dest):
        calls.append(src)
        raise error(code)
    with pytest.raises(OSError): helper(replace,sleeps,platform)(tmp_path / "status.json", {})
    assert len(calls) == 1 and sleeps == [] and list(tmp_path.iterdir()) == []


def test_nan_preserves_old_without_temp(tmp_path):
    target = tmp_path / "status.json"
    target.write_bytes(b"old")
    with pytest.raises(ValueError): loader.load(SOURCE)(target, {"bad":float("nan")})
    assert target.read_bytes() == b"old" and len(list(tmp_path.iterdir())) == 1


def test_collision_never_deletes_another_pending(tmp_path):
    target = tmp_path / "status.json"
    other = tmp_path / (".status.json."+str(os.getpid())+".fixed.pending")
    other.write_bytes(b"other writer")
    calls = []
    with pytest.raises(FileExistsError):
        helper(lambda *a:calls.append(a), [], nonce=SimpleNamespace(uuid4=lambda:SimpleNamespace(hex="fixed")))(target,{})
    assert other.read_bytes() == b"other writer" and calls == []


def test_write_error_cleans_owned_partial(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    target.write_bytes(b"old")
    original = Path.open
    class Broken:
        def __init__(self, stream): self.stream=stream
        def __enter__(self): return self
        def write(self, value):
            self.stream.write("partial")
            raise OSError("write failure")
        def __exit__(self,*args): self.stream.close()
    def opened(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        return Broken(stream) if args and args[0]=="x" else stream
    monkeypatch.setattr(Path,"open",opened)
    with pytest.raises(OSError,match="write failure"): loader.load(SOURCE)(target,{})
    assert target.read_bytes()==b"old" and not list(tmp_path.glob("*.pending"))


def test_concurrent_writers_unique_complete_JSON(tmp_path):
    target=tmp_path / "status.json"
    function=loader.load(SOURCE)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i:function(target,{"writer":i,"payload":"Za\u017c\u00f3\u0142\u0107 \u2192"*128}),range(20)))
    value = json.loads(target.read_text(encoding="utf-8"))
    assert value["writer"] in range(20) and value["payload"] == "Za\u017c\u00f3\u0142\u0107 \u2192"*128
    assert target.read_bytes() == (json.dumps(value, indent=2, allow_nan=False)+"\n").encode("utf-8")
    assert not list(tmp_path.glob("*.pending"))


def test_training_execution_profile_literal_preserved():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TRAINING_EXECUTION_PROFILE" for t in n.targets))
    assert ast.literal_eval(value) == "native24-reflect-cat-v1"
