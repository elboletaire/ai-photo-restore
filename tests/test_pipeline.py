import argparse
from pathlib import Path

import cv2
import numpy as np
import pytest

import restore

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class FakeBackend:
    """Returns the input image unchanged and records each call."""

    name = "fake"

    def __init__(self):
        self.calls = []

    def _echo(self, stage, path):
        self.calls.append(stage)
        return cv2.imencode(".png", restore.read_image(path))[1].tobytes()

    def conservative(self, path, fidelity, upscale):
        return self._echo("conservative", path)

    def generative(self, path, prompt, seed, megapixels):
        return self._echo("generative", path)

    def colorize(self, path):
        return self._echo("colorize", path)


def args(**kw):
    defaults = dict(fidelity=0.7, no_upscale=False, generative=True, prompt="p", seed=42, megapixels=1.0,
                    colorize=True, tone="keep", no_sharpen=False, saturation=0.85, force=False)
    return argparse.Namespace(**{**defaults, **kw})


@pytest.fixture
def photo(tmp_path):
    src = tmp_path / "in" / "photo.jpg"
    src.parent.mkdir()
    img = (np.random.default_rng(0).random((64, 48, 3)) * 255).astype(np.uint8)
    restore.write_image(src, img)
    return src


def test_rerun_with_same_settings_is_cached(photo, tmp_path):
    b = FakeBackend()
    restore.restore_one(photo, tmp_path / "out", b, args())
    restore.restore_one(photo, tmp_path / "out", b, args())
    assert b.calls == ["conservative", "generative", "colorize"]


def test_changed_seed_reruns_generative_and_what_depends_on_it(photo, tmp_path):
    b = FakeBackend()
    restore.restore_one(photo, tmp_path / "out", b, args())
    b.calls.clear()
    # The fake backend echoes its input, so make the new generative output differ.
    b.generative = lambda *a: (b.calls.append("generative"),
                               cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes())[1]
    restore.restore_one(photo, tmp_path / "out", b, args(seed=7))
    assert b.calls == ["generative", "colorize"]


def test_changed_fidelity_reruns_conservative_only(photo, tmp_path):
    b = FakeBackend()
    restore.restore_one(photo, tmp_path / "out", b, args())
    b.calls.clear()
    restore.restore_one(photo, tmp_path / "out", b, args(fidelity=0.9))
    assert b.calls == ["conservative"]


def test_force_reruns_everything(photo, tmp_path):
    b = FakeBackend()
    restore.restore_one(photo, tmp_path / "out", b, args())
    b.calls.clear()
    restore.restore_one(photo, tmp_path / "out", b, args(force=True))
    assert b.calls == ["conservative", "generative", "colorize"]


def test_rescanned_original_reruns_everything(photo, tmp_path):
    b = FakeBackend()
    restore.restore_one(photo, tmp_path / "out", b, args())
    b.calls.clear()
    restore.write_image(photo, np.full((64, 48, 3), 128, np.uint8))
    restore.restore_one(photo, tmp_path / "out", b, args())
    assert b.calls == ["conservative", "generative", "colorize"]


def test_examples_are_readable():
    photos = restore.collect_inputs([str(EXAMPLES)])
    assert len(photos) == 3
    for p in photos:
        assert restore.read_image(p).ndim == 3


def test_output_names_number_duplicate_stems():
    files = [Path("a/photo.jpg"), Path("b/photo.jpg"), Path("a/Photo.png"), Path("a/other.jpg")]
    assert restore.output_names(files) == ["photo", "photo-2", "Photo-3", "other"]


def test_combo_options_reads_old_and_new_formats():
    assert restore.combo_options([["a.pth", "b.pth"]]) == ["a.pth", "b.pth"]
    assert restore.combo_options(["COMBO", {"options": ["a.pth"]}]) == ["a.pth"]
    assert restore.combo_options(["INT", {"default": 1}]) is None
    assert restore.combo_options(None) is None


def fake_object_info(missing_model=None):
    """object_info with every node of every workflow, listing each model the workflows use."""
    info = {}
    for wf in ("conservative", "generative", "colorize"):
        for node in restore.load_workflow(wf).values():
            required = info.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
            for key, value in node["inputs"].items():
                if isinstance(value, str) and value.endswith((".pth", ".safetensors")):
                    required[key] = ["COMBO", {"options": [] if value == missing_model else [value]}]
    return info


def test_check_only_looks_at_requested_stages(monkeypatch):
    info = fake_object_info(missing_model="qwen_image_edit_2511_fp8mixed.safetensors")
    monkeypatch.setattr(restore.requests, "get", lambda *a, **k: type("R", (), {"json": lambda self: info})())
    backend = restore.ComfyBackend("http://comfy")
    assert backend.check(["conservative", "colorize"]) == []
    assert backend.check(["generative"]) == [
        "generative: UNETLoader.unet_name = qwen_image_edit_2511_fp8mixed.safetensors not installed (run setup-local)"]
