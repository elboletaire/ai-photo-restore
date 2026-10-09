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
                    repair=False, mask=None, repair_prompt="r", colorize=True, tone="keep", no_sharpen=False, saturation=0.85, force=False)
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


def sepia(h=300, w=200, level=90):
    lab = np.zeros((h, w, 3), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = level / 2.55, 8, 25
    return restore.from_lab(lab)


def test_damage_mask_finds_light_holes_but_not_redrawn_areas():
    original = sepia()
    original[100:140, 60:120] = (200, 225, 235)  # bare paper where the emulsion peeled off
    generative = sepia()
    generative[20:50, 20:60] = 255  # the model brightened something that wasn't damaged
    m = restore.damage_mask(original, generative)
    assert m[120, 90] == 255
    assert m[35, 40] == 0


def test_damage_mask_ignores_the_scanner_bed():
    original = np.full((300, 200, 3), 225, np.uint8)  # grey scanner lid
    original[20:280, 20:180] = sepia(260, 160)
    generative = sepia()  # the model extends the picture over the scanner bed
    assert not restore.damage_mask(original, generative)[:20].any()


def test_repair_only_changes_the_masked_area(tmp_path):
    img = (np.random.default_rng(1).random((120, 90, 3)) * 255).astype(np.uint8)
    original, mask = tmp_path / "original.png", tmp_path / "mask.png"
    restore.write_image(original, img)
    m = np.zeros((120, 90), np.uint8)
    m[50:70, 30:60] = 255
    restore.write_image(mask, m)

    class Black:
        def repair(self, path, mask, prompt, seed, megapixels):
            crop = restore.read_image(path)
            return cv2.imencode(".png", np.zeros((crop.shape[0] * 2, crop.shape[1] * 2, 3), np.uint8))[1].tobytes()

    out = cv2.imdecode(np.frombuffer(restore.repair_damage(Black(), original, mask, args()), np.uint8), cv2.IMREAD_COLOR)
    assert cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[60, 45] < 40  # repaired (with the photo's tint)
    far = np.ones((120, 90), bool)
    far[40:80, 20:70] = False  # beyond the feathered edge
    assert np.array_equal(out[far], img[far])


def test_repair_runs_generative_and_keeps_a_user_mask(photo, tmp_path):
    mask = tmp_path / "mask.png"
    restore.write_image(mask, np.full((64, 48), 255, np.uint8))
    b = FakeBackend()
    b.repair = lambda path, *a: b._echo("repair", path)
    restore.restore_one(photo, tmp_path / "out", b, args(generative=False, repair=True, colorize=False, mask=str(mask)))
    assert b.calls == ["conservative", "generative", "repair", "conservative"]
    assert (tmp_path / "out" / "photo" / "27-repaired-final.jpg").exists()


def test_repair_adds_a_colour_version_without_replacing_the_generative_one(photo, tmp_path):
    b = FakeBackend()
    b.repair = lambda path, *a: b._echo("repair", path)
    bases = []
    b.colorize = lambda path: (bases.append(path.name), b._echo("colorize", path))[1]
    restore.restore_one(photo, tmp_path / "out", b, args(repair=True))
    assert bases == ["20-generative.png", "27-repaired.png"]
