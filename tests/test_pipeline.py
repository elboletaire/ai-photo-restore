import argparse
import json
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
                    repair=False, mask=None, repair_prompt="r", full_frame=False, frame=None, frame_mask=None, frame_seed=None, frame_candidates=1, frame_quality="fast", frame_prompt="f",
                    colorize=True, tone="keep", no_sharpen=False, saturation=0.85, force=False, qwen_model="2511")
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
        def repair(self, path, mask, prompt, seed, megapixels, quality="fast"):
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


def cream_card(h=600, w=400):
    """A cream card with a 40 px border of the grey scanner bed around it."""
    img = np.full((h, w, 3), 225, np.uint8)
    img[40:h - 40, 30:w - 30] = (200, 215, 225)
    return img


def smooth_photo(h, w, colour=(110, 125, 140), seed=0):
    """A photo with mild grain: no edge of it looks like worn rim."""
    grain = np.random.default_rng(seed).normal(0, 2, (h, w, 1))
    return np.clip(np.array(colour, float) + grain, 0, 255).astype(np.uint8)


def vignette_card():
    img = cream_card()
    yy, xx = np.mgrid[0:600, 0:400]
    img[((xx - 200) / 130.0) ** 2 + ((yy - 300) / 230.0) ** 2 <= 1] = (40, 50, 60)
    return img


def test_vignette_is_cut_to_the_oval_and_its_corners_are_filled():
    img = cream_card()
    yy, xx = np.mgrid[0:600, 0:400]
    oval = ((xx - 200) / 130.0) ** 2 + ((yy - 300) / 230.0) ** 2 <= 1
    img[oval] = (40, 50, 60)
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert abs(x0 - 70) <= 4 and abs(x1 - 330) <= 4 and abs(y0 - 70) <= 4 and abs(y1 - 530) <= 4
    assert fill[y0 + 5, x0 + 5] > 0  # the rectangle's corner is outside the oval: to be generated
    assert fill[300, 200] == 0  # the centre of the picture is kept


def test_picture_to_the_edges_keeps_the_card_and_fills_the_missing_corner():
    img = np.full((600, 400, 3), 225, np.uint8)  # scanner bed
    img[20:580, 20:380] = (200, 215, 225)  # card reaching almost to the scan's edges
    img[20:70, 330:380] = 225  # a missing corner, showing the bed
    img[60:260, 60:340] = (120, 110, 100)  # the photo, darker than the mount
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert 20 < x0 and 20 < y0 and x1 < 380 and y1 < 580  # the worn rim is dropped
    assert fill[y0:y1, x0:x1].any()  # the missing corner is inside the frame
    assert fill[160, 200] == 0  # the photo is kept


def test_already_cropped_photo_is_kept_whole():
    img = smooth_photo(600, 400)
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert (x1 - x0) * (y1 - y0) >= 0.95 * 600 * 400
    assert fill.sum() == 0


def test_pale_sky_on_one_edge_is_part_of_the_picture():
    img = smooth_photo(600, 400)
    img[:120] = (228, 230, 232)  # neutral sky touching the top edge only
    (x0, y0, x1, y1), _ = restore.picture_region(img)
    assert (x1 - x0) * (y1 - y0) >= 0.9 * 600 * 400


def test_print_with_a_rectangular_picture_on_an_album_page_is_cropped_to_it_without_fill():
    img = np.full((600, 800, 3), (30, 40, 55), np.uint8)  # dark album page
    img[100:500, 150:650] = smooth_photo(400, 500, (110, 125, 140), seed=1)
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert abs(x0 - 150) <= 3 and abs(x1 - 650) <= 3 and abs(y0 - 100) <= 3 and abs(y1 - 500) <= 3
    assert (fill > 0).mean() < 0.001  # only the picture's rounded corners


def test_frame_mask_is_used_inside_the_frame():
    img = cream_card()
    mask = np.zeros((600, 400), np.uint8)
    mask[130:170, 130:170] = 255
    (x0, y0, x1, y1), fill = restore.picture_region(img, fill=mask)
    assert fill[150, 150] == 255
    assert fill.sum() == 40 * 40 * 255


def test_frame_override_is_used_and_checked():
    img = cream_card()
    box, _ = restore.picture_region(img, rect=(10, 10, 300, 400))
    assert box == (10, 10, 300, 400)
    with pytest.raises(ValueError):
        restore.picture_region(img, rect=(0, 0, 500, 100))


def test_full_frame_after_repair_also_gets_a_colour_version(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, vignette_card())  # has a mount to fill, unlike the random test photo
    b = FakeBackend()
    b.repair = lambda path, *a: b._echo("repair", path)
    bases = []
    b.colorize = lambda path: (bases.append(path.name), b._echo("colorize", path))[1]
    restore.restore_one(src, tmp_path / "out", b, args(generative=False, repair=True, full_frame=True, colorize=False))
    # the damage repair finds nothing to repair, so only the frame calls the model
    assert b.calls == ["conservative", "generative", "conservative", "repair", "conservative"]
    d = tmp_path / "out" / "card"
    assert (d / "29-full-frame-final.jpg").exists() and (d / "28-frame-overlay.jpg").exists()

    b.calls.clear()
    restore.restore_one(src, tmp_path / "colour", b, args(generative=False, full_frame=True, colorize=True))
    assert b.calls == ["conservative", "generative", "repair", "conservative", "colorize", "colorize"]
    assert bases == ["10-conservative.png", "32-full-frame-canvas.png"]  # the frame is coloured in its surroundings
    assert (tmp_path / "colour" / "card" / "32-full-frame-color-final.jpg").exists()


def test_the_frame_is_coloured_inside_its_surroundings_and_cropped_back(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, vignette_card())
    b = FakeBackend()
    b.repair = lambda path, *a: b._echo("repair", path)

    def tint(img):  # the colour depends on the whole picture, so the frame alone gets a different one
        k = int(round(img.mean() - 128))
        return np.clip(img.astype(int) + np.array([0, 0, k]), 0, 255).astype(np.uint8)

    b.colorize = lambda path: cv2.imencode(".png", tint(restore.read_image(path)))[1].tobytes()
    restore.restore_one(src, tmp_path / "out", b, args(generative=False, full_frame=True, colorize=True))
    d = tmp_path / "out" / "card"
    frame = restore.read_image(d / "29-full-frame.png")
    color = restore.read_image(d / "32-full-frame-color.png")
    canvas = restore.read_image(d / "32-full-frame-canvas.png")
    box, _ = restore.picture_region(restore.read_image(src))
    rect = restore.frame_rect(frame.shape, box)
    assert color.shape == frame.shape  # 32 has 29's size
    assert np.array_equal(restore.crop_back(canvas, rect), frame)  # the frame sits unchanged in its surroundings
    assert np.array_equal(color, restore.crop_back(tint(canvas), rect))  # one call on the canvas, cropped back
    assert not np.array_equal(color, tint(frame))  # and not the frame coloured alone


def test_colour_context_never_overflows_at_fractional_scales():
    rng = np.random.default_rng(1)
    for _ in range(200):
        base_h, base_w = (int(v) for v in rng.integers(50, 400, 2))
        x0, y0 = (int(v) for v in rng.integers(0, 30, 2))
        x1, y1 = base_w - int(rng.integers(0, 30)), base_h - int(rng.integers(0, 30))
        box = (x0, y0, x1, y1)
        frame = np.full((round((y1 - y0) * 2.37), round((x1 - x0) * 2.37), 3), 77, np.uint8)
        canvas = np.zeros((base_h, base_w, 3), np.uint8)
        out = restore.colour_context(canvas, frame, (base_h, base_w), box)
        x, y, w, h = restore.frame_rect(frame.shape, box)
        assert np.array_equal(restore.crop_back(out, (x, y, w, h)), frame)


def test_the_canvas_is_the_repaired_picture_when_repairing(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, vignette_card())
    b = FakeBackend()
    b.repair = lambda path, *a: b._echo("repair", path)
    restore.restore_one(src, tmp_path / "out", b, args(generative=False, repair=True, full_frame=True, colorize=True))
    d = tmp_path / "out" / "card"
    frame = restore.read_image(d / "29-full-frame.png")
    canvas = restore.read_image(d / "32-full-frame-canvas.png")
    box, _ = restore.picture_region(restore.read_image(d / "26-repaired-raw.png"))
    assert np.array_equal(restore.crop_back(canvas, restore.frame_rect(frame.shape, box)), frame)
    assert canvas.shape[:2] == restore.read_image(d / "27-repaired.png").shape[:2]


def test_full_frame_rerun_with_same_settings_is_cached(photo, tmp_path):
    b = FakeBackend()
    b.repair = lambda path, *a: b._echo("repair", path)
    b.colorize = lambda path: b._echo("colorize", path)
    restore.restore_one(photo, tmp_path / "out", b, args(full_frame=True, colorize=True))
    b.calls.clear()
    restore.restore_one(photo, tmp_path / "out", b, args(full_frame=True, colorize=True))
    assert b.calls == []


def test_full_frame_changes_only_the_fill(tmp_path):
    img = cream_card()
    yy, xx = np.mgrid[0:600, 0:400]
    img[((xx - 200) / 130.0) ** 2 + ((yy - 300) / 230.0) ** 2 <= 1] = (40, 50, 60)
    src = tmp_path / "card.png"
    restore.write_image(src, img)

    class Garbage(FakeBackend):
        def repair(self, path, mask, prompt, seed, megapixels, quality="fast"):
            out = np.random.default_rng(0).integers(0, 256, restore.read_image(path).shape, dtype=np.uint8)
            return cv2.imencode(".png", out)[1].tobytes()

    box, fill = restore.picture_region(img)
    restore.restore_one(src, tmp_path / "out", Garbage(), args(full_frame=True, colorize=False, generative=False))
    raw = restore.read_image(tmp_path / "out" / "card" / "28-full-frame-raw.png")
    base = restore.read_image(src)[box[1]:box[3], box[0]:box[2]]
    outside = fill[box[1]:box[3], box[0]:box[2]] == 0
    assert np.array_equal(raw[outside], base[outside])
    assert not np.array_equal(raw[~outside], base[~outside])


def test_print_in_the_scanner_corner_is_cut_from_its_bed_on_two_sides():
    img = np.full((1400, 1100, 3), 215, np.uint8)  # scanner bed, visible only on the right and bottom
    img[0:900, 0:700] = smooth_photo(900, 700)
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert (x1 - x0) * (y1 - y0) >= 0.85 * 900 * 700  # the worn rim on the two sides the bed touches is cropped
    assert (x0, y0) == (0, 0)  # the sides that are the image's own edges are not cropped  # no inset along the sides that touch the image edge
    assert fill[300:600, 200:500].sum() == 0  # the picture's interior is kept; only its edge band is regenerated


def test_small_print_on_a_large_bed_is_cut_to_the_print():
    img = np.full((2000, 1500, 3), 215, np.uint8)
    img[300:1200, 300:900] = smooth_photo(900, 600)
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert abs(x0 - 300) <= 120 and abs(y0 - 300) <= 120 and abs(x1 - 900) <= 120 and abs(y1 - 1200) <= 120
    assert fill[600:900, 400:800].sum() == 0  # the picture's interior is kept


def test_cropped_figure_on_a_plain_backdrop_is_kept_whole():
    img = smooth_photo(1000, 800, (120, 140, 160))
    yy, xx = np.mgrid[0:1000, 0:800]
    figure = ((xx - 400) / 150.0) ** 2 + ((yy - 380) / 200.0) ** 2 <= 1
    figure |= (yy > 560) & (np.abs(xx - 400) < 300)  # shoulders and body running to the bottom edge
    img[figure] = smooth_photo(1000, 800, (40, 50, 60), seed=2)[figure]
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert (x1 - x0) * (y1 - y0) >= 0.9 * 1000 * 800
    assert (fill > 0).mean() < 0.01


def test_dark_floor_strip_at_the_bottom_edge_is_kept():
    img = smooth_photo(1500, 1000, (150, 165, 180))
    img[1440:] = smooth_photo(60, 1000, (35, 40, 45), seed=3)  # dark floor, not worn
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert y1 >= 1490  # the strip stays
    assert fill.sum() == 0


def test_light_chip_at_a_corner_is_filled():
    img = np.full((900, 1200, 3), 215, np.uint8)  # scanner bed
    img[30:870, 30:1170] = smooth_photo(840, 1140, (120, 140, 160), seed=6)  # sepia paper, the picture to its edges
    img[30:110, 30:110] = smooth_photo(80, 80, (190, 200, 210), seed=5)  # the corner worn to bare, lighter paper
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    x0, y0 = x0 + 5, y0 + 5
    assert fill[y0, x0] > 0  # the chip, in the corner, comes from the generative variant
    assert fill[450, 600] == 0  # the paper is kept


def test_scanner_bed_is_not_damage_for_a_print_in_the_corner():
    img = np.full((1400, 1100, 3), 215, np.uint8)
    img[0:900, 0:700] = smooth_photo(900, 700)
    generative = img.copy()
    generative[0:900, 0:700] = 0  # the model turned the photo black: the difference is not damage
    bed = restore.scanner_background(img)
    assert not restore.damage_mask(img, generative)[bed > 0].any()
    assert restore.scanner_background(img)[1000:, 800:].all()


def test_cropped_photo_with_no_card_outline_is_kept_whole_and_not_worn():
    img = smooth_photo(1300, 830)
    img[0:150, 300:500] = (235, 235, 235)  # light detail (a forehead, a bright sky) touching the top edge
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert (x0, y0, x1, y1) == (0, 0, 830, 1300)
    assert fill.sum() == 0


def test_changing_the_frame_seed_reruns_only_the_full_frame_stages(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, vignette_card())
    b = FakeBackend()
    b.colorize = lambda path: b._echo("colorize", path)

    def repair(path, mask, prompt, seed, megapixels, quality="fast"):  # the output depends on the seed, as a model's does
        b.calls.append("repair")
        img = restore.read_image(path).astype(int) + seed % 11
        return cv2.imencode(".png", np.clip(img, 0, 255).astype(np.uint8))[1].tobytes()

    b.repair = repair
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=True))
    b.calls.clear()
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=True, frame_seed=7))
    assert b.calls == ["repair", "conservative", "colorize"]  # the fill, its upscale and its colour version


def test_dark_floor_strip_on_a_card_keeps_its_strip_and_regenerates_only_the_edge_band():
    img = np.full((1600, 1100, 3), 215, np.uint8)  # scanner bed
    img[50:1550, 50:1050] = smooth_photo(1500, 1000, (150, 165, 180))
    img[1450:1550, 50:1050] = smooth_photo(100, 1000, (45, 60, 75), seed=3)  # the dark sepia floor, to the card's edge
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    assert y1 >= 1450  # the floor strip is kept, less its worn outer rim
    assert fill[1300, 500] == 0  # the picture above the strip is kept


def seeded_repair(b):
    """A fake repair whose output depends on the seed, as a model's does, recording the seeds asked for."""
    seeds = []

    def repair(path, mask, prompt, seed, megapixels, quality="fast"):
        seeds.append(seed)
        b.calls.append("repair")
        img = restore.read_image(path).astype(int) + seed % 11
        return cv2.imencode(".png", np.clip(img, 0, 255).astype(np.uint8))[1].tobytes()

    b.repair = repair
    return seeds


def test_candidates_use_consecutive_seeds_and_choosing_one_makes_no_new_call(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, vignette_card())
    b = FakeBackend()
    seeds = seeded_repair(b)
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=False, frame_candidates=3))
    assert seeds == [42, 43, 44]
    d = tmp_path / "out" / "card"
    assert (d / "28-full-frame-candidates.jpg").exists()
    b.calls.clear()
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=False, frame_candidates=3, frame_seed=44))
    assert b.calls == ["conservative"]  # only the upscale of the chosen candidate
    assert seeds == [42, 43, 44]
    chosen = restore.read_image(d / "28-full-frame-raw.png")
    assert np.array_equal(chosen, restore.read_image(d / "28-full-frame-raw-s44.png"))


def test_high_quality_drops_the_lightning_lora_and_samples_with_guidance(monkeypatch):
    backend = restore.ComfyBackend("http://comfy")
    sent = {}
    monkeypatch.setattr(backend, "_upload", lambda path: "x.png")
    monkeypatch.setattr(backend, "_run", lambda wf: sent.setdefault("wf", wf) and b"")
    backend.repair(Path("crop.png"), Path("mask.png"), "p", 7, 1.0, quality="high")
    wf = sent["wf"]
    assert not [n for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"]
    sampler = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "Sampler")
    assert sampler["inputs"]["steps"] == restore.HIGH_STEPS and sampler["inputs"]["cfg"] == restore.HIGH_CFG
    aura = next(n for n in wf.values() if n["class_type"] == "ModelSamplingAuraFlow")
    unet = next(k for k, n in wf.items() if n["class_type"] == "UNETLoader")
    assert aura["inputs"]["model"][0] == unet  # the model now comes straight from the UNET


def test_full_frame_figures_compare_with_the_original_cropped_to_the_frame(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, vignette_card())
    b = FakeBackend()
    b.repair = lambda path, *a: b._echo("repair", path)
    b.colorize = lambda path: b._echo("colorize", path)
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=True))
    page = restore.write_sheet(tmp_path / "out").read_text(encoding="utf-8")
    assert 'class="before" src="card/28-original-frame.jpg"' in page  # full frame, and its colour version
    assert page.count('class="before" src="card/28-original-frame.jpg"') == 2
    assert 'class="before" src="card/00-original.png"' in page  # the other variants keep the whole original
    cropped = restore.read_image(tmp_path / "out" / "card" / "28-original-frame.jpg")
    assert cropped.shape[:2] != restore.read_image(src).shape[:2]


def bed_card():
    """A card on a scanner bed that runs to the bed on all four sides, with a dark photo and a light corner."""
    img = np.full((900, 1200, 3), 215, np.uint8)
    img[30:870, 30:1170] = smooth_photo(840, 1140, (120, 140, 160), seed=6)
    img[30:110, 30:110] = smooth_photo(80, 80, (190, 200, 210), seed=5)  # a chip of bare paper in the corner
    return img


def test_edge_to_edge_card_crops_the_worn_rim_and_takes_the_corners_from_the_generative(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, bed_card())
    b = FakeBackend()
    b.generative = lambda *a: cv2.imencode(".png", np.full((900, 1200, 3), 60, np.uint8))[1].tobytes()
    calls_repair = []
    b.repair = lambda *a, **k: calls_repair.append(1) or b._echo("repair", a[0])
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=False, generative=False))
    assert not calls_repair  # no model call for the corners: the generative variant is their source
    d = tmp_path / "out" / "card"
    (x0, y0, x1, y1), fill = restore.picture_region(restore.read_image(src))
    assert x0 >= 30 + round(0.03 * 1200) and y0 >= 30 + round(0.03 * 1200)  # the rim is cropped
    raw = restore.read_image(d / "28-full-frame-raw.png")
    base = restore.read_image(src)[y0:y1, x0:x1]
    outside = fill[y0:y1, x0:x1] == 0
    assert np.array_equal(raw[outside], base[outside])  # pixels outside the fill are the original's
    assert raw[0, 0].mean() < base[0, 0].mean() - 40  # the corner is now the generative's (60 grey)
    assert not (d / "28-full-frame-candidates.jpg").exists()


def test_bare_bed_inside_a_corner_square_takes_the_generative_exactly(tmp_path):
    img = bed_card()
    img[30:200, 1000:1170] = 215  # the top-right corner is cut off by the bed: no paper, no photo there
    src = tmp_path / "cut.png"
    restore.write_image(src, img)
    gen = np.full((900, 1200, 3), 60, np.uint8)
    gen[:, :] = (np.arange(1200)[None, :, None] * 0.2 + np.arange(900)[:, None, None] * 0.1 + 20).astype(np.uint8)
    box, fill, mode = restore.frame_region(img)
    assert mode == "generative"
    x0, y0, x1, y1 = box
    png = restore.generative_frame(img, gen, box, fill, None)
    out = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    bed = (restore.scanner_lid(img) > 0)[y0:y1, x0:x1]
    expected = restore.tone_match(gen[y0:y1, x0:x1], img[y0:y1, x0:x1])
    corner = restore.corner_squares(box, img.shape[:2], max(img.shape[:2]))[y0:y1, x0:x1] > 0
    assert (bed & corner).sum() > 0  # the cut corner really sits inside a corner square
    assert np.array_equal(out[bed], expected[bed])  # every bare bed pixel is the generative's, no blend


def test_a_single_candidate_takes_the_explicit_seed_and_drops_a_stale_sheet(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, vignette_card())
    b = FakeBackend()
    seeds = seeded_repair(b)
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=False, frame_candidates=3))
    assert (tmp_path / "out" / "card" / "28-full-frame-candidates.jpg").exists()
    seeds.clear()
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=False, frame_candidates=1, frame_seed=7))
    assert seeds == [7]  # only the explicit seed is generated
    assert not (tmp_path / "out" / "card" / "28-full-frame-candidates.jpg").exists()  # the stale sheet is gone


def png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def captured_backend(qwen):
    """A ComfyBackend that records the workflow it would have sent, without a server."""
    sent = []
    b = restore.ComfyBackend(url="http://unused", qwen=qwen)
    b._upload = lambda path: "x.png"
    b._run = lambda wf: (sent.append(wf), png(np.full((32, 32, 3), 90, np.uint8)))[1]
    return b, sent


def unet(wf):
    return next(n["inputs"]["unet_name"] for n in wf.values() if n["class_type"] == "UNETLoader")


def test_turbo_sends_the_turbo_graphs_and_the_default_sends_2511(tmp_path):
    img = tmp_path / "p.png"
    restore.write_image(img, np.full((32, 32, 3), 120, np.uint8))
    mask = tmp_path / "m.png"
    restore.write_image(mask, np.zeros((32, 32, 3), np.uint8))
    b, sent = captured_backend("2511")
    b.generative(img, "p", 1, 1.0)
    b.repair(img, mask, "p", 1, 1.0)
    assert [unet(w) for w in sent] == ["qwen_image_edit_2511_fp8mixed.safetensors"] * 2
    b, sent = captured_backend("turbo")
    b.generative(img, "p", 1, 1.0)
    b.repair(img, mask, "p", 1, 1.0, quality="high")  # no LoRA and no 20-step override on turbo
    assert [unet(w) for w in sent] == ["qwen_image_2.1_turbo_int8_convrot.safetensors"] * 2
    assert all(n["class_type"] != "LoraLoaderModelOnly" for n in sent[1].values())
    sampler = sent[1][restore.find_node(sent[1], "Sampler")]
    assert sampler["inputs"]["steps"] == 8


def test_turbo_workflows_have_the_titles_the_backend_sets():
    for name, titles in [("generative-turbo", ["Input image", "Prompt", "Sampler", "Scale", "Output"]),
                         ("repair-turbo", ["Input image", "Prompt", "Sampler", "Scale", "Output", "Mask"])]:
        wf = restore.load_workflow(name)
        for title in titles:
            restore.find_node(wf, title)  # raises when the title is missing


def test_the_two_qwen_models_never_share_a_cached_stage(photo, tmp_path):
    b = FakeBackend()
    out = tmp_path / "out"
    restore.restore_one(photo, out, b, args(generative=True, colorize=False, qwen_model="2511"))
    keys = json.loads((out / photo.stem / "stages.json").read_text(encoding="utf-8"))
    assert "qwen" not in keys["20-generative.png"]  # the default keeps its existing cache keys
    restore.restore_one(photo, out, b, args(generative=True, colorize=False, qwen_model="turbo"))
    assert b.calls.count("generative") == 2  # the turbo run did not reuse the 2511 file
    keys = json.loads((out / photo.stem / "stages.json").read_text(encoding="utf-8"))
    assert keys["20-generative.png"]["qwen"] == "turbo"
    restore.restore_one(photo, out, b, args(generative=True, colorize=False, qwen_model="turbo"))
    assert b.calls.count("generative") == 2  # the same model again is cached


def test_check_asks_for_the_turbo_models_when_turbo_is_chosen(monkeypatch):
    names = {"generative-turbo": "qwen_image_2.1_turbo_int8_convrot.safetensors"}
    classes = set()
    for wf in ("generative-turbo", "repair-turbo"):
        classes |= {n["class_type"] for n in restore.load_workflow(wf).values()}

    def info(installed):
        obj = {cls: {"input": {"required": {}}} for cls in classes}
        obj["UNETLoader"] = {"input": {"required": {"unet_name": [sorted(installed)]}}}
        obj["CLIPLoader"] = {"input": {"required": {"clip_name": [["qwen3vl_8b_w4a8.safetensors"]]}}}
        obj["VAELoader"] = {"input": {"required": {"vae_name": [["qwen_image_2.1_vae_bf16.safetensors"]]}}}

        class Resp:
            def json(self):
                return obj
        return lambda url, timeout=None: Resp()

    monkeypatch.setattr(restore.requests, "get", info(["something_else.safetensors"]))
    problems = restore.ComfyBackend(qwen="turbo").check(["generative", "repair"])
    assert any(names["generative-turbo"] in p for p in problems)
    monkeypatch.setattr(restore.requests, "get", info([names["generative-turbo"]]))
    assert restore.ComfyBackend(qwen="turbo").check(["generative", "repair"]) == []


def test_fal_refuses_the_turbo_model(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "k")
    problems = restore.FalBackend(qwen="turbo").check(["generative"])
    assert any("local only" in p for p in problems)
    assert restore.FalBackend(qwen="2511").check(["generative"]) == []



def test_the_default_model_still_inpaints_the_frame_in_place_with_the_same_call_and_key(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, vignette_card())
    b = FakeBackend()
    seen = []

    def repair(path, mask, prompt, seed, megapixels, quality="fast"):
        img = restore.read_image(path)
        seen.append((prompt, seed, megapixels, quality, restore.read_mask(mask, img.shape[:2]).any()))
        return b._echo("repair", path)

    b.repair = repair
    b.generative = lambda path, prompt, *a: seen.append(("generative", prompt)) or b._echo("generative", path)
    restore.restore_one(src, tmp_path / "out", b, args(generative=False, full_frame=True, colorize=False))
    assert b.calls == ["conservative", "generative", "repair", "conservative"]
    assert seen == [("generative", "p"), ("f", 42, 1.0, "fast", True)]  # the fill is a masked repair, not an edit
    keys = json.loads((tmp_path / "out" / "card" / "stages.json").read_text(encoding="utf-8"))
    key = keys["28-full-frame-raw-s42.png"]
    box, _ = restore.picture_region(restore.read_image(src))
    assert {k: v for k, v in key.items() if k != "inputs"} == {"backend": "fake", "prompt": "f", "seed": 42,
                                                                "megapixels": 1.0, "rect": list(box), "quality": "fast"}
    assert len(key["inputs"]) == 2  # the base and the frame mask
    # and the graph it sends is still the 2511 repair graph, untouched
    cb, sent = captured_backend("2511")
    cb.repair(src, src, "f", 42, 1.0)
    assert [n["class_type"] for n in sent[0].values()] == [n["class_type"] for n in restore.load_workflow("repair").values()]
    assert unet(sent[0]) == "qwen_image_edit_2511_fp8mixed.safetensors"


def test_turbo_edits_a_prefilled_rectangle_and_takes_only_the_fill_from_it(tmp_path):
    img = vignette_card()
    src = tmp_path / "card.png"
    restore.write_image(src, img)
    b = FakeBackend()
    seen = []

    def generative(path, prompt, seed, megapixels):
        b.calls.append("generative")
        seen.append((prompt, seed, restore.read_image(path)))
        out = restore.read_image(path).astype(int) + 12  # the model renders everything a bit lighter
        return png(np.clip(out, 0, 255).astype(np.uint8))

    b.generative = generative
    b.repair = lambda *a, **k: pytest.fail("turbo must not inpaint the frame with its reference in view")
    restore.restore_one(src, tmp_path / "out", b, args(generative=False, full_frame=True, colorize=False, qwen_model="turbo"))
    assert b.calls == ["conservative", "generative", "generative", "conservative"]
    (x0, y0, x1, y1), fill = restore.picture_region(img)
    crop, m = img[y0:y1, x0:x1], fill[y0:y1, x0:x1] > 0
    prompt, seed, pre = seen[1]
    assert (prompt, seed) == ("f", 42) and pre.shape == crop.shape  # the frame prompt, on the rectangle alone
    mount = np.abs(pre.astype(int) - [200, 215, 225]).sum(axis=2) < 15
    assert not (mount & m).any()  # the pre-filled copy shows the model no cream mount in the fill
    assert np.array_equal(pre[~m], crop[~m])  # and the picture itself
    raw = restore.read_image(tmp_path / "out" / "card" / "28-full-frame-raw.png")
    assert np.array_equal(raw[~m], crop[~m])  # the kept picture is bit for bit the original's
    assert not np.array_equal(raw[m], crop[m])
    # the edit's lightness is matched to the kept picture: the fill continues it instead of stepping up by 12
    L = restore.to_lab(raw)[..., 0]
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    deep = (dist > 0.01 * 600 + 2) & (dist < 0.02 * 600)  # past the ramp, near the boundary
    assert abs(np.median(L[deep]) - np.median(restore.to_lab(crop)[..., 0][~m])) < 1.0
    keys = json.loads((tmp_path / "out" / "card" / "stages.json").read_text(encoding="utf-8"))
    assert keys["28-full-frame-raw-s42.png"]["qwen"] == "turbo"
    assert keys["28-full-frame-raw-s42.png"]["fill"] == "prefilled-edit"  # an earlier turbo inpaint is never reused


def test_turbo_leaves_a_card_on_a_bed_to_the_generative_corners(tmp_path):
    src = tmp_path / "card.png"
    restore.write_image(src, bed_card())
    b = FakeBackend()
    b.generative = lambda *a: png(np.full((900, 1200, 3), 60, np.uint8))
    b.repair = lambda *a, **k: pytest.fail("no model fill for a card on a bed")
    restore.restore_one(src, tmp_path / "out", b, args(full_frame=True, colorize=False, generative=False, qwen_model="turbo"))
    assert b.calls == ["conservative", "conservative"]  # the generative variant and the upscale; no fill call
    keys = json.loads((tmp_path / "out" / "card" / "stages.json").read_text(encoding="utf-8"))
    assert keys["28-full-frame-raw.png"]["mode"] == "generative-corners"


def test_match_luminance_removes_the_step_along_the_fill_and_keeps_the_gradient_beyond_it():
    base = np.full((200, 200, 3), 90, np.uint8)
    fill = np.zeros((200, 200), np.uint8)
    fill[:, 120:] = 255
    gen = np.clip(base.astype(int) + 20, 0, 255).astype(np.uint8)  # rendered lighter everywhere
    out = restore.match_luminance(gen, base, fill, band=10)
    L = restore.to_lab(out)[..., 0]
    assert abs(L[:, 125:140].mean() - restore.to_lab(base)[..., 0].mean()) < 0.5  # continues the base at the boundary
    assert abs(L[:, 190:].mean() - L[:, 125:140].mean()) < 0.5  # a flat shift stays flat deep in the fill
