# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "requests"]
# ///
"""Old photo restoration pipeline. Run with uv (works on Linux, macOS and Windows):

    uv run restore.py run <image|dir>... [--backend local|fal] [--generative] [--colorize] ...
    uv run restore.py doctor [--backend local|fal]
    uv run restore.py setup-local --comfy <ComfyUI dir> --python <ComfyUI's python>
    uv run restore.py sheet [out]

Backends:
  local  ComfyUI server (COMFY_URL, default http://127.0.0.1:8188) running the workflows/*.json graphs.
  fal    fal.ai hosted models (FAL_KEY). Same model families: CodeFormer, Qwen-Image-Edit-2511, DDColor.

Each photo gets <out>/<name>/ with numbered stages; finished files end in -final.jpg.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path

import cv2
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
WORKFLOWS = HERE.parent / "workflows"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

REPAIR_PROMPT = (
    "Repair the damaged areas of this old photograph: fill in the missing, torn and peeled patches so they "
    "continue the surrounding picture seamlessly. Keep it a black and white photograph."
)
HIGH_STEPS, HIGH_CFG = 20, 4.0  # --frame-quality high: the sampler settings without the Lightning LoRA
FRAME_PROMPT = (
    "Extend this photograph to fill the whole rectangular frame: continue the background, clothing and body "
    "naturally into the empty areas. No oval, mount, border or vignette. Keep it a black and white photograph."
)
RESTORE_PROMPT = (
    "Restore this old photograph. Remove scratches, dust, stains, creases and paper damage. "
    "Recover sharp natural detail and clean tonal range. Keep the exact same people, faces, "
    "expressions, pose, clothing, framing and composition. Keep it a black and white photograph."
)


# ---------------------------------------------------------------- image helpers

def read_image(path: Path) -> np.ndarray:
    # imdecode instead of imread: handles non-ASCII paths on Windows.
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """A black and white mask (white = damaged) at `shape` (h, w), as 0/255."""
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"cannot read mask {path}")
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def write_image(path: Path, img: np.ndarray) -> None:
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(path.suffix, img, params)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(str(path))


def to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def from_lab(lab: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)
    return np.clip(bgr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def tone_match(restored: np.ndarray, original: np.ndarray, bins: int = 64) -> np.ndarray:
    """Keep the restored luminance, rebuild chroma from the original print's tint.

    AI restorers output RGB and invent colour (blue eyes, pink skin) on B/W or sepia prints. For each
    lightness level we take the median a/b of the original, which reproduces the toning (brown
    shadows, cream highlights) without bringing back colour stains from damaged areas.
    """
    src = to_lab(original)
    l, a, b = (src[..., i].ravel() for i in range(3))
    idx = np.clip((l / 100.0 * bins).astype(int), 0, bins - 1)
    centers, ca, cb = [], [], []
    for i in range(bins):
        mask = idx == i
        if mask.sum() >= 50:
            centers.append((i + 0.5) * 100.0 / bins)
            ca.append(np.median(a[mask]))
            cb.append(np.median(b[mask]))
    out = to_lab(restored)
    if centers:
        k = np.ones(5) / 5  # smooth so a few stained pixels in one bin can't create a colour band
        smooth = lambda v: np.convolve(np.pad(np.array(v), 2, mode="edge"), k, mode="valid")
        out[..., 1] = np.interp(out[..., 0], centers, smooth(ca))
        out[..., 2] = np.interp(out[..., 0], centers, smooth(cb))
    else:
        out[..., 1:] = 0
    return from_lab(out)


def finish(img: np.ndarray, tone: str, sharpen: bool = True, saturation: float = 1.0) -> np.ndarray:
    """Tonal finishing: contrast stretch on L, optional re-toning, light unsharp mask.

    The colorized variant only gets its saturation adjusted: DDColor's output already has the
    restored tonal range, and stretching it crushed dark clothes into black.
    """
    lab = to_lab(img)
    if tone == "color":
        lab[..., 1:] *= saturation
        return from_lab(lab)
    L = lab[..., 0]
    lo, hi = np.percentile(L, [0.3, 99.7])
    if hi - lo > 1:
        L = np.clip((L - lo) / (hi - lo) * 100.0, 0, 100)
    if sharpen:
        blur = cv2.GaussianBlur(L, (0, 0), 1.2)
        detail = L - blur
        L = np.clip(L + 0.6 * np.where(np.abs(detail) > 0.8, detail, 0), 0, 100)
    lab[..., 0] = L
    shade = 1.0 - L / 100.0
    if tone == "neutral":
        lab[..., 1:] = 0
    elif tone == "sepia":
        lab[..., 1] = 2.0 + 6.0 * shade
        lab[..., 2] = 8.0 + 14.0 * shade
    return from_lab(lab)


def align_to(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Resize `img` to `ref`'s size and fine-align it (affine): generative output is rescaled and slightly shifted."""
    h, w = ref.shape[:2]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    s = min(1.0, 1000 / max(h, w))
    small = [cv2.resize(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s).astype(np.float32) for i in (ref, img)]
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(*small, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    except cv2.error:
        return img  # didn't converge: keep the plain resize
    warp[:, 2] /= s
    return cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def connected_to_border(mask: np.ndarray) -> np.ndarray:
    """The pixels of a binary mask that touch the image border through other mask pixels."""
    _, labels = cv2.connectedComponents(mask.astype(np.uint8))
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return np.isin(labels, border[border > 0])


def scanner_bed(img: np.ndarray) -> np.ndarray:
    """Mask of the grey scanner bed around the photo: neutral pixels connected to the image border.

    Old prints and their mounts are toned (sepia, cream), the scanner lid is not.
    """
    lab = to_lab(cv2.GaussianBlur(img, (0, 0), 2))
    return connected_to_border(np.hypot(lab[..., 1], lab[..., 2]) < 6).astype(np.uint8)


def scanner_background(img: np.ndarray) -> np.ndarray:
    """scanner_bed plus the soft shadow along the card's edge."""
    bg = scanner_bed(img)
    r = max(3, round(max(img.shape[:2]) / 200))
    return cv2.dilate(bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def solidity(mask: np.ndarray) -> float:
    """How much of its convex hull a shape fills: 1 for an oval or a rectangle, less for a ragged or concave one."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    hull = cv2.convexHull(np.vstack(contours))
    return float(mask.sum()) / max(1.0, cv2.contourArea(hull))


def scanner_lid(img: np.ndarray) -> np.ndarray:
    """The scanner bed as far as a print is concerned: it runs along two or more sides of the scan, and what
    it leaves is a solid card. A pale sky or wall along one edge of a photo is not a lid."""
    bed = scanner_bed(img)
    sides = [bed[0], bed[-1], bed[:, 0], bed[:, -1]]
    if not bed.any() or sum(s.mean() >= 0.5 for s in sides) < 2 or solidity(bed == 0) < 0.9:
        return np.zeros(img.shape[:2], np.uint8)
    return bed


PICTURE_DISTANCE = 12  # Lab distance from the mount colour that separates the picture from the mount
SOLID_PICTURE = 0.95  # a picture this solid (an oval, a rectangle) is a vignette or a mount's picture
FRAME_REACH = 0.06  # share of the long side the frame may come in from the card's edge
EDGE_BAND = 0.03  # share of the long side, along each edge of a card on a scanner bed, that is regenerated
CORNER_SQUARE = 0.05  # share of the long side, the square in each corner taken from the generative variant
MIN_PICTURE = 0.1  # the share of the card a picture in a vignette or on a mount covers, at least
VIGNETTE_COVER = 0.8  # below this share of the card, the picture sits in a vignette or a mount
VIGNETTE_EDGE = 0.015  # share of the long side, the outermost sliver of a vignette's frame, regenerated too
VIGNETTE_GROW = 0.025  # share of the long side the fill grows into a curved vignette, its dark shadow ring
FILLED_BOX = 0.97  # a picture this full of its bounding box is a rectangle: nothing to fill


def reach_depth(lines: np.ndarray) -> int:
    """How far flagged pixels reach from the edge: `lines` (one row or column per line, flags from the edge).

    The median line, not the farthest: a missing or rounded corner is filled in, it does not move the whole edge.
    """
    if lines.size == 0:
        return 0
    extent = np.where(lines.any(axis=1), lines.shape[1] - np.argmax(lines[:, ::-1], axis=1), 0)
    return int(np.percentile(extent, 50))


def corner_squares(box: tuple, shape: tuple[int, int], long: int) -> np.ndarray:
    """A square in each corner of the rectangle `box` (0/1), the size of the rounded and chipped corners."""
    x0, y0, x1, y1 = box
    corner = round(CORNER_SQUARE * long)
    squares = np.zeros(shape, np.uint8)
    for cx, cy in [(x0, y0), (x1 - corner, y0), (x0, y1 - corner), (x1 - corner, y1 - corner)]:
        squares[max(0, cy):cy + corner, max(0, cx):cx + corner] = 1
    return squares


def generative_fill(base: np.ndarray, gen: np.ndarray, corners: np.ndarray, bare: np.ndarray, corner: int) -> np.ndarray:
    """Fill the corners and the missing paper from the generative variant, aligned and tone-matched to the print.

    The corners ramp in from the generative (0 at their inner edge, 1 deep in the corner); bare paper takes it fully.
    Outside the fill the base is kept bit for bit. The model is not called: the generative variant is already there.
    """
    g = tone_match(gen, base)
    dist = cv2.distanceTransform(corners.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.maximum(np.clip(dist / (0.6 * corner), 0, 1), bare.astype(np.float32))
    alpha = (alpha * ((corners | bare) > 0))[..., None]
    return (base * (1 - alpha) + g * alpha + 0.5).astype(np.uint8)


def picture_region(img: np.ndarray, rect: tuple[int, int, int, int] | None = None, fill: np.ndarray | None = None):
    """The rectangle a --full-frame photo should fill, and the pixels inside it to generate (0/255)."""
    box, fill, _ = frame_region(img, rect, fill)
    return box, fill


def frame_region(img: np.ndarray, rect: tuple[int, int, int, int] | None = None, fill: np.ndarray | None = None):
    """Like picture_region, and how the fill is made: "generative" (the corners come from the generative variant,
    for a card on a scanner bed that runs to the edges) or "inpaint" (the model fills it).

    A picture in an oval vignette, or on a mount, keeps only its own pixels: the vignette and the mount are filled.
    A picture running to the card's edges keeps the card minus its bare paper, and the rectangle comes in past the
    ragged outline of the card (its rounded or missing corners). A rectangular picture on a mount is cropped.
    `rect` and `fill` (a mask) override.
    """
    h, w = img.shape[:2]
    long = max(h, w)
    lid = scanner_lid(img)
    card = lid == 0
    outlined = lid.any()  # a card on a scanner bed has a worn outline; a cropped photo has none
    if not card.any():
        card[:] = True
    ys, xs = np.nonzero(card)
    cx0, cy0, cx1, cy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    lab = to_lab(cv2.GaussianBlur(img, (0, 0), 2))

    # The mount colour: the median of a band just inside the card.
    band = max(3, round(0.03 * long))
    edge = np.zeros((h, w), bool)
    edge[cy0:cy1, cx0:cx1] = True
    edge[cy0 + band:cy1 - band, cx0 + band:cx1 - band] = False
    d = np.linalg.norm(lab - np.median(lab[edge & card], axis=0), axis=2)
    mounted = (d[edge & card] <= PICTURE_DISTANCE).mean() >= 0.9  # a uniform band: a mount, not the picture
    pic = ((d > PICTURE_DISTANCE) & card).astype(np.uint8)
    pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(pic)
    pic = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) if n > 1 else card.copy()
    pic = pic | ~connected_to_border(~pic)  # holes: a face, a light patch of the picture

    picture_share = pic.sum() / card.sum()  # a blemish or a chip is not a picture on a mount
    vignette = mounted and MIN_PICTURE <= picture_share < VIGNETTE_COVER and solidity(pic) >= SOLID_PICTURE
    if vignette:
        keep = pic
        ys, xs = np.nonzero(pic)
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        curved = pic.sum() < FILLED_BOX * (box[2] - box[0]) * (box[3] - box[1])
    else:
        # The outline: where the bed bites into the card's bounding box, the rectangle comes in past it.
        reach = max(3, round(FRAME_REACH * long))
        bed = ~card
        strips = [bed[cy0:cy1, cx0:cx0 + reach], bed[cy0:cy1, cx1 - reach:cx1][:, ::-1],
                  bed[cy0:cy0 + reach, cx0:cx1].T, bed[cy1 - reach:cy1, cx0:cx1][::-1].T]
        # Only a side the scanner bed touches has a worn rim to crop: the other sides are the image's own edges.
        on_bed = [outlined and strip.any() for strip in strips]
        margin = round(0.005 * long)
        crop = round(EDGE_BAND * long)  # the worn rim is cropped, not generated
        il, ir, it, ib = (min(reach, reach_depth(strip) + margin) + crop if b else 0
                          for strip, b in zip(strips, on_bed))
        keep = card
        box = (cx0 + il, cy0 + it, cx1 - ir, cy1 - ib)
        curved = False
        if box[2] <= box[0] or box[3] <= box[1]:
            box = (cx0, cy0, cx1, cy1)
    if rect is not None:  # --frame
        x0, y0, x1, y1 = rect
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"--frame {x0},{y0},{x1},{y1} is outside the {w}x{h} photo")
        box = rect
    x0, y0, x1, y1 = (int(v) for v in box)
    inside = np.zeros((h, w), np.uint8)
    inside[y0:y1, x0:x1] = 1
    if fill is not None:  # --frame-mask: the user's own mask
        return (x0, y0, x1, y1), ((fill > 0).astype(np.uint8) * inside) * 255, "inpaint"
    if vignette and curved:  # the vignette's soft edge, its dark shadow ring included, is grown into
        grow = round(long * VIGNETTE_GROW)
    elif vignette:
        grow = 0
    else:
        grow = 2  # the antialiased edge of the card
    outside = (~keep).astype(np.uint8)
    if grow:
        outside = cv2.dilate(outside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    if outlined and not vignette:  # a card on a scanner bed: its corners come from the generative variant
        corners = corner_squares((x0, y0, x1, y1), (h, w), long)
        return (x0, y0, x1, y1), (np.maximum(outside, corners) & inside) * 255, "generative"
    if vignette and curved:  # the outermost sliver of an oval's rectangle too: the model would continue the mount's edge
        r = round(VIGNETTE_EDGE * long)
        sliver = inside.copy()
        sliver[y0 + r:max(y0 + r, y1 - r), x0 + r:max(x0 + r, x1 - r)] = 0
        outside = np.maximum(outside, sliver)
    return (x0, y0, x1, y1), (outside & inside) * 255, "inpaint"


def damage_mask(original: np.ndarray, generative: np.ndarray, threshold: float = 15) -> np.ndarray:
    """Where the original is damaged (0/255), found by comparing it with the generative restoration.

    Holes, peeled emulsion and stains show the light paper underneath, so they are where the original is
    much lighter than the generative version once both have the same overall brightness. Requiring
    "lighter" (not just "different") leaves out what the model merely redrew: faces, lace, highlights.
    """
    h, w = original.shape[:2]
    L0 = to_lab(original)[..., 0]
    L1 = to_lab(align_to(generative, original))[..., 0]
    (a25, a50, a75), (b25, b50, b75) = np.percentile(L0, [25, 50, 75]), np.percentile(L1, [25, 50, 75])
    L1 = (L1 - b50) * ((a75 - a25) / (b75 - b25) if b75 - b25 > 1 else 1.0) + a50  # same median and spread
    sigma = max(h, w) / 400
    lighter = cv2.GaussianBlur(L0, (0, 0), sigma) - cv2.GaussianBlur(L1, (0, 0), sigma) > threshold
    m = cv2.morphologyEx(lighter.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= h * w * 0.0002)
    m = np.isin(labels, big[big > 0]).astype(np.uint8)
    m[scanner_background(original) > 0] = 0
    r = max(2, round(max(h, w) / 300))  # cover the ragged edges of the holes
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) * 255


def damage_overlay(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = original.copy()
    hit = mask > 0
    out[hit] = (0.5 * out[hit] + [0, 0, 127]).astype(np.uint8)
    return out


def mask_box(mask: np.ndarray, pad: float = 0.1) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the mask, padded so the model sees the surroundings."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    p = round(pad * max(h, w))
    return max(0, xs.min() - p), max(0, ys.min() - p), min(w, xs.max() + 1 + p), min(h, ys.max() + 1 + p)


def paste(original: np.ndarray, patch: np.ndarray, mask: np.ndarray, outward: bool = True) -> np.ndarray:
    """Original everywhere except under `mask`, where `patch` (same size) goes, with feathered edges.

    Damage is repaired outwards: the feather reaches past the mask so no damage shows. A fill is feathered
    inwards instead, so everything outside the mask stays bit-identical to the original.
    """
    r = max(1.0, max(mask.shape) / 600)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), r)
    alpha = np.maximum(alpha, mask > 0) if outward else np.minimum(alpha, mask > 0)
    alpha = alpha[..., None]
    return (original * (1 - alpha) + patch * alpha + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- backends

QWEN_MODELS = ("2511", "turbo")  # --qwen-model: Qwen-Image-Edit-2511 (default) or the experimental 2.1 Turbo
QWEN_NAMES = {"2511": "Qwen-Image-Edit-2511", "turbo": "Qwen-Image-2.1-Turbo"}


def qwen_workflow(name: str, qwen: str) -> str:
    """The workflow of a Qwen stage: the Turbo graph for --qwen-model turbo, else the 2511 one."""
    return f"{name}-turbo" if qwen == "turbo" and name in ("generative", "repair") else name


class ComfyBackend:
    name = "local"

    def __init__(self, url: str | None = None, qwen: str = "2511"):
        self.url = (url or os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.qwen = qwen

    def check(self, stages: list[str]) -> list[str]:
        """Problems that would stop the workflows of `stages` from running: missing nodes or models."""
        try:
            info = requests.get(f"{self.url}/object_info", timeout=10).json()
        except requests.RequestException:
            return [f"ComfyUI not reachable at {self.url} (start it, or set COMFY_URL)"]
        problems = []
        for stage_name in stages:
            wf = qwen_workflow(stage_name, self.qwen)
            for node in load_workflow(wf).values():
                cls = node["class_type"]
                if cls not in info:
                    problems.append(f"{wf}: missing ComfyUI node {cls} (run setup-local, then restart ComfyUI)")
                    continue
                inputs = info[cls].get("input", {})
                spec = {**inputs.get("optional", {}), **inputs.get("required", {})}
                for key, value in node["inputs"].items():
                    options = combo_options(spec.get(key))
                    if options is not None and isinstance(value, str) and value not in options and cls not in ("LoadImage", "LoadImageMask"):
                        problems.append(f"{wf}: {cls}.{key} = {value} not installed (run setup-local)")
        return problems

    def _upload(self, path: Path) -> str:
        # Named after the content: every stage file is called 00-original.jpg, 10-conservative.png...,
        # so a fixed name would let two runs at once overwrite each other's input.
        name = f"{digest(path)}{path.suffix.lower()}"
        with open(path, "rb") as f:
            r = requests.post(f"{self.url}/upload/image", files={"image": (name, f)},
                              data={"overwrite": "true", "subfolder": "photo-restore"}, timeout=120)
        r.raise_for_status()
        d = r.json()
        return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]

    def _run(self, wf: dict) -> bytes:
        r = requests.post(f"{self.url}/prompt", json={"prompt": wf, "client_id": self.client_id}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:500]}")
        pid = r.json()["prompt_id"]
        deadline = time.time() + 1800
        while time.time() < deadline:
            time.sleep(1)
            entry = requests.get(f"{self.url}/history/{pid}", timeout=30).json().get(pid)
            if not entry:
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                err = next((m[1] for m in status.get("messages", []) if m[0] == "execution_error"), {})
                raise RuntimeError(f"ComfyUI error in {err.get('node_type')}: {err.get('exception_message')}")
            if status.get("completed"):
                out_id = find_node(wf, "Output")
                img = entry["outputs"][out_id]["images"][0]
                v = requests.get(f"{self.url}/view", params=img, timeout=120)
                v.raise_for_status()
                return v.content
        raise TimeoutError("ComfyUI did not finish in 30 minutes")

    def conservative(self, path: Path, fidelity: float, upscale: bool) -> bytes:
        wf = load_workflow("conservative")
        set_input(wf, "Input image", "image", self._upload(path))
        set_input(wf, "Face restore", "codeformer_fidelity", fidelity)
        if not upscale:
            bypass(wf, "Upscale", "image")
        return self._run(wf)

    def generative(self, path: Path, prompt: str, seed: int, megapixels: float) -> bytes:
        wf = load_workflow(qwen_workflow("generative", self.qwen))
        set_input(wf, "Input image", "image", self._upload(path))
        set_input(wf, "Prompt", "prompt", prompt)
        set_input(wf, "Sampler", "seed", seed)
        set_input(wf, "Scale", "megapixels", megapixels)
        return self._run(wf)

    def repair(self, path: Path, mask: Path, prompt: str, seed: int, megapixels: float, quality: str = "fast") -> bytes:
        wf = load_workflow(qwen_workflow("repair", self.qwen))
        if quality == "high" and self.qwen == "2511":  # no Lightning LoRA: real guidance, more steps (Turbo has its own)
            bypass_class(wf, "LoraLoaderModelOnly", "model")
            set_input(wf, "Sampler", "steps", HIGH_STEPS)
            set_input(wf, "Sampler", "cfg", HIGH_CFG)
        set_input(wf, "Input image", "image", self._upload(path))
        set_input(wf, "Mask", "image", self._upload(mask))
        set_input(wf, "Prompt", "prompt", prompt)
        set_input(wf, "Sampler", "seed", seed)
        set_input(wf, "Scale", "megapixels", megapixels)
        return self._run(wf)

    def colorize(self, path: Path) -> bytes:
        wf = load_workflow("colorize")
        set_input(wf, "Input image", "image", self._upload(path))
        return self._run(wf)


class FalBackend:
    name = "fal"

    def __init__(self, qwen: str = "2511"):
        self.key = os.environ.get("FAL_KEY", "")
        self.qwen = qwen

    def check(self, stages: list[str]) -> list[str]:
        problems = [] if self.key else ["FAL_KEY is not set (create one at https://fal.ai/dashboard/keys)"]
        if "repair" in stages:
            problems.append("--repair and --full-frame only work with the local backend for now")
        if self.qwen == "turbo":
            problems.append("--qwen-model turbo is local only (fal runs the 2511 models)")
        return problems

    def _call(self, endpoint: str, payload: dict) -> bytes:
        r = requests.post(f"https://fal.run/{endpoint}", json=payload, timeout=600,
                          headers={"Authorization": f"Key {self.key}"})
        if r.status_code != 200:
            raise RuntimeError(f"fal {endpoint}: HTTP {r.status_code} {r.text[:500]}")
        data = r.json()
        image = data.get("image") or (data.get("images") or [None])[0]
        if not image:
            raise RuntimeError(f"fal {endpoint}: no image in response")
        url = image["url"]
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
        return requests.get(url, timeout=120).content

    @staticmethod
    def _data_uri(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    def conservative(self, path: Path, fidelity: float, upscale: bool) -> bytes:
        return self._call("fal-ai/codeformer", {"image_url": self._data_uri(path), "fidelity": fidelity,
                                                "upscale_factor": 2 if upscale else 1})

    def generative(self, path: Path, prompt: str, seed: int, megapixels: float) -> bytes:
        return self._call("fal-ai/qwen-image-edit-2511", {"image_urls": [self._data_uri(path)], "prompt": prompt,
                                                          "seed": seed, "output_format": "png"})

    def repair(self, path: Path, mask: Path, prompt: str, seed: int, megapixels: float, quality: str = "fast") -> bytes:
        raise RuntimeError("--repair and --full-frame only work with the local backend for now")

    def colorize(self, path: Path) -> bytes:
        return self._call("fal-ai/ddcolor", {"image_url": self._data_uri(path)})


def combo_options(spec) -> list | None:
    """The choices of a ComfyUI dropdown input, in the old ([...],) or newer ("COMBO", {"options": [...]}) form."""
    if not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def make_backend(name: str, qwen: str = "2511"):
    return ComfyBackend(qwen=qwen) if name == "local" else FalBackend(qwen=qwen)


# ---------------------------------------------------------------- workflow helpers

def load_workflow(name: str) -> dict:
    return json.loads((WORKFLOWS / f"{name}.json").read_text(encoding="utf-8"))


def find_node(wf: dict, title: str) -> str:
    for node_id, node in wf.items():
        if node.get("_meta", {}).get("title") == title:
            return node_id
    raise KeyError(f"workflow has no node titled {title!r}")


def set_input(wf: dict, title: str, key: str, value) -> None:
    wf[find_node(wf, title)]["inputs"][key] = value


def bypass(wf: dict, title: str, passthrough: str) -> None:
    """Remove a node, rewiring its consumers to the node's `passthrough` input."""
    bypass_node(wf, find_node(wf, title), passthrough)


def bypass_class(wf: dict, class_type: str, passthrough: str) -> None:
    """Like bypass, for the node of this class (for nodes without a title)."""
    bypass_node(wf, next(k for k, v in wf.items() if v["class_type"] == class_type), passthrough)


def bypass_node(wf: dict, node_id: str, passthrough: str) -> None:
    source = wf.pop(node_id)["inputs"][passthrough]
    for node in wf.values():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and v and v[0] == node_id:
                node["inputs"][k] = source


# ---------------------------------------------------------------- pipeline

def collect_inputs(args: list[str]) -> list[Path]:
    files: list[Path] = []
    for a in args:
        p = Path(a).expanduser().resolve()
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(a)
    return files


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def qwen_key(a) -> dict:
    """The Qwen model is part of a stage's key when it is not the default: a turbo file never reuses a 2511 one."""
    qwen = getattr(a, "qwen_model", "2511")
    return {} if qwen == "2511" else {"qwen": qwen}


def stage(target: Path, label: str, inputs: list[Path], settings: dict, force: bool, produce) -> None:
    """Run an AI stage unless `target` was already made from the same input files and settings.

    What each stage was made from is recorded in stages.json next to it, so changing --seed,
    --fidelity, the backend, or recomputing an upstream stage reruns it automatically.
    """
    record = target.parent / "stages.json"
    done = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    key = json.loads(json.dumps({**settings, "inputs": [digest(p) for p in inputs]}))  # as stages.json stores it
    if target.exists() and not force and done.get(target.name) == key:
        print(f"  = {label} (cached)")
        return
    t0 = time.time()
    target.write_bytes(produce())
    done[target.name] = key
    record.write_text(json.dumps(done, indent=2), encoding="utf-8")
    print(f"  ✓ {label} ({time.time() - t0:.1f}s)", flush=True)


def inpaint(backend, img: np.ndarray, mask: np.ndarray, prompt: str, a, box=None, outward: bool = True,
            seed: int | None = None, quality: str = "fast") -> np.ndarray:
    """Regenerate the masked pixels of `img`: send a region around them (or `box`) to the model, then paste
    its result back under the mask. Everything outside the mask stays the same."""
    if not mask.any():
        return img.copy()
    x0, y0, x1, y1 = box or mask_box(mask)
    crop, mcrop = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    with tempfile.TemporaryDirectory() as tmp:
        crop_path, mask_path = Path(tmp) / "crop.png", Path(tmp) / "mask.png"
        write_image(crop_path, crop)
        write_image(mask_path, mcrop)
        out = cv2.imdecode(np.frombuffer(backend.repair(crop_path, mask_path, prompt, a.seed if seed is None else seed,
                                                        a.megapixels, quality),
                                         np.uint8), cv2.IMREAD_COLOR)
    patch = tone_match(cv2.resize(out, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4), crop)
    result = img.copy()
    result[y0:y1, x0:x1] = paste(crop, patch, mcrop, outward)
    return result


def repair_damage(backend, original: Path, mask: Path, a) -> bytes:
    """Regenerate only the damaged areas, see inpaint."""
    img = read_image(original)
    return cv2.imencode(".png", inpaint(backend, img, read_mask(mask, img.shape[:2]), a.repair_prompt, a))[1].tobytes()


def merge_boxes(boxes: list[tuple]) -> list[tuple]:
    """Join the boxes that overlap, so two crops never meet along a seam."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = boxes[i], boxes[j]
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    boxes[i] = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def candidate_sheet(candidates: list[tuple[int, Path]], fill: np.ndarray, out: Path) -> None:
    """The candidate fills side by side, cropped to where they were generated and labelled with their seeds."""
    ys, xs = np.nonzero(fill)
    if len(xs):
        pad = round(0.05 * max(fill.shape))
        y0, y1 = max(0, ys.min() - pad), min(fill.shape[0], ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(fill.shape[1], xs.max() + 1 + pad)
    else:
        y0, x0, y1, x1 = 0, 0, fill.shape[0], fill.shape[1]
    tiles = []
    for seed, path in candidates:
        im = read_image(path)[y0:y1, x0:x1]
        scale = 400 / im.shape[0]
        im = cv2.resize(im, (max(1, round(im.shape[1] * scale)), 400), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"seed {seed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles += [im, np.full((400, 12, 3), 255, np.uint8)]
    write_image(out, np.hstack(tiles[:-1]))


def frame_rect(frame_shape: tuple, box: tuple) -> tuple[int, int, int, int]:
    """Where the frame (upscaled from the rectangle `box`) sits in the upscaled picture: x, y, w, h."""
    x0, y0, x1, y1 = box
    fh, fw = frame_shape[:2]
    return round(x0 * fw / (x1 - x0)), round(y0 * fh / (y1 - y0)), fw, fh


def colour_context(canvas: np.ndarray, frame: np.ndarray, base_shape: tuple, box: tuple) -> np.ndarray:
    """The full frame placed back into its uncropped surroundings, at the frame's own scale.

    `canvas` is the upscaled picture (uncropped), `base_shape` the size of the picture it was made from, `box` the
    rectangle in that picture. The canvas is scaled to the frame's scale and the frame is pasted at its rectangle.
    """
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = box
    sx, sy = fw / (x1 - x0), fh / (y1 - y0)
    x, y, w, h = frame_rect(frame.shape, box)
    # The usual size, grown by a pixel where the rounding of the frame's own position would overflow it.
    W = max(round(base_shape[1] * sx), x + w + round((base_shape[1] - x1) * sx))
    H = max(round(base_shape[0] * sy), y + h + round((base_shape[0] - y1) * sy))
    out = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_CUBIC)
    out[y:y + h, x:x + w] = frame
    return out


def crop_back(img: np.ndarray, rect: tuple) -> np.ndarray:
    x, y, w, h = rect
    return img[y:y + h, x:x + w]


def colour_frame_bytes(backend, canvas: Path, frame_shape: tuple, box: tuple) -> bytes:
    """DDColor on the canvas (one call), cropped back to the frame's rectangle: the frame's exact size."""
    out = cv2.imdecode(np.frombuffer(backend.colorize(canvas), np.uint8), cv2.IMREAD_COLOR)
    return cv2.imencode(".png", crop_back(out, frame_rect(frame_shape, box)))[1].tobytes()


def generative_frame(base_img: np.ndarray, gen_img: np.ndarray, box: tuple, fill: np.ndarray, a) -> bytes:
    """The rectangle `box` with its corners and missing paper taken from the aligned generative variant."""
    x0, y0, x1, y1 = box
    long = max(base_img.shape[:2])
    corners = corner_squares(box, base_img.shape[:2], long)[y0:y1, x0:x1] > 0
    f = fill[y0:y1, x0:x1] > 0
    # Bare bed (and its antialiased rim) takes the generative fully, corners included: a ramp there leaves a seam.
    bed = (scanner_lid(base_img) > 0)[y0:y1, x0:x1].astype(np.uint8)
    bare = (cv2.dilate(bed, np.ones((5, 5), np.uint8)) > 0) & f
    out = generative_fill(base_img[y0:y1, x0:x1], gen_img[y0:y1, x0:x1], corners, bare,
                          round(CORNER_SQUARE * long))
    # The same speck filter as the model fills: bright specks on the smooth dark areas (fill only).
    med = cv2.medianBlur(out, 5)
    speck = (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(int) - cv2.cvtColor(med, cv2.COLOR_BGR2GRAY) > 20) & f
    out[speck] = med[speck]
    return cv2.imencode(".png", out)[1].tobytes()


def full_frame(backend, base: Path, box: tuple, fill: np.ndarray, a, seed: int) -> bytes:
    """Fill the picture's missing edges, corners and mount, inside the whole rectangle `box`.

    Each separate hole is generated from a padded crop around it, so the model works near native resolution.
    Crops that overlap are generated together.
    """
    x0, y0, x1, y1 = box
    img = read_image(base)[y0:y1, x0:x1]
    m = fill[y0:y1, x0:x1]
    todo = m > 0
    h, w = img.shape[:2]
    n, parts = cv2.connectedComponents(todo.astype(np.uint8))
    crops = []
    for i in range(1, n):
        ys, xs = np.nonzero(parts == i)
        size = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        pad = max(16, round(0.5 * size), round(0.1 * max(h, w)))  # the floor and backdrop around it, not just the hole
        crops.append((max(0, xs.min() - pad), max(0, ys.min() - pad),
                      min(w, xs.max() + 1 + pad), min(h, ys.max() + 1 + pad)))
    for bx in merge_boxes(crops):
        if not todo[bx[1]:bx[3], bx[0]:bx[2]].any():  # already generated with a neighbouring crop
            continue
        mask = np.zeros((h, w), np.uint8)
        mask[bx[1]:bx[3], bx[0]:bx[2]] = todo[bx[1]:bx[3], bx[0]:bx[2]] * 255
        img = inpaint(backend, img, mask, a.frame_prompt, a, box=bx, outward=False, seed=seed,
                      quality=a.frame_quality)
        todo[bx[1]:bx[3], bx[0]:bx[2]] = False
    # The model sprinkles bright specks over the smooth dark areas it generates: median them out (fill only).
    med = cv2.medianBlur(img, 5)
    speck = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(int) - cv2.cvtColor(med, cv2.COLOR_BGR2GRAY) > 20) & (m > 0)
    img[speck] = med[speck]
    return cv2.imencode(".png", img)[1].tobytes()


TURBO_BAND = 0.02  # share of the long side: the band of kept picture along the fill where the Turbo edit is matched to it
TURBO_RAMP = 0.01  # share of the long side: the Turbo fill ramps in from the kept picture over this width


def prefill(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The masked area replaced by a smooth continuation of its surroundings (Telea at quarter size, blurred).

    An edit model copies what it sees: shown the cream mount, it paints the mount back. Shown a plausible backdrop,
    it continues the picture.
    """
    h, w = img.shape[:2]
    m = (mask > 0).astype(np.uint8) * 255
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    big = cv2.resize(cv2.inpaint(small, ms, 5, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), 6)
    out = img.copy()
    out[m > 0] = big[m > 0]
    return out


def smooth_field(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """A smooth field from weighted samples (normalized convolution): fine where there are samples, coarser away
    from them, so it extrapolates into the unsampled area instead of falling to zero."""
    out = None
    for s in [sigma * 16, sigma * 8, sigma * 4, sigma * 2, sigma]:
        num = cv2.GaussianBlur(values * weight, (0, 0), s)
        den = cv2.GaussianBlur(weight, (0, 0), s)
        f = num / np.maximum(den, 1e-6)
        out = f if out is None else np.where(den > 0.05, f, out)
    return out


def match_luminance(gen: np.ndarray, base: np.ndarray, fill: np.ndarray, band: int) -> np.ndarray:
    """`gen`, a whole-image edit of `base`, with its lightness shifted by a smooth field so that it continues the
    kept picture across the fill's boundary: the model renders the backdrop a few L lighter or darker than the
    print, and the step along the oval reads as a ring. The shift is measured where both show the same thing, the
    band of kept picture `band` px inside the fill's boundary, and extrapolated into the fill."""
    lab = to_lab(gen)
    kept = (fill == 0).astype(np.uint8)
    inward = cv2.distanceTransform(kept, cv2.DIST_L2, 5)
    where = ((inward > 0) & (inward <= band)).astype(np.float32)
    shift = smooth_field(to_lab(base)[..., 0] - lab[..., 0], where, float(band))
    lab[..., 0] = np.clip(lab[..., 0] + shift, 0, 100)
    return from_lab(lab)


def turbo_full_frame(backend, base: Path, box: tuple, fill: np.ndarray, a, seed: int) -> bytes:
    """Like full_frame, for Qwen-Image-2.1-Turbo, whose masked inpaint copies the mount it sees in its reference.

    The whole rectangle is edited at once from a pre-filled copy (the mount replaced by a continuation of the
    picture), the edit is aligned to the rectangle, matched to the kept picture's lightness along the fill's
    boundary, and only the fill is taken from it, ramping in from the kept picture. Outside the ramp the picture
    is kept bit for bit.
    """
    x0, y0, x1, y1 = box
    whole = read_image(base)
    img, m, long = whole[y0:y1, x0:x1], fill[y0:y1, x0:x1], max(whole.shape[:2])
    if not m.any():
        return cv2.imencode(".png", img)[1].tobytes()
    pre = prefill(img, m)
    with tempfile.TemporaryDirectory() as tmp:
        pre_path = Path(tmp) / "prefilled.png"
        write_image(pre_path, pre)
        gen = cv2.imdecode(np.frombuffer(backend.generative(pre_path, a.frame_prompt, seed, a.megapixels), np.uint8),
                           cv2.IMREAD_COLOR)
    gen = tone_match(match_luminance(align_to(gen, pre), img, m, max(8, round(TURBO_BAND * long))), img)
    dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = (np.clip(dist / max(4, round(TURBO_RAMP * long)), 0, 1) * (m > 0))[..., None]
    out = (img * (1 - alpha) + gen * alpha + 0.5).astype(np.uint8)
    med = cv2.medianBlur(out, 5)  # the same speck filter as full_frame
    speck = (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(int) - cv2.cvtColor(med, cv2.COLOR_BGR2GRAY) > 20) & (m > 0)
    out[speck] = med[speck]
    return cv2.imencode(".png", out)[1].tobytes()


def frame_overlay(img: np.ndarray, box: tuple, fill: np.ndarray) -> np.ndarray:
    """The frame rectangle in green, the area the model will fill in red."""
    out = damage_overlay(img, fill)
    x0, y0, x1, y1 = box
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), max(2, max(img.shape[:2]) // 300))
    return out


def output_names(files: list[Path]) -> list[str]:
    """One output folder name per input: the file stem, numbered when two inputs share it."""
    names, seen = [], set()
    for f in files:
        name, n = f.stem, 1
        while name.lower() in seen:  # case-insensitive, like Windows and macOS file systems
            n += 1
            name = f"{f.stem}-{n}"
        seen.add(name.lower())
        names.append(name)
    return names


def restore_one(src: Path, out_root: Path, backend, a, name: str | None = None) -> None:
    name = name or src.stem
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    print(f"▶ {name}  [{backend.name}]", flush=True)
    original = d / f"00-original{src.suffix.lower()}"
    if not original.exists() or digest(original) != digest(src):
        shutil.copyfile(src, original)

    conservative = d / "10-conservative.png"
    stage(conservative, "conservative (upscale + CodeFormer)", [original],
          {"backend": backend.name, "fidelity": a.fidelity, "upscale": not a.no_upscale}, a.force,
          lambda: backend.conservative(original, a.fidelity, not a.no_upscale))
    variants = [(conservative, a.tone)]

    color_base = conservative
    if a.generative or a.repair or a.full_frame:  # the repair finds the damage, a full frame takes its corners
        generative = d / "20-generative.png"
        stage(generative, f"generative ({QWEN_NAMES[a.qwen_model]})", [original],
              {"backend": backend.name, "prompt": a.prompt, "seed": a.seed, "megapixels": a.megapixels, **qwen_key(a)}, a.force,
              lambda: backend.generative(original, a.prompt, a.seed, a.megapixels))
    if a.generative or a.repair:
        variants.append((generative, a.tone))
        color_base = generative

    if a.repair:
        orig_img = read_image(original)
        if a.mask:
            m = read_mask(Path(a.mask).expanduser(), orig_img.shape[:2])
        else:
            m = damage_mask(orig_img, read_image(generative))
        mask = d / "25-damage-mask.png"
        write_image(mask, m)
        write_image(d / "25-damage-overlay.jpg", damage_overlay(orig_img, m))
        print(f"  ✓ damage mask ({'from ' + a.mask if a.mask else 'detected'}: {100 * (m > 0).mean():.1f}% of the photo)")
        repaired_raw = d / "26-repaired-raw.png"
        stage(repaired_raw, f"repair ({QWEN_NAMES[a.qwen_model]}, damaged areas only)", [original, mask],
              {"backend": backend.name, "prompt": a.repair_prompt, "seed": a.seed, "megapixels": a.megapixels, **qwen_key(a)},
              a.force, lambda: repair_damage(backend, original, mask, a))
        repaired = d / "27-repaired.png"
        stage(repaired, "repaired: upscale + CodeFormer", [repaired_raw],
              {"backend": backend.name, "fidelity": a.fidelity, "upscale": not a.no_upscale}, a.force,
              lambda: backend.conservative(repaired_raw, a.fidelity, not a.no_upscale))
        variants.append((repaired, a.tone))

    frame = None
    frame_seed = a.frame_seed if a.frame_seed is not None else a.seed
    if a.full_frame:
        base = d / "26-repaired-raw.png" if a.repair else original
        base_img = read_image(base)
        user_fill = read_mask(Path(a.frame_mask).expanduser(), base_img.shape[:2]) if a.frame_mask else None
        box, fill, mode = frame_region(base_img, a.frame, user_fill)
        write_image(d / "28-frame-overlay.jpg", frame_overlay(base_img, box, fill))
        x0, y0, x1, y1 = box  # the original in the same rectangle: the "before" of the full-frame comparisons
        write_image(d / "28-original-frame.jpg", read_image(original)[y0:y1, x0:x1])
        bw, bh = box[2] - box[0], box[3] - box[1]
        kept = bw * bh / (base_img.shape[0] * base_img.shape[1])
        source = "from --frame" if a.frame else "from --frame-mask" if a.frame_mask else "detected"
        print(f"  ✓ frame {bw}x{bh} at {','.join(map(str, box))} ({source}; "
              f"{100 * (fill > 0).sum() / (bw * bh):.1f}% to fill)")
        if kept < 0.9:
            print(f"  ! the frame keeps only {kept:.0%} of the photo: check 28-frame-overlay.jpg")
        frame_mask = d / "28-frame-mask.png"  # an input of the stage, so a change of detection reruns it
        write_image(frame_mask, fill)
        frame_raw = d / "28-full-frame-raw.png"
        sheet = d / "28-full-frame-candidates.jpg"
        sheet.unlink(missing_ok=True)  # a sheet from an earlier run with more candidates is stale
        if mode == "generative":
            # A card on a scanner bed: the corners come from the generative variant, aligned to the base. No model
            # call, so no candidates.
            if a.frame_candidates > 1 or a.frame_seed is not None:
                print("  · corners come from the generative variant: --frame-candidates and --frame-seed do not apply")
            generative_path = d / "20-generative.png"
            stage(frame_raw, "full frame: corners from the generative variant", [base, frame_mask, generative_path],
                  {"mode": "generative-corners", "rect": box}, a.force,
                  lambda: generative_frame(base_img, align_to(read_image(generative_path), base_img), box, fill, a))
        else:
            # Candidates are the seeds from --seed on; --frame-seed picks one (generated if it is not one of them).
            seeds = list(range(a.seed, a.seed + a.frame_candidates))
            if a.frame_candidates == 1:
                seeds = [frame_seed]
            elif frame_seed not in seeds:
                seeds.append(frame_seed)
            candidates = []
            # Turbo edits the whole rectangle from a pre-filled copy (its masked inpaint copies the mount it sees);
            # 2511 inpaints the fill in place. The key tells them apart so an earlier Turbo fill is not reused.
            turbo = a.qwen_model == "turbo"
            fill_frame = turbo_full_frame if turbo else full_frame
            how = "edits the pre-filled rectangle" if turbo else "fills the rectangle"
            for seed in seeds:
                raw = d / f"28-full-frame-raw-s{seed}.png"
                stage(raw, f"full frame, seed {seed} ({QWEN_NAMES[a.qwen_model]}, {how})", [base, frame_mask],
                      {"backend": backend.name, "prompt": a.frame_prompt, "seed": seed, "megapixels": a.megapixels,
                       "rect": box, "quality": a.frame_quality, **qwen_key(a), **({"fill": "prefilled-edit"} if turbo else {})},
                      a.force, lambda seed=seed: fill_frame(backend, base, box, fill, a, seed))
                candidates.append((seed, raw))
            if len(candidates) > 1:
                candidate_sheet([(seed, raw) for seed, raw in candidates if seed in seeds[:a.frame_candidates]],
                                fill[box[1]:box[3], box[0]:box[2]], sheet)
            chosen = d / f"28-full-frame-raw-s{frame_seed}.png"
            stage(frame_raw, "full frame: chosen candidate", [chosen], {"seed": frame_seed}, a.force,
                  lambda: chosen.read_bytes())
        frame = d / "29-full-frame.png"
        stage(frame, "full frame: upscale + CodeFormer", [frame_raw],
              {"backend": backend.name, "fidelity": a.fidelity, "upscale": not a.no_upscale}, a.force,
              lambda: backend.conservative(frame_raw, a.fidelity, not a.no_upscale))
        variants.append((frame, a.tone))

    if a.colorize:
        # The generative (or conservative) colour version, plus one of the repaired image: it keeps the
        # original faces, but the generative one usually looks cleaner, so it doesn't replace it.
        bases = [(color_base, d / "30-color.png")]
        if a.repair:
            bases.append((repaired, d / "31-repaired-color.png"))
        for src, color in bases:
            stage(color, f"colorize {src.stem} (DDColor)", [src], {"backend": backend.name}, a.force,
                  lambda src=src: backend.colorize(src))
            variants.append((color, "color"))
        if frame:
            # The frame alone looks green to DDColor (no surroundings to tell the skin and paper apart), so it is
            # coloured inside its uncropped picture and cropped back: one colorize call, the frame's exact size.
            canvas_src = repaired if a.repair else conservative
            canvas = d / "32-full-frame-canvas.png"

            def compose():
                out = colour_context(read_image(canvas_src), read_image(frame), base_img.shape[:2], box)
                return cv2.imencode(".png", out)[1].tobytes()

            stage(canvas, "full frame: placed in its surroundings", [canvas_src, frame, base],
                  {"rect": box}, a.force, compose)
            color = d / "32-full-frame-color.png"
            stage(color, "colorize the full frame in its surroundings (DDColor)", [canvas],
                  {"backend": backend.name, "rect": box}, a.force,
                  lambda: colour_frame_bytes(backend, canvas, read_image(frame).shape, box))
            variants.append((color, "color"))

    orig_img = read_image(original)
    for path, tone in variants:
        img = read_image(path)
        if tone != "color":
            img = tone_match(img, orig_img)
        final = path.with_name(path.stem + "-final.jpg")
        write_image(final, finish(img, tone, sharpen=not a.no_sharpen, saturation=a.saturation))
        print(f"  ✓ {final.name} (tone: {tone})")

    manifest = {k: v for k, v in vars(a).items() if k != "func"}
    manifest.update(input=str(src), backend=backend.name, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def cmd_run(a) -> int:
    backend = make_backend(a.backend, a.qwen_model)
    if a.qwen_model == "turbo" and a.frame_quality == "high":
        print("· --frame-quality high does not apply to --qwen-model turbo (fixed 8 steps): using fast")
        a.frame_quality = "fast"
    stages = (["conservative"] + ["generative"] * (a.generative or a.repair or a.full_frame)
              + ["repair"] * (a.repair or a.full_frame)
              + ["colorize"] * a.colorize)
    problems = backend.check(stages)
    if problems:
        print("backend not ready:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 2
    out_root = Path(a.out).expanduser().resolve()
    failed = 0
    files = collect_inputs(a.inputs)
    if a.mask and (len(files) != 1 or not a.repair):
        print("--mask needs --repair and a single photo", file=sys.stderr)
        return 2
    if (a.frame or a.frame_mask) and (len(files) != 1 or not a.full_frame):
        print("--frame and --frame-mask need --full-frame and a single photo", file=sys.stderr)
        return 2
    if a.frame_candidates < 1:
        print("--frame-candidates needs at least 1", file=sys.stderr)
        return 2
    if a.frame:
        h, w = read_image(files[0]).shape[:2]
        x0, y0, x1, y1 = a.frame
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            print(f"--frame {x0},{y0},{x1},{y1} is outside the {w}x{h} photo", file=sys.stderr)
            return 2
    for src, name in zip(files, output_names(files)):
        try:
            restore_one(src, out_root, backend, a, name)
        except Exception as e:  # keep going with the other photos
            failed += 1
            print(f"  ✗ {src.name}: {e}", file=sys.stderr)
    print(f"comparison page: {write_sheet(out_root)}")
    return 1 if failed else 0


def cmd_doctor(a) -> int:
    backend = make_backend(a.backend, a.qwen_model)
    failed = []
    for name, flag in (("conservative", ""), ("generative", " (--generative)"), ("repair", " (--repair, --full-frame)"),
                       ("colorize", " (--colorize)")):
        problems = backend.check([name])
        print(f"{'✗' if problems else '✓'} {name}{flag}")
        for p in problems:
            print(f"    {p}")
        if problems:
            failed.append(name)
    if "conservative" in failed:
        print(f"backend '{a.backend}' not ready")
        return 1
    print(f"backend '{a.backend}' ready" + (f", without {' and '.join(failed)}" if failed else ""))
    return 0


# ---------------------------------------------------------------- local setup

NODES = {
    "facerestore_cf": "https://github.com/mav-rik/facerestore_cf.git",
    "ComfyUI-DDColor": "https://github.com/kijai/ComfyUI-DDColor.git",
}
HF = "https://huggingface.co"
MODELS = {
    "upscale_models/RealESRGAN_x2plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "facerestore_models/codeformer-v0.1.0.pth":
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "vae/qwen_image_vae.safetensors":
        f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors":
        f"{HF}/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors":
        f"{HF}/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
    "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors":
        f"{HF}/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
}


TURBO_HF = f"{HF}/Comfy-Org/Qwen-Image-2.1/resolve/main"
TURBO_MODELS = {
    "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors": f"{TURBO_HF}/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
    "text_encoders/qwen3vl_8b_w4a8.safetensors": f"{TURBO_HF}/text_encoders/qwen3vl_8b_w4a8.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors": f"{TURBO_HF}/vae/qwen_image_2.1_vae_bf16.safetensors",
}


def download(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = done + int(r.headers.get("content-length", 0))
        with open(part, "ab" if r.status_code == 206 else "wb") as f:
            got = done if r.status_code == 206 else 0
            last = 0.0
            for chunk in r.iter_content(8 << 20):
                f.write(chunk)
                got += len(chunk)
                if time.time() - last > 5:
                    print(f"    {dest.name}: {got / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
                    last = time.time()
    part.replace(dest)


def cmd_setup_local(a) -> int:
    comfy = Path(a.comfy).expanduser().resolve()
    if not (comfy / "main.py").exists():
        print(f"{comfy} is not a ComfyUI checkout (no main.py)", file=sys.stderr)
        return 2
    for name, repo in NODES.items():
        dest = comfy / "custom_nodes" / name
        if not dest.exists():
            subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)], check=True)
        req = dest / "requirements.txt"
        if req.exists():
            has_pip = subprocess.run([a.python, "-m", "pip", "--version"], capture_output=True).returncode == 0
            # uv-created environments have no pip; install into them with uv instead.
            cmd = [a.python, "-m", "pip", "install", "-q"] if has_pip else ["uv", "pip", "install", "-q", "--python", a.python]
            subprocess.run([*cmd, "-r", str(req)], check=True)
        print(f"✓ node {name}")
    models = Path(a.models).expanduser().resolve() if a.models else comfy / "models"
    for rel, url in {**MODELS, **(TURBO_MODELS if a.turbo else {})}.items():
        if a.skip_generative and any(s in rel for s in ("qwen", "Qwen")):
            continue
        dest = models / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"  downloading {rel}", flush=True)
            download(url, dest)
        print(f"✓ model {rel}")
    print("done. Restart ComfyUI so it picks up the new nodes.")
    return 0


# ---------------------------------------------------------------- comparison page

FULL_FRAME_VARIANTS = {"29-full-frame-final.jpg", "32-full-frame-color-final.jpg"}
LABELS = {"10-conservative-final.jpg": "Conservative", "20-generative-final.jpg": "Generative",
          "27-repaired-final.jpg": "Repaired", "29-full-frame-final.jpg": "Full frame",
          "30-color-final.jpg": "Colorized", "31-repaired-color-final.jpg": "Repaired, colorized",
          "32-full-frame-color-final.jpg": "Full frame, colorized"}


def write_sheet(out_root: Path) -> Path:
    rows = []
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        original = next((f.name for f in d.iterdir() if f.name.startswith("00-original")), None)
        if not original:
            continue
        src = lambda f: urllib.parse.quote(f"{d.name}/{f}")
        cells = [f'<figure><img src="{src(original)}" alt="Original"><figcaption>Original</figcaption></figure>']
        for f, label in LABELS.items():
            if (d / f).exists():
                # Full-frame variants are cropped to their rectangle: compare them with the original in that rectangle.
                before = "28-original-frame.jpg" if f in FULL_FRAME_VARIANTS and (d / "28-original-frame.jpg").exists() else original
                cells.append(f"""<figure><div class="cmp" style="--pos:50%">
<img src="{src(f)}" alt="{label}"><img class="before" src="{src(before)}" alt="Original">
<input type="range" min="0" max="100" value="50" aria-label="Compare with original"></div>
<figcaption>{label} <a href="{src(f)}" target="_blank">open</a></figcaption></figure>""")
            if f == "29-full-frame-final.jpg" and (d / "28-full-frame-candidates.jpg").exists():
                cells.append(f'<figure><img src="{src("28-full-frame-candidates.jpg")}" alt="Full frame candidates">'
                             f'<figcaption>Full frame candidates (seeds)</figcaption></figure>')
        rows.append(f'<section><h2>{html.escape(d.name)}</h2><div class="grid">{"".join(cells)}</div></section>')
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo restorations</title><style>
:root {{ --bg:#f6f3ee; --fg:#2a2622; --muted:#7a7168; --card:#fff; --line:#e2dcd3; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1b1917; --fg:#ece6de; --muted:#a39a90; --card:#262320; --line:#3a3530; }} }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; }}
h1 {{ font-weight:600; margin:0 0 4px; }} p {{ color:var(--muted); margin:0 0 24px; }}
section {{ margin-bottom:40px; }} h2 {{ font-size:16px; font-weight:600; margin:0 0 12px; }}
.grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); }}
figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px; }}
figure img {{ display:block; width:100%; height:auto; border-radius:4px; }}
figcaption {{ color:var(--muted); font-size:13px; padding-top:6px; display:flex; justify-content:space-between; }}
figcaption a {{ color:inherit; }}
.cmp {{ position:relative; }}
.cmp .before {{ position:absolute; inset:0; height:100%; object-fit:fill; clip-path:inset(0 calc(100% - var(--pos)) 0 0); }}
.cmp::after {{ content:""; position:absolute; top:0; bottom:0; left:var(--pos); width:2px; background:#fff8; pointer-events:none; }}
.cmp input {{ position:absolute; inset:0; width:100%; height:100%; margin:0; opacity:0; cursor:ew-resize; }}
</style></head><body>
<h1>Photo restorations</h1><p>Drag across each variant to compare it with the original (left side = original).</p>
{chr(10).join(rows)}
<script>for (const i of document.querySelectorAll('.cmp input')) i.addEventListener('input', () => i.parentElement.style.setProperty('--pos', i.value + '%'))</script>
</body></html>"""
    out = out_root / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


# ---------------------------------------------------------------- cli

def parse_frame(text: str) -> tuple[int, int, int, int]:
    try:
        x0, y0, x1, y1 = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected X0,Y0,X1,Y1 in pixels") from None
    return x0, y0, x1, y1


def main() -> int:
    # Windows consoles/pipes default to a legacy code page that can't print ✓/✗/▶.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Old photo restoration (ComfyUI or fal.ai)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="restore photos")
    r.add_argument("inputs", nargs="+", help="image files or folders")
    r.add_argument("--backend", choices=["local", "fal"], default=os.environ.get("RESTORE_BACKEND", "local"))
    r.add_argument("--out", default="restored")
    r.add_argument("--fidelity", type=float, default=0.7, help="CodeFormer fidelity 0..1 (higher = closer to original)")
    r.add_argument("--no-upscale", action="store_true", help="skip the x2 upscale before face restoration")
    r.add_argument("--generative", action="store_true", help="add a Qwen-Image-Edit restoration variant")
    r.add_argument("--prompt", default=RESTORE_PROMPT, help="generative restoration prompt")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--frame-seed", type=int, help="seed of the full-frame fill only (default: --seed); "
                                                  "rerolls a bad fill without rerunning the other AI stages")
    r.add_argument("--megapixels", type=float, default=1.0, help="generative working size (local backend)")
    r.add_argument("--qwen-model", choices=QWEN_MODELS, default="2511",
                   help="Qwen model for the generative, repair and full-frame fills: 2511 (default, Apache-2.0) or "
                        "turbo (experimental, Qwen Research License, non-commercial; local only; needs setup-local --turbo)")
    r.add_argument("--repair", action="store_true",
                   help="add a variant that regenerates only the damaged areas (holes, peeled emulsion, stains); "
                        "implies --generative, which it uses to find the damage")
    r.add_argument("--mask", help="with --repair and a single photo: use this damage mask (white = damaged) "
                                  "instead of the detected one")
    r.add_argument("--repair-prompt", default=REPAIR_PROMPT, help="instruction for --repair")
    r.add_argument("--full-frame", action="store_true",
                   help="add a variant with the mount, vignette and borders removed: a rectangular picture, "
                        "with the missing edges and corners generated (local only). Check 28-frame-overlay.jpg")
    r.add_argument("--frame", type=parse_frame, help="with --full-frame and a single photo: use this rectangle "
                                                     "X0,Y0,X1,Y1 (pixels) instead of the detected one")
    r.add_argument("--frame-mask", help="with --full-frame and a single photo: fill this mask (white = to generate) "
                                        "instead of the detected one; the frame stays the detected one or --frame")
    r.add_argument("--frame-prompt", default=FRAME_PROMPT, help="instruction for --full-frame")
    r.add_argument("--frame-candidates", type=int, default=1,
                   help="with --full-frame: generate this many fills, with consecutive seeds from --seed; "
                        "--frame-seed picks one (its contact sheet is 28-full-frame-candidates.jpg)")
    r.add_argument("--frame-quality", choices=["fast", "high"], default="fast",
                   help="--full-frame fill quality: fast (4 Lightning steps) or high (20 steps, no LoRA, slower)")
    r.add_argument("--colorize", action="store_true", help="add a DDColor colorized variant")
    r.add_argument("--saturation", type=float, default=0.85,
                   help="colorized variant: colour strength (1 = as DDColor makes it, 0 = grey)")
    r.add_argument("--tone", choices=["keep", "neutral", "sepia"], default="keep",
                   help="B/W variants: keep the print's tint, neutral grey, or sepia")
    r.add_argument("--no-sharpen", action="store_true")
    r.add_argument("--force", action="store_true", help="recompute cached AI stages")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("doctor", help="check that a backend is ready")
    d.add_argument("--backend", choices=["local", "fal"], default=os.environ.get("RESTORE_BACKEND", "local"))
    d.add_argument("--qwen-model", choices=QWEN_MODELS, default="2511")
    d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("setup-local", help="install the ComfyUI nodes and models this pipeline needs")
    s.add_argument("--comfy", required=True, help="ComfyUI directory (the one containing main.py)")
    s.add_argument("--python", required=True, help="python executable of ComfyUI's environment")
    s.add_argument("--models", help="models directory (default: <comfy>/models)")
    s.add_argument("--skip-generative", action="store_true", help="skip the ~31 GB Qwen-Image-Edit models")
    s.add_argument("--turbo", action="store_true", help="also download the experimental Qwen-Image-2.1-Turbo files "
                                                       "(~14 GB, Qwen Research License, non-commercial)")
    s.set_defaults(func=cmd_setup_local)

    sh = sub.add_parser("sheet", help="rebuild the comparison page")
    sh.add_argument("out", nargs="?", default="restored")
    sh.set_defaults(func=lambda a: print(write_sheet(Path(a.out).resolve())) or 0)

    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
