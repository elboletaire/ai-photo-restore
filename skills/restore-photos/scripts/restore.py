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


def finish(img: np.ndarray, tone: str, sharpen: bool = True) -> np.ndarray:
    """Tonal finishing: contrast stretch on L, optional re-toning, light unsharp mask."""
    lab = to_lab(img)
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
    elif tone == "color":
        lab[..., 1:] *= 1.08
    return from_lab(lab)


# ---------------------------------------------------------------- backends

class ComfyBackend:
    name = "local"
    NODES = ["FaceRestoreCFWithModel", "DDColor_Colorize", "TextEncodeQwenImageEditPlus", "ImageUpscaleWithModel"]

    def __init__(self, url: str | None = None):
        self.url = (url or os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")
        self.client_id = uuid.uuid4().hex

    def check(self) -> list[str]:
        try:
            info = requests.get(f"{self.url}/object_info", timeout=10).json()
        except requests.RequestException:
            return [f"ComfyUI not reachable at {self.url} (start it, or set COMFY_URL)"]
        problems = [f"missing ComfyUI node {n} (run setup-local)" for n in self.NODES if n not in info]
        for wf in ("conservative", "generative", "colorize"):
            for node in load_workflow(wf).values():
                spec = info.get(node["class_type"], {}).get("input", {}).get("required", {})
                for key, value in node["inputs"].items():
                    options = spec.get(key, [None])[0]
                    if isinstance(options, list) and isinstance(value, str) and value not in options \
                            and node["class_type"] != "LoadImage":
                        problems.append(f"{wf}: {node['class_type']}.{key} = {value} not installed")
        return problems

    def _upload(self, path: Path) -> str:
        with open(path, "rb") as f:
            r = requests.post(f"{self.url}/upload/image", files={"image": (path.name, f)},
                              data={"overwrite": "true"}, timeout=120)
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
        wf = load_workflow("generative")
        set_input(wf, "Input image", "image", self._upload(path))
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

    def __init__(self):
        self.key = os.environ.get("FAL_KEY", "")

    def check(self) -> list[str]:
        return [] if self.key else ["FAL_KEY is not set (create one at https://fal.ai/dashboard/keys)"]

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

    def colorize(self, path: Path) -> bytes:
        return self._call("fal-ai/ddcolor", {"image_url": self._data_uri(path)})


def make_backend(name: str):
    return ComfyBackend() if name == "local" else FalBackend()


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
    node_id = find_node(wf, title)
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


def stage(target: Path, label: str, inputs: Path, settings: dict, force: bool, produce) -> None:
    """Run an AI stage unless `target` was already made from the same input file and settings.

    What each stage was made from is recorded in stages.json next to it, so changing --seed,
    --fidelity, the backend, or recomputing an upstream stage reruns it automatically.
    """
    record = target.parent / "stages.json"
    done = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    key = {**settings, "input": digest(inputs)}
    if target.exists() and not force and done.get(target.name) == key:
        print(f"  = {label} (cached)")
        return
    t0 = time.time()
    target.write_bytes(produce())
    done[target.name] = key
    record.write_text(json.dumps(done, indent=2), encoding="utf-8")
    print(f"  ✓ {label} ({time.time() - t0:.1f}s)", flush=True)


def restore_one(src: Path, out_root: Path, backend, a) -> None:
    d = out_root / src.stem
    d.mkdir(parents=True, exist_ok=True)
    print(f"▶ {src.stem}  [{backend.name}]", flush=True)
    original = d / f"00-original{src.suffix.lower()}"
    if not original.exists() or digest(original) != digest(src):
        shutil.copyfile(src, original)

    conservative = d / "10-conservative.png"
    stage(conservative, "conservative (upscale + CodeFormer)", original,
          {"backend": backend.name, "fidelity": a.fidelity, "upscale": not a.no_upscale}, a.force,
          lambda: backend.conservative(original, a.fidelity, not a.no_upscale))
    variants = [(conservative, a.tone)]

    color_base = conservative
    if a.generative:
        generative = d / "20-generative.png"
        stage(generative, "generative (Qwen-Image-Edit-2511)", original,
              {"backend": backend.name, "prompt": a.prompt, "seed": a.seed, "megapixels": a.megapixels}, a.force,
              lambda: backend.generative(original, a.prompt, a.seed, a.megapixels))
        variants.append((generative, a.tone))
        color_base = generative

    if a.colorize:
        color = d / "30-color.png"
        stage(color, "colorize (DDColor)", color_base, {"backend": backend.name}, a.force,
              lambda: backend.colorize(color_base))
        variants.append((color, "color"))

    orig_img = read_image(original)
    for path, tone in variants:
        img = read_image(path)
        if tone != "color":
            img = tone_match(img, orig_img)
        final = path.with_name(path.stem + "-final.jpg")
        write_image(final, finish(img, tone, sharpen=not a.no_sharpen))
        print(f"  ✓ {final.name} (tone: {tone})")

    manifest = {k: v for k, v in vars(a).items() if k != "func"}
    manifest.update(input=str(src), backend=backend.name, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def cmd_run(a) -> int:
    backend = make_backend(a.backend)
    problems = backend.check()
    if problems:
        print("backend not ready:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 2
    out_root = Path(a.out).expanduser().resolve()
    failed = 0
    for src in collect_inputs(a.inputs):
        try:
            restore_one(src, out_root, backend, a)
        except Exception as e:  # keep going with the other photos
            failed += 1
            print(f"  ✗ {src.name}: {e}", file=sys.stderr)
    print(f"comparison page: {write_sheet(out_root)}")
    return 1 if failed else 0


def cmd_doctor(a) -> int:
    problems = make_backend(a.backend).check()
    for p in problems:
        print(f"✗ {p}")
    if not problems:
        print(f"✓ backend '{a.backend}' ready")
    return 1 if problems else 0


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
    for rel, url in MODELS.items():
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

LABELS = {"10-conservative-final.jpg": "Conservative", "20-generative-final.jpg": "Generative",
          "30-color-final.jpg": "Colorized"}


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
                cells.append(f"""<figure><div class="cmp" style="--pos:50%">
<img src="{src(f)}" alt="{label}"><img class="before" src="{src(original)}" alt="Original">
<input type="range" min="0" max="100" value="50" aria-label="Compare with original"></div>
<figcaption>{label} <a href="{src(f)}" target="_blank">open</a></figcaption></figure>""")
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
    r.add_argument("--megapixels", type=float, default=1.0, help="generative working size (local backend)")
    r.add_argument("--colorize", action="store_true", help="add a DDColor colorized variant")
    r.add_argument("--tone", choices=["keep", "neutral", "sepia"], default="keep",
                   help="B/W variants: keep the print's tint, neutral grey, or sepia")
    r.add_argument("--no-sharpen", action="store_true")
    r.add_argument("--force", action="store_true", help="recompute cached AI stages")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("doctor", help="check that a backend is ready")
    d.add_argument("--backend", choices=["local", "fal"], default=os.environ.get("RESTORE_BACKEND", "local"))
    d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("setup-local", help="install the ComfyUI nodes and models this pipeline needs")
    s.add_argument("--comfy", required=True, help="ComfyUI directory (the one containing main.py)")
    s.add_argument("--python", required=True, help="python executable of ComfyUI's environment")
    s.add_argument("--models", help="models directory (default: <comfy>/models)")
    s.add_argument("--skip-generative", action="store_true", help="skip the ~31 GB Qwen-Image-Edit models")
    s.set_defaults(func=cmd_setup_local)

    sh = sub.add_parser("sheet", help="rebuild the comparison page")
    sh.add_argument("out", nargs="?", default="restored")
    sh.set_defaults(func=lambda a: print(write_sheet(Path(a.out).resolve())) or 0)

    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
