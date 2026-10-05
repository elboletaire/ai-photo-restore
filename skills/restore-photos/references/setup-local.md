# Local backend setup (ComfyUI)

Requirements: Python 3.10+, git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and either an NVIDIA GPU (≥12 GB VRAM recommended for the generative stage; 8 GB is enough without it) or an Apple Silicon Mac (works, much slower). Disk: ~1 GB without the generative models, ~32 GB with them.

Run every command yourself and report progress; downloads take a while.

## 1. Reuse an existing ComfyUI if there is one

Ask the user. If ComfyUI is already installed you only need its folder (the one with `main.py`) and the python of its environment:

| Install type | ComfyUI folder | Python |
|---|---|---|
| Windows portable | `...\ComfyUI_windows_portable\ComfyUI` | `...\ComfyUI_windows_portable\python_embeded\python.exe` |
| ComfyUI Desktop (Win/Mac) | the "ComfyUI" base path chosen at install (Settings → About) | `<base path>/.venv/bin/python` (`.venv\Scripts\python.exe` on Windows) |
| comfy-cli / git clone | the clone | the venv used to install it |

Skip to step 3.

## 2. Fresh install with comfy-cli

Linux / WSL / macOS:
```bash
mkdir -p ~/comfy && cd ~/comfy
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install comfy-cli pip   # comfy-cli needs pip inside the venv
comfy --skip-prompt --workspace ~/comfy/ComfyUI install --nvidia --skip-manager --fast-deps   # --m-series on Apple Silicon
```

Windows (PowerShell):
```powershell
mkdir $HOME\comfy; cd $HOME\comfy
uv venv --python 3.12 .venv; .venv\Scripts\activate
uv pip install comfy-cli pip   # comfy-cli needs pip inside the venv
comfy --skip-prompt --workspace $HOME\comfy\ComfyUI install --nvidia --skip-manager --fast-deps
```

Check the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"` (on Mac: `torch.backends.mps.is_available()`).

## 3. Nodes and models

```bash
uv run <skill-dir>/scripts/restore.py setup-local --comfy <ComfyUI folder> --python <ComfyUI python>
```

It clones the `facerestore_cf` and `ComfyUI-DDColor` custom nodes, installs their requirements and downloads the models (resumable). Add `--skip-generative` to skip the ~31 GB Qwen-Image-Edit files when the GPU has < 12 GB or disk is short; then don't use `--generative` (or use the fal backend for it). Use `--models <dir>` to point at a shared models folder.

## 4. Start ComfyUI

- comfy-cli: `comfy --workspace <ComfyUI folder> launch --background` (stop with `comfy stop`).
- Windows portable: `run_nvidia_gpu.bat`. ComfyUI Desktop: just open the app (it serves on port 8000 → set `COMFY_URL=http://127.0.0.1:8000`).

Restart ComfyUI after `setup-local` so it loads the new nodes, then:

```bash
uv run <skill-dir>/scripts/restore.py doctor --backend local
```

## Troubleshooting
- `ComfyUI not reachable`: not running, or another port → set `COMFY_URL`.
- `missing ComfyUI node ...`: ComfyUI wasn't restarted, or the node failed to import → look for `IMPORT FAILED` in the ComfyUI console/log.
- Out of memory on the generative stage: lower `--megapixels` (e.g. 0.7), or close other GPU apps.
- WSL: models on `/mnt/c` or `/mnt/d` load very slowly; keep the ComfyUI folder inside the Linux filesystem.
