---
name: restore-photos
description: Restore old, vintage or damaged photographs (scratches, stains, tears, fading, blurry faces) and optionally colorize them, locally with ComfyUI or in the cloud with fal.ai. Use when the user asks to restore, repair, clean up, enhance, upscale or colorize old/family/scanned photos (also "restaurar fotos antigues", "restaurar fotos viejas", "acolorir", "colorear").
---

# Restore old photos

All work goes through one script, run with [uv](https://docs.astral.sh/uv/) (installs its own deps, works on Linux, macOS and Windows):

```bash
uv run <skill-dir>/scripts/restore.py <command> ...
```

`<skill-dir>` is the directory containing this file. Never modify the user's originals: the script copies each photo into `<out>/<name>/00-original.*` and writes every stage next to it.

## Pipeline

| Stage | Model | File | When |
|---|---|---|---|
| x2 upscale + face restoration | RealESRGAN + CodeFormer | `10-conservative.png` | always |
| Generative restoration (removes scratches, stains, tears) | Qwen-Image-Edit-2511 | `20-generative.png` | `--generative` |
| Colorization | DDColor | `30-color.png` | `--colorize` |
| Tone match: keep restored luminance, rebuild the print's original tint (removes colour the AI invents in eyes/skin of B/W photos) | — | (in memory) | B/W variants |
| Finish: contrast, tone, sharpen | — | `*-final.jpg` | always |
| Before/after page with sliders | — | `<out>/index.html` | always |

## 1. Pick and check a backend

- **local**: ComfyUI on this machine (`COMFY_URL`, default `http://127.0.0.1:8188`). Needs an NVIDIA GPU (≥12 GB VRAM for `--generative`) or Apple Silicon (slow). Photos never leave the machine.
- **fal**: fal.ai hosted models, needs `FAL_KEY`. No GPU needed. Costs roughly $0.005 (conservative) to $0.04 (with generative) per photo. Photos are uploaded to fal.ai — say so before using it with family photos.

Ask the user which one if it isn't obvious, then run `restore.py doctor --backend <b>`.
If it fails, follow [references/setup-local.md](references/setup-local.md) or [references/setup-cloud.md](references/setup-cloud.md).
The backend can be preset with `RESTORE_BACKEND=local|fal`.

## 2. Look at each photo first

Read the images and decide per photo:
- **Damage** (tears, stains, scratches, missing areas) → add `--generative`. The conservative pass does not remove damage.
- **Faces**: `--fidelity` 0.7 by default. Faces look waxy/changed → 0.8–0.9. Very blurry faces → 0.5.
- **Large sharp scan** (> ~3000 px, faces already detailed) → `--no-upscale`.
- **Tone** for B/W variants: `keep` (default, the print's own tint), `neutral` (grey) or `sepia`.
- **Colorize** only if asked.

## 3. Run

```bash
uv run <skill-dir>/scripts/restore.py run <files or folders> --out restored [--backend local|fal] \
  [--generative] [--colorize] [--fidelity 0.7] [--tone keep|neutral|sepia] [--seed 42]
```

AI stages are cached per photo and rerun on their own when their input or settings change (`--seed`, `--fidelity`, `--prompt`, `--no-upscale`, backend). Changing tone/finish is instant (just rerun). Try another `--seed` when a generative result is off. `--force` recomputes every AI stage; to redo just one, delete its file (e.g. `20-generative.png`).
`--prompt "..."` overrides the generative instruction for a specific photo (e.g. "also remove the handwritten date in the corner"). Keep the "keep the exact same people, faces, expressions…" part.

## 4. Review before reporting

Read each `*-final.jpg` next to `00-original.*` and check:
- **Identity** — the most important check. Generative output can change face proportions, eyes or expression (e.g. remove a smile). If a person looks different, retry with another seed, or recommend the conservative variant.
- **Artifacts** — pasted-looking face edges or waxy skin (CodeFormer: raise fidelity), invented objects or text, broken hands.
- **Composition** — framing, oval vignettes and borders kept.
- **Colour** — DDColor sometimes over-saturates small items (pocket squares, flowers).

Then point the user to `<out>/index.html` and recommend one variant per photo, saying why.

## Performance (RTX 5080, local)
Conservative 4–10 s, generative 15–40 s (first run loads ~30 GB of models), DDColor 5–15 s per photo.
