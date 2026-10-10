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
| Damage mask: where the original is much lighter than the generative version (bare paper) | — | `25-damage-mask.png`, `25-damage-overlay.jpg` | `--repair` |
| Repair: regenerate only the masked areas, then upscale + CodeFormer | Qwen-Image-Edit-2511, RealESRGAN, CodeFormer | `26-repaired-raw.png`, `27-repaired.png` | `--repair` (local only) |
| Full frame: picture rectangle found, mount/vignette removed, worn rim cropped and corners taken from the generative variant (vignettes: generated), then upscale + CodeFormer | Qwen-Image-Edit-2511, RealESRGAN, CodeFormer | `28-frame-overlay.jpg`, `28-full-frame-raw.png`, `29-full-frame.png` | `--full-frame` (local only) |
| Colorization of the generative variant (or the conservative one without `--generative`/`--repair`) | DDColor | `30-color.png` | `--colorize` |
| Colorization of the repaired variant | DDColor | `31-repaired-color.png` | `--colorize --repair` |
| Colorization of the full frame | DDColor | `32-full-frame-color.png` | `--colorize --full-frame` |
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
- **Holes, missing corners, peeled emulsion** (light paper showing through) → add `--repair`. It regenerates only the damaged areas, so faces, the mount and the photographer's signature stay the original pixels. Prefer it to `--generative` for family photos: the generative variant redraws the whole photo, faces included. `--repair` also produces the generative variant (it uses it to find the damage), so you can compare both.
- **Other damage** (stains, scratches, fading) → `--generative`. The conservative pass does not remove damage.
- **Faces**: `--fidelity` 0.7 by default. Faces look waxy/changed → 0.8–0.9. Very blurry faces → 0.5.
- **Large sharp scan** (> ~3000 px, faces already detailed) → `--no-upscale`.
- **Tone** for B/W variants: `keep` (default, the print's own tint), `neutral` (grey) or `sepia`.
- **Mount, oval vignette or borders** (a card with a picture inside an oval, or worn edges and rounded or missing corners) → add `--full-frame`. It makes a clean rectangular picture: the mount and the borders are removed and the missing parts are generated, so the variant keeps the picture's own proportions. Use it when the user wants the photo without its frame; it is an extra variant, the others are kept. Always review `28-frame-overlay.jpg` (green = the rectangle, red = what gets generated). To fix it, for a single photo: `--frame X0,Y0,X1,Y1` (pixels) moves the rectangle, keeping the detected fill; `--frame-mask <file>` (white = generate) replaces the fill, e.g. to paint over a mount the detection missed; both can be combined. A print that keeps less than 90% of the scan gets a warning. A reroll of a bad fill: `--frame-seed N` (default: `--seed`) reruns only the full-frame stages. `--frame-seed` (and `--frame-candidates`, `--frame-quality`, `--frame-prompt`) applies to vignettes and mounts only: a card on a scanner bed has no model fill to reroll.

Limitations of `--full-frame`:
- On a card that runs to the scan's edges (a scanner bed around it), the worn rim is cropped (about 3% per side, on the sides the bed touches) and the corners (squares of 5%) are taken from the generative variant: a card on a bed uses no candidates and no retries. Check that the rectangle does not cut a face, and fix the fill with `--frame-mask` if needed.
- Generated corners and hands can invent content (a dark wedge where a floor should be, a claw-like hand). Compare with the original and retry (section 5).
- `--full-frame` always computes the generative variant as well (one extra model call), even for a vignette that does not use it.
- Vignette mode also regenerates a thin band just inside the oval's shadow ring, and the outermost 1.5% of the rectangle. `--frame-quality high` can still leave a light line at the frame edge on a vignette (it continues the cream mount); fast does not. Try fast first.
- Vignette fills are generated at about 1 MP.
- **Colorize** only if asked. `--saturation` (default 0.85) sets the colour strength; lower it if colours look painted-on, raise it towards 1 for DDColor's full colour.

## 3. Run

```bash
uv run <skill-dir>/scripts/restore.py run <files or folders> --out restored [--backend local|fal] \
  [--generative] [--colorize] [--fidelity 0.7] [--tone keep|neutral|sepia] [--seed 42] [--qwen-model 2511|turbo]
```

`--qwen-model turbo` (experimental, local only) runs the Qwen stages (generative, repair, full-frame fills) on Qwen-Image-2.1-Turbo: ~14 GB instead of ~30 GB, 8 steps, about 2x faster in tests. It keeps composition and identity better for whole-image restoration, but it is weaker at repair and full frame (it tended to keep the oval), and `--frame-quality high` does not apply to it. Its licence is the Qwen Research License, **non-commercial use only**: say so to the user before using it for anything but personal photos. Outputs of the two models are cached separately. Needs `setup-local --turbo`.

AI stages are cached per photo and rerun on their own when their input or settings change (`--seed`, `--fidelity`, `--prompt`, `--no-upscale`, backend). Changing tone/finish is instant (just rerun). Try another `--seed` when a generative result is off. `--force` recomputes every AI stage; to redo just one, delete its file (e.g. `20-generative.png`).
`--prompt "..."` overrides the generative instruction for a specific photo (e.g. "also remove the handwritten date in the corner"). Keep the "keep the exact same people, faces, expressions…" part.

## 4. Review before reporting

With `--full-frame`, read `28-frame-overlay.jpg` first: the green rectangle must hold the whole picture and the red area must be only the mount, vignette, missing corners or worn/bare paper at the edges. Fix the rectangle with `--frame`, and the fill area with `--frame-mask` (white = generate).

With `--repair`, first read `25-damage-overlay.jpg` (damage in red). Thin cracks, small specks and damage on light areas are often missed; damage on dark areas is found best. If the mask misses something important or covers something it shouldn't, make a corrected black and white mask (white = damaged, any size, the user can paint it in any editor) and rerun that photo with `--mask <file>`.

Read each `*-final.jpg` next to `00-original.*` and check:
- **Identity** — the most important check. Generative output can change face proportions, eyes or expression (e.g. remove a smile). If a person looks different, retry with another seed, or recommend the conservative variant.
- **Artifacts** — pasted-looking face edges or waxy skin (CodeFormer: raise fidelity), invented objects or text, broken hands.
- **Composition** — framing, oval vignettes and borders kept (for `--full-frame`: the `29-full-frame-final.jpg` must have no mount, vignette or scanner bed left, and the face must be the original's).
- **Colour** — DDColor sometimes over-saturates small items (pocket squares, flowers). If the whole photo looks too colourful, rerun with a lower `--saturation` (instant, nothing is recomputed).

Then point the user to `<out>/index.html` and recommend one variant per photo, saying why.

## 5. Full frame: prompts, candidates and retries

Use this flow for every `--full-frame` photo (local backend only).

1. **Before the run, write a per-photo `--frame-prompt`.** Look at the photo and describe what must appear in the area to fill, e.g. for an oval portrait: "his left hand resting on his right forearm, dark suit sleeve, plain dark studio backdrop". Keep the parts that matter: "Extend this photograph to fill the whole rectangular frame", "No oval, mount, border or vignette", "Keep it a black and white photograph". `--frame-prompt` applies to the whole run, so run one photo per command when each needs its own prompt; a batch of photos that need the same generic prompt can run together.
2. **Run** with `--repair --full-frame`. Add `--frame-candidates 3` for a fill that matters (hands, faces at the edge, corners).
3. **Review** `28-frame-overlay.jpg` and `29-full-frame-final.jpg` against `00-original.*`. Also look at `28-full-frame-candidates.jpg` if there is one. Look for: deformed hands or faces, objects that don't belong, visible seams, a dark wedge or a light patch in a corner.
4. **Retry automatically, without asking the user,** if the fill is wrong. Only on the local backend: a better `--frame-prompt`, `--frame-candidates 3`, and `--frame-quality high` (for a vignette, a light line at the edge is the known side effect of high quality: prefer fast when it is the only problem). For a card on a scanner bed none of `--frame-prompt`, `--frame-candidates`, `--frame-seed` or `--frame-quality` apply: its corners come from the generative variant and there is no model fill to tune. At most **2 retry rounds per photo**. Then pick the best candidate with `--frame-seed N` (instant: it reuses the candidates already generated, only the upscale reruns).
5. **Ask the user before** anything that costs money (fal), and before a slow `--frame-quality high` run over many photos. Also ask when no candidate is acceptable after the retries: show `28-full-frame-candidates.jpg` and ask whether to keep one or to drop the full-frame version for that photo.
6. **Report** per photo: which version is recommended and why, and what was retried (prompt, candidates, quality, seed).

## Performance (RTX 5080, local)
Conservative 4–10 s, generative 15–40 s (first run loads ~30 GB of models), DDColor 5–15 s per photo.
