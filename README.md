# photo-restore

A [Claude Code](https://claude.com/claude-code) plugin that restores old photographs: removes scratches, stains and tears, sharpens faces, upscales, and optionally colorizes. Claude looks at each photo, picks the settings, runs the pipeline, and checks the result against the original before showing it to you.

It runs **locally** on your GPU through [ComfyUI](https://github.com/Comfy-Org/ComfyUI), or **in the cloud** through [fal.ai](https://fal.ai). Both use the same model families, so results are comparable.

## What it does

For every photo you get up to four variants, each one finished and compared with the original on a single page (`restored/index.html`, with before/after sliders):

| Variant | Models | Good for |
|---|---|---|
| **Conservative** (always) | RealESRGAN x2 + [CodeFormer](https://github.com/sczhou/CodeFormer) | Faithful: sharper faces and more resolution, nothing invented. Doesn't remove damage. |
| **Generative** (`--generative`) | [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Damaged photos: tears, stains, scratches, missing corners. Can subtly alter faces, so always compare. |
| **Repaired** (`--repair`, local only) | Qwen-Image-Edit-2511 on the damaged areas only, then RealESRGAN + CodeFormer | Holes, missing corners, peeled emulsion. Everything that isn't damaged, faces included, stays the original. |
| **Full frame** (`--full-frame`, local only) | Qwen-Image-Edit-2511 on the missing parts, then RealESRGAN + CodeFormer | A rectangular picture with the mount, vignette and borders removed. On a card on a scanner bed the worn rim is cropped and the corners are taken from the generative variant. Keeps the picture's own proportions. |
| **Colorized** (`--colorize`) | [DDColor](https://github.com/piddnad/DDColor) | Natural colours, keeps the original detail (only predicts colour, never changes luminance). |

Black-and-white and sepia results keep the print's own tint. AI restorers invent colour (blue eyes, pink skin) even on monochrome photos; the pipeline keeps only their luminance and rebuilds the tint from the original.

Your originals are never modified.

## Install

```
/plugin marketplace add elboletaire/photo-restore
/plugin install photo-restore@photo-restore
```

The only hard requirement is [uv](https://docs.astral.sh/uv/getting-started/installation/). It runs the pipeline script and installs its Python dependencies automatically. Works on Linux, macOS, Windows and WSL.

Then pick a backend:

### Local (ComfyUI)
- NVIDIA GPU: 12 GB+ VRAM for the generative variant, 8 GB without it. Apple Silicon works, slowly.
- ~32 GB of disk for all models (~1 GB without the generative one).
- Already have ComfyUI? It can reuse it (portable, Desktop or a git checkout).
- Photos never leave your machine.

Ask Claude to *"set up the local photo restoration backend"*. It follows [setup-local.md](skills/restore-photos/references/setup-local.md): installs ComfyUI with [comfy-cli](https://github.com/Comfy-Org/comfy-cli) if needed, adds two custom nodes and downloads the models.

### Cloud (fal.ai)
- No GPU needed. Roughly $0.005 per photo for the conservative variant, ~$0.04 with the generative one.
- Photos are uploaded to fal.ai.

Create a key at <https://fal.ai/dashboard/keys> and export it as `FAL_KEY`. See [setup-cloud.md](skills/restore-photos/references/setup-cloud.md).

## Use

Just ask:

> Restore the photos in `~/Pictures/grandparents`, they're quite damaged. Colorize the wedding one too.

Or run the script yourself:

```bash
uv run skills/restore-photos/scripts/restore.py doctor --backend local
uv run skills/restore-photos/scripts/restore.py run ~/Pictures/old --generative --out restored
uv run skills/restore-photos/scripts/restore.py run photo.jpg --colorize --tone neutral --backend fal
uv run skills/restore-photos/scripts/restore.py run portrait.jpg --full-frame --out restored  # without the mount or vignette
uv run skills/restore-photos/scripts/restore.py run portrait.jpg --full-frame --frame-candidates 3 --frame-quality high  # pick a fill with --frame-seed
```

`run --help` lists every option (`--fidelity`, `--tone keep|neutral|sepia`, `--saturation`, `--seed`, `--prompt`, `--no-upscale`, `--force`, ...). AI stages are cached per photo and only rerun when their input or settings change, so retoning or rerunning is instant.

## Where it works

| Client | Local backend | Cloud backend |
|---|---|---|
| Claude Code (terminal, IDE, Desktop "Code" tab) on Linux, macOS, Windows | ✓ | ✓ |
| Claude Desktop / claude.ai chat (skills run in Anthropic's sandbox, not on your machine) | ✗ | only if the sandbox allows network access to fal.ai |

## Layout

```
.claude-plugin/          plugin and marketplace manifests
skills/restore-photos/
  SKILL.md               how Claude decides settings and reviews results
  scripts/restore.py     the whole pipeline (run / doctor / setup-local / sheet)
  workflows/*.json       ComfyUI API workflows. Open them in ComfyUI to tweak them, and keep the node titles.
  references/            backend setup guides
examples/                reference photos with different kinds of damage
tests/                   pytest suite (fake backend)
```

## Performance

On an RTX 5080: conservative 4–10 s, generative 15–40 s (the first run loads ~30 GB of models), colorization 5–15 s per photo.

## Development

```bash
uv run --with pytest --with numpy --with opencv-python-headless --with requests pytest
```

The tests use a fake backend, so they need neither ComfyUI nor fal. [examples/](examples) has real damaged photos to try every pipeline on.

## License

[MIT](LICENSE) for this plugin. The models are downloaded from their authors and keep their own licenses:

- **CodeFormer**: [S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/master/LICENSE), **non-commercial use only**.
- Qwen-Image-Edit-2511 and its Lightning LoRA, DDColor: Apache-2.0.
- Real-ESRGAN: BSD-3-Clause.

Restoring your own family photos is fine. For commercial work, review the model licenses first: CodeFormer in particular doesn't allow it.
