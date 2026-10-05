# Cloud backend setup (fal.ai)

No GPU or ComfyUI needed. The same model families as the local backend run on fal.ai:

| Stage | fal endpoint | Price (as of 2026-10) |
|---|---|---|
| Conservative | `fal-ai/codeformer` | $0.0021 / megapixel |
| Generative | `fal-ai/qwen-image-edit-2511` | $0.03 / megapixel |
| Colorize | `fal-ai/ddcolor` | $0.001 / megapixel |

Check current prices at `https://fal.ai/models/<endpoint>` before quoting them.

**Privacy**: photos are uploaded to fal.ai to be processed. Tell the user before the first run.

## Steps

1. The user creates an account at https://fal.ai, adds credit, and creates a key at https://fal.ai/dashboard/keys.
2. Make it available as `FAL_KEY`. The user should set it themselves; don't ask them to paste it into the chat.
   - Linux/macOS: add `export FAL_KEY=...` to `~/.bashrc` / `~/.zshrc`, or prefix the command.
   - Windows PowerShell: `setx FAL_KEY "..."`, then open a new terminal.
3. `uv run <skill-dir>/scripts/restore.py doctor --backend fal`, then run with `--backend fal`.

`--megapixels` has no effect on fal (the model picks its working size). `--fidelity`, `--no-upscale`, `--seed` and `--prompt` work the same.
