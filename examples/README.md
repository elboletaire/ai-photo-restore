# Reference photos

Real family photos (public domain) used to test the pipeline. Each one covers a different kind of damage:

| Photo | Damage |
|---|---|
| `francesca-ma-osa-grau.jpg` | Heavy emulsion loss across the lower half, foxing, worn corners, photographer's imprint |
| `lluis-casajuana.jpg` | Tear across the vase, foxing, scratches, stained corner |
| `josep-casajuana.jpg` | Light: a few spots on the mount, oval vignette that must be kept |

Run them all with:

```bash
uv run skills/restore-photos/scripts/restore.py run examples --repair --colorize --out restored
```
