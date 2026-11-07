# Pixelated

Solo entry for **House Edge** (slot-machine forecasting) at TAMU Datathon 2026.
For each query, predict the next spin's probabilities of paying 0x, 1x, 2x, 5x or 20x.

## Key facts (from the kit's RULES.md)

| | |
|---|---|
| Main round | **60 pts**, closes **Sat Nov 7, 9 PM Central** |
| Round two (adaptation) | **40 pts**, new kit Sat 9 PM, closes **Sun Nov 8, 11 AM Central** |
| Score | `points = max / (1 + R)`, R = mean over 4 conditions of our error / pooled baseline error |
| Baseline | pooled starter scores 30 + 20 = 50; perfect odds score 100 |
| Conditions | stable, short_history, drift, shared_change (equal weight) |
| Runner | `python` image (3.13, stdlib), 2 CPU, 16 GiB, 30 min, no network |
| Upload | test CSV + code ZIP through each round's Google Form; latest response before the deadline counts |

## Layout

```
entry/              what gets zipped and rerun by organizers (stdlib only)
  predict.py        python predict.py --data /data --out /output/submission.csv
  reproduce.json
  pixelated/        data loading + model
scripts/
  validate.py       score the entry on public validation (true probabilities)
  package.py        build dist/<round>/submission.csv + code ZIP and verify reproduction
  fetch_kit.sh      download + hash-verify a kit into kit/<round>/
kit/<round>/        organizer kit (git-ignored)
runs/               experiment outputs; runs/log.tsv keeps every validation score
pyproject.toml      exploration env (numpy, pandas, scipy, matplotlib) via uv
```

## Setup

```sh
bash scripts/fetch_kit.sh main 1WXrMzPXHv_ZuwpKvdRrcKYoJaMR5J6VV   # kit/main/
uv sync                                                          # exploration env, optional
```

## Workflow

```sh
py -3.13 scripts/validate.py --name <experiment>   # score on validation, logs to runs/log.tsv
py -3.13 scripts/package.py                        # build upload files in dist/main/
```

Upload `dist/main/submission.csv` and `dist/main/pixelated-entry.zip` to the round's Google Form.

Round two: `bash scripts/fetch_kit.sh adaptation <id>`, then the same commands with `--round adaptation`.

## Rules to keep in mind

- Use only ticks strictly before each query's cutoff; missing ticks are unknown, not zero.
- Other machines' earlier results may be used only within the same episode.
- Condition labels exist for train/validation only; the test predictor must not need them.
- Keep whole episodes together when splitting validation.

## Results

| Version | Validation points (/60) | stable | short_history | drift | shared_change |
|---|---|---|---|---|---|
| kit pooled baseline | 30.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| v0 pooled_recent | 30.826 | 0.958 | 1.001 | 0.946 | 0.881 |

## Round two runbook (Sat 9 PM -> Sun 11 AM)

1. Get the adaptation kit's Google Drive file ID from the organizers' announcement.
2. Run, from the repo root:

   ```bash
   bash scripts/round2.sh <google-drive-file-id>
   ```

   It tags the main entry as `main-final`, downloads and hash-checks the kit, cross-fits
   "adaptation data only" vs "adaptation + main histories", keeps the better one, fits the
   final model, writes it into `entry/`, and packages `dist/adaptation/` (with a ZIP rerun
   that must print `MATCH`). About 25 minutes on this laptop.
3. Upload `dist/adaptation/submission.csv` and `dist/adaptation/pixelated-entry.zip` through
   the **Adaptation** Google Form.
4. Rehearsal without touching anything shipped: `bash scripts/round2.sh --dry-run`.
