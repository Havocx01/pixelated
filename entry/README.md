# Pixelated: House Edge entry

- Team: Pixelated (solo)
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image, **Python standard library only** (no extra packages)

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

- What it does: fits the model on `/data/train` (a few seconds), then predicts `/data/test`.
- Model/config files: none beyond the source in `pixelated/`; parameters are in `pixelated/model.py`.
- Randomness: none; output is deterministic.
- Tested on: Windows 11, Python 3.13, about 10 seconds on a laptop CPU.

## Local check

```sh
python check_submission.py submission.csv --data public
```
