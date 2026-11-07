# Pixelated: House Edge entry

- Team: Pixelated (solo)
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13) plus `requirements.txt`: lightgbm==4.7.0, numpy==2.5.4, scipy==1.18.1 (exact pins)

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

- What it does: loads the pre-trained models in `model/`, builds strictly causal features from `/data/test` histories (only ticks before each cutoff), and writes the five probabilities per query. Prediction only; no training, no network.
- Model files: `model/win.txt` (LightGBM win-probability model), `model/size.txt` (LightGBM win-size model), `model/meta.json` (payout tables, feature list).
- Method: `p_0 = 1 - w`, other payouts = `w x mix`, where `w` comes from a LightGBM model trained self-supervised on train + validation histories (never test). Its features include each machine's recent and long-run lag effects and recent win rates for pairs of earlier ticks. `mix` is projected onto the machine type's two payout tables.
- Randomness: none at prediction time; output is deterministic.
- Tested on: Windows 11, Python 3.13, fresh virtual environment with only `requirements.txt` installed, `OMP_NUM_THREADS=2`: about 11 seconds, identical CSV.

## Local check

```sh
python check_submission.py submission.csv --data public
```
