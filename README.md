# Pixelated

Given a slot machine's patchy spin history, Pixelated predicts the probability that the next spin pays 0x, 1x, 2x, 5x or 20x.

## How it works

The forecast is split into two questions: will the spin win, and if it wins, how big?

- **Win probability:** a LightGBM model using recent and long-run patterns in each machine's history.
- **Hidden-rule filter:** tracks 448 candidate rules based on pairs of earlier spins and passes its forecast and confidence to the win model.
- **Payout size:** a second LightGBM model, with its output matched to the machine type's payout tables.

Predictions use only spins before each query's cutoff.

## Results

Scored out of 60 on the public validation set:

| Version | Points |
|---|---|
| Pooled baseline | 30.00 |
| v0, pooled recent average | 30.83 |
| v1.1, LightGBM | 53.97 |
| **c11, v1.1 + hidden-rule filter (final)** | **55.59** |

These episodes were reused during development, so the scores are likely a little optimistic. The filter's improvement also held up on training episodes kept out of fitting it.

## Final submission

The submitted c11 files are in [`released/`](released/):

- [`submission.csv`](released/submission.csv): the forecast
- [`pixelated-entry.zip`](released/pixelated-entry.zip): the code and trained model that reproduce it

> The `entry/` folder contains the earlier v1.1 model. Use the ZIP in `released/` for c11.

## Reproduce the forecast

You'll need Python 3.13 and the House Edge competition data, which isn't included in this repo. Unzip `released/pixelated-entry.zip`, then run this from the unzipped folder:

```sh
pip install -r requirements.txt
python predict.py --data /path/to/data --out submission.csv
```

Prediction only: no training or network access is needed.

## Repository layout

```
released/   final c11 forecast and code ZIP
entry/      earlier v1.1 predictor
scripts/    validation and packaging tools
```

## Built with

Python, LightGBM, NumPy, SciPy and Docker.
