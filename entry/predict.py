"""Recreate the Pixelated forecast CSV.

Organizer runner:  python predict.py --data /data --out /output/submission.csv
Local check:       python predict.py --data <kit>/public --out x.csv --split validation

Prediction only: loads the pre-trained LightGBM models in model/ (trained by
scripts/train_v1.py on train + validation histories), builds strictly causal
features from the requested split's histories and writes the probabilities.
Deterministic; no network, no randomness.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pixelated.data import write_submission  # noqa: E402
from pixelated.dataset import Split  # noqa: E402
from pixelated.v1 import V1Model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="Round's public/ folder")
    ap.add_argument("--out", required=True, type=Path, help="CSV to write")
    ap.add_argument("--split", default="test", choices=("test", "validation"))
    args = ap.parse_args(argv)

    model = V1Model.load(HERE / "model")
    split = Split(args.data / args.split)
    preds = model.predict_split(split)
    write_submission(args.out, preds)
    print(f"wrote {len(preds)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
