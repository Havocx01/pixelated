"""Recreate the Pixelated forecast CSV.

Organizer runner:  python predict.py --data /data --out /output/submission.csv
Local validation:  python predict.py --data <kit>/public --out runs/x/validation.csv --split validation

Fits on the round's training split at run time (a few seconds), then predicts
the requested split. Standard library only; deterministic, no randomness.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pixelated.data import read_history, read_labels, read_queries, write_submission  # noqa: E402
from pixelated.model import PixelatedModel  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="Round's public/ folder")
    ap.add_argument("--out", required=True, type=Path, help="CSV to write")
    ap.add_argument("--split", default="test", choices=("test", "validation"))
    args = ap.parse_args(argv)

    train_q = read_queries(args.data / "train")
    train_h = read_history(args.data / "train")
    model = PixelatedModel().fit(train_q, train_h, read_labels(args.data / "train"))

    q = read_queries(args.data / args.split)
    h = read_history(args.data / args.split)
    write_submission(args.out, model.predict(q, h))
    print(f"wrote {len(q)} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
