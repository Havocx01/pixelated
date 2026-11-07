"""Score the current entry on public validation, exactly as the kit scores it.

    py -3.13 scripts/validate.py [--name my-experiment] [--round main]

Runs entry/predict.py on the validation split, then scores it with the kit's
own `score_round` (true probabilities, per-condition loss ratios vs. the frozen
pooled reference). Appends one line to runs/log.tsv so every experiment is kept.
"""

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="main", help="kit/<round> folder to use (main or adaptation)")
    ap.add_argument("--name", default=dt.datetime.now().strftime("val-%H%M%S"))
    args = ap.parse_args()

    kit = ROOT / "kit" / args.round
    data = kit / "public"
    out = ROOT / "runs" / args.name / "validation_predictions.csv"
    subprocess.run([sys.executable, str(ROOT / "entry" / "predict.py"), "--data", str(data),
                    "--out", str(out), "--split", "validation"], check=True)

    sys.path.insert(0, str(kit))
    from slot_forecasting.rounds import round_metadata, score_round  # kit code

    result = score_round(out, data)
    max_pts = round_metadata(data)["max_points"]
    ratios = {c: r["loss_ratio"] for c, r in result["by_condition"].items()}
    print(f"\n{args.name}: {result['technical_points']:.3f} / {max_pts} validation points")
    for c, r in ratios.items():
        print(f"  {c:<14} loss ratio {r:.4f}   (1.0 = baseline, lower is better)")

    log = ROOT / "runs" / "log.tsv"
    new = not log.exists()
    with open(log, "a", encoding="utf-8") as f:
        if new:
            f.write("time\tround\tname\tpoints\t" + "\t".join(ratios) + "\n")
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}\t{args.round}\t{args.name}\t"
                f"{result['technical_points']:.3f}\t" + "\t".join(f"{r:.4f}" for r in ratios.values()) + "\n")


if __name__ == "__main__":
    main()
