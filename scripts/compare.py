"""Paired episode bootstrap between two runs' out-of-fold validation predictions.

    uv run python scripts/compare.py runs/A runs/B [--set dev|audit|all] [--n 2000]

Resamples whole episodes within each condition (keeping all 8 machines),
recomputes the exact contest points for both runs, and reports B - A.
Also checks the run's points against the kit's own scorer.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDS = ("stable", "short_history", "drift", "shared_change")
COLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]


def load(public):
    v = public / "validation"
    q = pd.read_csv(v / "queries.csv").merge(pd.read_csv(v / "conditions.csv"), on="query_id")
    q = q.sort_values(["episode_id", "machine_id"]).reset_index(drop=True)
    truth = pd.read_csv(v / "targets.csv").set_index("query_id").loc[q.query_id, COLS].to_numpy()
    ref = pd.read_csv(v / "reference.csv").set_index("query_id").loc[q.query_id, COLS].to_numpy()
    return q, truth, ((ref - truth) ** 2).mean(1)


def audit_mask(q):
    eps = q.drop_duplicates("episode_id")
    order = np.argsort([hashlib.md5(e.encode()).hexdigest() for e in eps.episode_id])
    aud = set()
    for c in CONDS:
        idx = [i for i in order if eps.condition.iloc[i] == c][:6]
        aud |= set(eps.episode_id.iloc[idx])
    return q.episode_id.isin(aud).to_numpy()


def points(mse, base, cond, groups):
    """groups: list (per condition) of arrays of query-row indices."""
    r = [mse[g].mean() / base[g].mean() for g in groups]
    return 60 / (1 + np.mean(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--set", default="dev", choices=("dev", "audit", "all"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--round", default="main")
    args = ap.parse_args()
    public = ROOT / "kit" / args.round / "public"
    q, truth, base = load(public)
    aud = audit_mask(q)
    keep = {"dev": ~aud, "audit": aud, "all": np.ones(len(q), bool)}[args.set]
    mse = {}
    for name in (args.a, args.b):
        p = pd.read_csv(Path(name) / "oof.csv").set_index("query_id").loc[q.query_id, COLS].to_numpy()
        mse[name] = ((p - truth) ** 2).mean(1)
    # episodes per condition within the chosen set
    ep_rows = {}
    for c in CONDS:
        sub = q[keep & (q.condition == c).to_numpy()]
        ep_rows[c] = [g.index.to_numpy() for _, g in sub.groupby("episode_id")]
    full = [np.concatenate(ep_rows[c]) for c in CONDS]
    pa, pb = (points(mse[n], base, None, full) for n in (args.a, args.b))
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(args.n):
        groups = []
        for c in CONDS:
            eps = ep_rows[c]
            pick = rng.integers(0, len(eps), len(eps))
            groups.append(np.concatenate([eps[i] for i in pick]))
        diffs.append(points(mse[args.b], base, None, groups) - points(mse[args.a], base, None, groups))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"[{args.set}] A {pa:.3f}  B {pb:.3f}  B-A {pb - pa:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"P(B>A) {np.mean(diffs > 0):.3f}")
    if args.set == "all":
        sys.path.insert(0, str(ROOT / "kit" / args.round))
        from slot_forecasting.rounds import score_round
        for name in (args.a, args.b):
            print(f"  kit scorer {name}: {score_round(Path(name) / 'oof.csv', public)['technical_points']:.4f}")


if __name__ == "__main__":
    main()
