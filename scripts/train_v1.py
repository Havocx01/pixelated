"""Train and honestly evaluate the v1 forecaster.

    uv run python scripts/train_v1.py --name v1 [--round main] [--final]
    uv run python scripts/train_v1.py --name v1-nopeer --drop peer_effx peer_sum

Evaluation: validation episodes are split into 4 condition-stratified folds.
Each fold is predicted by a model trained on train histories plus the OTHER
folds' validation histories (never the evaluated episodes, never test).
Scores are reported on all 96 episodes, on the 72 "dev" episodes used for
choosing between variants, and on 24 "audit" episodes kept for the final check.

--final also fits on train + all validation histories and saves the model
into entry/model/ for the submission.
"""

import argparse
import datetime as dt
import hashlib
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "entry"))

from pixelated.dataset import Split  # noqa: E402
from pixelated.features import PAYOUTS, TYPES, build  # noqa: E402
from pixelated.v1 import V1Model  # noqa: E402

COLS = [f"p_{p}" for p in PAYOUTS]
CONDS = ("stable", "short_history", "drift", "shared_change")


def payout_tables(public):
    """The two win-size mixes per machine type, read off validation truth."""
    v = public / "validation"
    d = pd.read_csv(v / "queries.csv").merge(pd.read_csv(v / "targets.csv"), on="query_id")
    w = 1 - d.p_0
    mix = d[["p_1", "p_2", "p_5", "p_20"]].to_numpy() / w.to_numpy()[:, None]
    tables = np.zeros((3, 2, 4))
    for ti, ty in enumerate(TYPES):
        m = mix[(d.machine_type == ty).to_numpy() & (w.to_numpy() > 1e-6)]
        keys, inv, cnt = np.unique(m.round(4), axis=0, return_inverse=True, return_counts=True)
        top = np.argsort(-cnt)[:2]
        if len(top) < 2 or cnt[top].sum() < 0.98 * len(m):
            raise SystemExit(f"payout tables for {ty} are not two clean clusters: {cnt}")
        found = [m[inv.reshape(-1) == k].mean(0) for k in sorted(top)]
        tables[ti] = sorted(found, key=lambda r: -r[0])  # table 0 = more 1x-heavy
    return tables


def features(split, tables, cache):
    h = hashlib.sha1(tables.tobytes())
    for f in ("entry/pixelated/features.py", "entry/pixelated/dataset.py"):
        h.update((ROOT / f).read_bytes())
    for f in sorted(Path(split.dir).glob("*.csv")):  # data changes also invalidate the cache
        h.update(f.name.encode() + f.read_bytes())
    key = h.hexdigest()[:10]
    path = cache.with_name(cache.name + f"-{key}.npz")
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["F"], list(z["names"])
    F, names = build(split.x, split.types, split.signals, split.short, tables)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, F=F, names=np.array(names, dtype=object))
    return F, names


def conditions(split, public, name):
    c = pd.read_csv(public / name / "conditions.csv").set_index("query_id").condition
    return np.array([c[split.query_id[e, 0]] for e in range(len(split.episodes))])


def rows(split, F, ep_mask, cond, min_t=24):
    """Self-supervised rows: every observed tick t in [min_t, cutoff) of selected episodes."""
    X, ywin, ysize, cw, tix = [], [], [], [], []
    for e in np.flatnonzero(ep_mask):
        for m in range(split.x.shape[1]):
            ts = np.flatnonzero(split.x[e, m, : split.cutoff[e]] >= 0)
            ts = ts[ts >= min_t]
            X.append(F[e, m, ts])
            v = split.x[e, m, ts].astype(int)
            ywin.append(v > 0)
            ysize.append(v - 1)
            cw.append(np.repeat(cond[e], len(ts)))
            tix.append(np.full(len(ts), TYPES.index(split.types[e, m])))
    return (np.concatenate(X), np.concatenate(ywin), np.concatenate(ysize),
            np.concatenate(cw), np.concatenate(tix))


def score(probs, truth, cond, base_mse):
    """Contest points from per-query 5-vector MSE, per-condition ratios to the frozen baseline."""
    mse = ((probs - truth) ** 2).mean(1)
    ratios = {c: mse[cond == c].mean() / base_mse[c] for c in CONDS}
    return 60.0 / (1.0 + np.mean(list(ratios.values()))), ratios, mse


def fit_models(X, ywin, ysize, weight, args, seed=7):
    common = dict(learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.min_leaf,
                  feature_fraction=args.colsample, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, seed=seed, deterministic=True, force_row_wise=True,
                  num_threads=args.threads, verbose=-1)
    obj = {"binary": dict(objective="binary"), "brier": dict(objective="regression")}[args.loss]
    win = lgb.train({**common, **obj}, lgb.Dataset(X, ywin.astype(float), weight=weight),
                    num_boost_round=args.trees)
    size = None
    if args.mix == "gbm":
        sel = ywin
        size = lgb.train({**common, "objective": "multiclass", "num_class": 4},
                         lgb.Dataset(X[sel], ysize[sel], weight=weight[sel]),
                         num_boost_round=args.size_trees)
    return win, size


def row_weight(cond, tix, tables, base_mse, args):
    w = np.ones(len(cond))
    if args.mix_weight:
        w *= 1.0 + (tables[tix] ** 2).sum(-1).mean(-1)
    if args.cond_weight:
        w *= np.array([1.0 / base_mse[c] for c in cond]) * np.mean(list(base_mse.values()))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="main")
    ap.add_argument("--name", default=dt.datetime.now().strftime("v1-%H%M%S"))
    ap.add_argument("--drop", nargs="*", default=[], help="feature-name prefixes to leave out")
    ap.add_argument("--loss", choices=("binary", "brier"), default="binary")
    ap.add_argument("--mix", choices=("gbm", "flat"), default="gbm")
    ap.add_argument("--mix-mode", choices=("project", "raw"), default="project")
    ap.add_argument("--no-mix-weight", dest="mix_weight", action="store_false")
    ap.add_argument("--no-cond-weight", dest="cond_weight", action="store_false")
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--size-trees", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--colsample", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    public = ROOT / "kit" / args.round / "public"
    cache = ROOT / "runs" / "cache" / args.round
    tables = payout_tables(public)
    tr, va = Split(public / "train"), Split(public / "validation")
    Ftr, names = features(tr, tables, cache / "train")
    Fva, _ = features(va, tables, cache / "validation")
    keep = [n for n in names if not any(n.startswith(p) for p in args.drop)]
    cols = [names.index(n) for n in keep]
    ctr, cva = conditions(tr, public, "train"), conditions(va, public, "validation")

    # truth and frozen-baseline MSE per condition
    tgt = pd.read_csv(public / "validation/targets.csv").set_index("query_id")[COLS]
    ref = pd.read_csv(public / "validation/reference.csv").set_index("query_id")[COLS]
    qids = va.query_id.reshape(-1)
    truth = tgt.loc[qids].to_numpy()
    qcond = np.repeat(cva, va.x.shape[1])
    base_mse = {c: ((ref.loc[qids].to_numpy() - truth) ** 2).mean(1)[qcond == c].mean() for c in CONDS}

    # 4 condition-stratified folds; 6 audit episodes per condition
    E = len(va.episodes)
    fold, audit = np.zeros(E, int), np.zeros(E, bool)
    for c in CONDS:
        idx = [e for e in np.argsort([hashlib.md5(x.encode()).hexdigest() for x in va.episodes]) if cva[e] == c]
        for r, e in enumerate(idx):
            fold[e] = r % 4
            audit[e] = r < 6

    Xtr, wtr, str_, ctr_r, ttr = rows(tr, Ftr, np.ones(len(tr.episodes), bool), ctr)
    oof = np.zeros((E * va.x.shape[1], 5))
    tix_q = np.array([TYPES.index(t) for t in va.types.reshape(-1)])
    for f in range(4):
        Xv, wv, sv, cv, tv = rows(va, Fva, fold != f, cva)
        X = np.concatenate([Xtr, Xv])[:, cols]
        cond = np.concatenate([ctr_r, cv])
        tix = np.concatenate([ttr, tv])
        weight = row_weight(cond, tix, tables, base_mse, args)
        win, size = fit_models(X, np.concatenate([wtr, wv]), np.concatenate([str_, sv]), weight, args)
        model = V1Model(win, size, tables, names, keep, args.mix_mode)
        Xq = model.query_rows(va, Fva)
        sel = np.repeat(fold == f, va.x.shape[1])
        oof[sel] = model.predict_rows(Xq[sel], tix_q[sel])
        print(f"  fold {f}: {len(X):,} rows, {time.time() - t0:.0f}s")

    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(oof, columns=COLS, index=pd.Index(qids, name="query_id")).to_csv(out / "oof.csv")
    np.save(out / "oof.npy", oof)
    qaudit = np.repeat(audit, va.x.shape[1])
    ref_mse = ((ref.loc[qids].to_numpy() - truth) ** 2).mean(1)

    def subset_base(mask):  # normalise each subset by the baseline's error on that same subset
        return {c: ref_mse[mask & (qcond == c)].mean() for c in CONDS}

    pts, ratios, _ = score(oof, truth, qcond, base_mse)
    dev_pts, dev_r, _ = score(oof[~qaudit], truth[~qaudit], qcond[~qaudit], subset_base(~qaudit))
    aud_pts, _, _ = score(oof[qaudit], truth[qaudit], qcond[qaudit], subset_base(qaudit))
    print(f"\n{args.name}: all {pts:.3f} | dev {dev_pts:.3f} | audit {aud_pts:.3f}   ({time.time() - t0:.0f}s)")
    print("  ratios " + "  ".join(f"{c} {ratios[c]:.3f}" for c in CONDS))
    log = ROOT / "runs" / "v1_log.tsv"
    new = not log.exists()
    with open(log, "a", encoding="utf-8") as fh:
        if new:
            fh.write("time\tname\tall\tdev\taudit\t" + "\t".join(CONDS) + "\targs\n")
        fh.write(f"{dt.datetime.now():%H:%M}\t{args.name}\t{pts:.3f}\t{dev_pts:.3f}\t{aud_pts:.3f}\t"
                 + "\t".join(f"{ratios[c]:.4f}" for c in CONDS) + f"\t{' '.join(sys.argv[1:])}\n")

    if args.final:
        Xv, wv, sv, cv, tv = rows(va, Fva, np.ones(E, bool), cva)
        X = np.concatenate([Xtr, Xv])[:, cols]
        cond, tix = np.concatenate([ctr_r, cv]), np.concatenate([ttr, tv])
        win, size = fit_models(X, np.concatenate([wtr, wv]), np.concatenate([str_, sv]),
                               row_weight(cond, tix, tables, base_mse, args), args)
        V1Model(win, size, tables, names, keep, args.mix_mode).save(ROOT / "entry" / "model")
        print("saved final model to entry/model/")


if __name__ == "__main__":
    main()
