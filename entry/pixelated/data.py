"""Standard-library loaders for the House Edge public data."""

import csv
from collections import defaultdict
from pathlib import Path

PAYOUTS = (0, 1, 2, 5, 20)
PAYOUT_INDEX = {p: i for i, p in enumerate(PAYOUTS)}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def read_queries(split_dir):
    """Return a list of query dicts with typed fields."""
    out = []
    for r in _rows(Path(split_dir) / "queries.csv"):
        out.append({
            "query_id": r["query_id"],
            "episode_id": r["episode_id"],
            "machine_id": r["machine_id"],
            "machine_type": r["machine_type"],
            "signal": int(r["signal"]),
            "cutoff": int(r["cutoff"]),
        })
    return out


def read_history(split_dir):
    """Return {(episode_id, machine_id): [(tick, payout), ...]} sorted by tick."""
    hist = defaultdict(list)
    for r in _rows(Path(split_dir) / "history.csv"):
        hist[(r["episode_id"], r["machine_id"])].append((int(r["tick"]), int(r["payout"])))
    for plays in hist.values():
        plays.sort()
    return dict(hist)


def read_labels(split_dir):
    """Return {query_id: observed next payout} (train and validation only)."""
    path = Path(split_dir) / "labels.csv"
    if not path.exists():
        return {}
    return {r["query_id"]: int(r["outcome"]) for r in _rows(path)}


def read_conditions(split_dir):
    """Return {query_id: condition} (train and validation only)."""
    path = Path(split_dir) / "conditions.csv"
    if not path.exists():
        return {}
    return {r["query_id"]: r["condition"] for r in _rows(path)}


def read_targets(split_dir):
    """Return {query_id: true probabilities} (validation only)."""
    path = Path(split_dir) / "targets.csv"
    if not path.exists():
        return {}
    return {r["query_id"]: tuple(float(r[f"p_{p}"]) for p in PAYOUTS) for r in _rows(path)}


def write_submission(path, predictions):
    """Write {query_id: 5 probabilities} in the required format, renormalised."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for qid in sorted(predictions):
            probs = predictions[qid]
            total = sum(probs)
            w.writerow([qid, *(format(p / total, ".17g") for p in probs)])
