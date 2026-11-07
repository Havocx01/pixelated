"""Pixelated forecaster, v0.

A recency-weighted, per-machine payout frequency shrunk toward a
(machine_type, signal) prior learned from training data. This mirrors the
kit's `pooled_recent` baseline so we start from a known score; later versions
replace the internals while keeping `fit` / `predict` stable.

Uses only the Python standard library, to match the organizers' runner.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from .data import PAYOUTS, PAYOUT_INDEX

K = len(PAYOUTS)
DEFAULT_PARAMS = {
    "prior_strength": 20.0,   # pseudo-counts given to the context prior
    "context_strength": 12.0,  # pseudo-counts tying context priors to the global prior
    "half_life": 24.0,         # ticks; recency weighting of a machine's own history
}


class PixelatedModel:
    def __init__(self, params=None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.global_prior = None
        self.context_priors = {}

    @staticmethod
    def _key(q):
        return f"{q['machine_type']}|{q['signal']}"

    def fit(self, queries, history, labels=None):
        counts = [0.0] * K
        contexts = defaultdict(lambda: [0.0] * K)
        for q in queries:
            outcomes = [p for t, p in history.get((q["episode_id"], q["machine_id"]), []) if t < q["cutoff"]]
            if labels and q["query_id"] in labels:
                outcomes.append(labels[q["query_id"]])
            for o in outcomes:
                i = PAYOUT_INDEX[o]
                counts[i] += 1
                contexts[self._key(q)][i] += 1
        total = math.fsum(counts)
        self.global_prior = [(c + 0.5) / (total + 0.5 * K) for c in counts]
        s = self.params["context_strength"]
        self.context_priors = {
            key: [(c + s * g) / (math.fsum(vals) + s) for c, g in zip(vals, self.global_prior)]
            for key, vals in contexts.items()
        }
        return self

    def predict(self, queries, history):
        hl = self.params["half_life"]
        strength = self.params["prior_strength"]
        preds = {}
        for q in queries:
            prior = self.context_priors.get(self._key(q), self.global_prior)
            counts = [strength * p for p in prior]
            for tick, payout in history.get((q["episode_id"], q["machine_id"]), []):
                if tick >= q["cutoff"]:
                    continue  # never use information at or after the cutoff
                age = q["cutoff"] - 1 - tick
                counts[PAYOUT_INDEX[payout]] += 2 ** (-age / hl)
            total = math.fsum(counts)
            preds[q["query_id"]] = [c / total for c in counts]
        return preds

    def save(self, path):
        Path(path).write_text(json.dumps({
            "version": 0, "params": self.params,
            "global_prior": self.global_prior, "context_priors": self.context_priors,
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path):
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(blob["params"])
        m.global_prior = blob["global_prior"]
        m.context_priors = blob["context_priors"]
        return m
