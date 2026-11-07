"""Load one split into dense [episode, machine, tick] arrays."""

from pathlib import Path

import numpy as np

from .data import PAYOUT_INDEX, read_history, read_queries
from .features import T


class Split:
    def __init__(self, split_dir):
        split_dir = Path(split_dir)
        self.dir = split_dir
        queries = read_queries(split_dir)
        hist = read_history(split_dir)
        by_ep = {}
        for q in queries:
            by_ep.setdefault(q["episode_id"], []).append(q)
        self.episodes = sorted(by_ep)
        E = len(self.episodes)
        M = max(len(v) for v in by_ep.values())
        if any(len(v) != M for v in by_ep.values()):
            raise ValueError("every episode must have the same number of machines")
        self.x = np.full((E, M, T), -1, dtype=np.int8)
        self.types = np.empty((E, M), dtype=object)
        self.signals = np.zeros((E, M), dtype=np.int64)
        self.cutoff = np.zeros(E, dtype=np.int64)
        self.query_id = np.empty((E, M), dtype=object)
        self.short = np.zeros(E, dtype=bool)
        for e, ep in enumerate(self.episodes):
            qs = sorted(by_ep[ep], key=lambda q: q["machine_id"])
            cut = {q["cutoff"] for q in qs}
            if len(cut) != 1:
                raise ValueError(f"episode {ep} has several cutoffs")
            self.cutoff[e] = cut.pop()
            for m, q in enumerate(qs):
                self.types[e, m] = q["machine_type"]
                self.signals[e, m] = q["signal"]
                self.query_id[e, m] = q["query_id"]
                plays = hist.get((ep, q["machine_id"]), [])
                for tick, payout in plays:
                    if tick >= self.cutoff[e]:
                        raise ValueError(f"history at/after cutoff in {ep}")
                    self.x[e, m, tick] = PAYOUT_INDEX[payout]
                if len(plays) < self.cutoff[e]:
                    self.short[e] = True
