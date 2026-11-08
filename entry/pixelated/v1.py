"""v1 forecaster: LightGBM win model x payout-table-structured win-size model.

p_0 = 1 - w,  p_{1,2,5,20} = w * mix,
w   = P(win | causal features)               (binary LightGBM)
mix = projection of a 4-class LightGBM win-size forecast onto the segment
      between the machine type's two known payout tables.
"""

import json
from pathlib import Path

import numpy as np

from .features import TYPES, build


def project_mix(mix, tables, tix):
    """Closest point to `mix` on the segment table0..table1 of each row's type."""
    m0, m1 = tables[tix, 0], tables[tix, 1]
    d = m1 - m0
    pi = np.clip(((mix - m0) * d).sum(-1) / (d * d).sum(-1), 0.0, 1.0)
    return m0 + pi[:, None] * d


class V1Model:
    def __init__(self, win_booster, size_booster, tables, names, keep, mix_mode="project"):
        self.win = win_booster
        self.size = size_booster
        self.tables = np.asarray(tables, dtype=np.float64)
        self.names = list(names)
        self.keep = list(keep)
        self.mix_mode = mix_mode

    def columns(self):
        idx = {n: i for i, n in enumerate(self.names)}
        return [idx[n] for n in self.keep]

    def features(self, split):
        F, names = build(split.x, split.types, split.signals, split.short, self.tables)
        if names != self.names:
            raise ValueError("feature layout differs from the trained model")
        return F

    def query_rows(self, split, F):
        e = np.arange(len(split.episodes))
        rows = F[e, :, split.cutoff, :]  # [E, M, F]
        return rows.reshape(-1, F.shape[-1])[:, self.columns()].astype(np.float64)

    def predict_rows(self, X, tix):
        w = np.clip(self.win.predict(X), 1e-4, 1.0 - 1e-4)  # squared-error model can overshoot
        if self.size is not None:
            mix = self.size.predict(X)
            if self.mix_mode == "project":
                mix = project_mix(mix, self.tables, tix)
        else:
            mix = self.tables[tix].mean(1)
        return np.concatenate([(1.0 - w)[:, None], w[:, None] * mix], axis=1)

    def predict_split(self, split, F=None):
        F = self.features(split) if F is None else F
        X = self.query_rows(split, F)
        tix = np.array([TYPES.index(t) for t in split.types.reshape(-1)])
        probs = self.predict_rows(X, tix)
        return dict(zip(split.query_id.reshape(-1), probs.tolist()))

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.win.save_model(str(folder / "win.txt"))
        if self.size is not None:
            self.size.save_model(str(folder / "size.txt"))
        meta = {"tables": self.tables.tolist(), "names": self.names, "keep": self.keep,
                "mix_mode": self.mix_mode, "has_size": self.size is not None}
        (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder):
        import lightgbm as lgb
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        win = lgb.Booster(model_file=str(folder / "win.txt"))
        size = lgb.Booster(model_file=str(folder / "size.txt")) if meta["has_size"] else None
        return cls(win, size, meta["tables"], meta["names"], meta["keep"], meta["mix_mode"])
