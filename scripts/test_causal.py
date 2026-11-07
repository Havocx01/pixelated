"""Leak test: features at tick t must not change when ticks >= t are erased.

    uv run python scripts/test_causal.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "entry"))
sys.path.insert(0, str(ROOT / "scripts"))
from pixelated.dataset import Split  # noqa: E402
from pixelated.features import build  # noqa: E402
from train_v1 import payout_tables  # noqa: E402

public = ROOT / "kit" / "main" / "public"
tables = payout_tables(public)
va = Split(public / "validation")
sel = slice(0, 6)
x, ty, sg, sh = va.x[sel], va.types[sel], va.signals[sel], va.short[sel]
F, _ = build(x, ty, sg, sh, tables)
rng = np.random.default_rng(1)
worst = 0.0
for t in rng.integers(30, 190, 8):
    xt = x.copy()
    xt[..., t:] = -1
    Ft, _ = build(xt, ty, sg, sh, tables)
    a, b = F[:, :, t], Ft[:, :, t]
    diff = np.nanmax(np.abs(np.nan_to_num(a, nan=-9) - np.nan_to_num(b, nan=-9)))
    worst = max(worst, diff)
    print(f"t={t}: max |feature change| = {diff:.2e}")
print("PASS" if worst < 1e-6 else "FAIL: features use the future")
sys.exit(0 if worst < 1e-6 else 1)
