"""Print the candidate with the highest finite cross-fit score (used by round2.sh).

    python scripts/pick_best.py <prefix> <candidate> [<candidate> ...]
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
prefix, cands = sys.argv[1], sys.argv[2:]
scores = {}
for c in cands:
    s = json.loads((ROOT / "runs" / f"{prefix}-{c}" / "score.json").read_text())["all"]
    print(f">>> {c}: {s:.3f} validation points (cross-fit)", file=sys.stderr)
    if math.isfinite(s):
        scores[c] = s
if not scores:
    sys.exit("no candidate produced a finite score")
print(max(scores, key=scores.get))
