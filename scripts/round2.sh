#!/usr/bin/env bash
# Round two (adaptation) in one command.
#   bash scripts/round2.sh <google-drive-file-id>   real run: fetch kit, refit, package dist/adaptation/
#   bash scripts/round2.sh --dry-run                rehearsal on the main kit; touches nothing shipped
#
# Steps: fetch + verify kit -> cross-fit (a) adaptation data only and (b) adaptation + main
# histories -> keep whichever scores higher on adaptation validation -> final fit on all its
# rows -> write the model into the entry -> package CSV + ZIP and verify reproduction.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
THREADS="${THREADS:-14}"

if [ "${1:-}" = "--dry-run" ]; then
  ROUND=main; PREFIX=dry; CANDIDATES=("own")
  WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
  ENTRY="$WORK/entry"; cp -r entry "$ENTRY"; rm -rf "$ENTRY/model"
  DIST="$ROOT/runs/dryrun-dist"
else
  ID="${1:?usage: round2.sh <google-drive-file-id> | --dry-run}"
  ROUND=adaptation; PREFIX=r2; CANDIDATES=("own" "plus-main")
  if [ -n "$(git status --porcelain -- entry)" ]; then
    echo "entry/ has uncommitted changes: commit the final main entry first"; exit 1
  fi
  git rev-parse -q --verify refs/tags/main-final >/dev/null || git tag main-final
  echo ">>> main entry preserved at tag main-final ($(git rev-parse --short main-final))"
  bash scripts/fetch_kit.sh adaptation "$ID"   # always re-download and hash-verify
  grep -q '"round": *"adaptation"' kit/adaptation/public/metadata.json \
    || { echo "kit/adaptation is not the adaptation round"; exit 1; }
  ENTRY="$ROOT/entry"; DIST="$ROOT/dist/adaptation"
fi

for c in "${CANDIDATES[@]}"; do
  extra=(); [ "$c" = "plus-main" ] && extra=(--extra-round main)
  uv run python scripts/train_v1.py --round "$ROUND" --name "$PREFIX-$c" --threads "$THREADS" ${extra[@]+"${extra[@]}"}
done
best=$(uv run python scripts/pick_best.py "$PREFIX" "${CANDIDATES[@]}")
echo ">>> using: $best"

extra=(); [ "$best" = "plus-main" ] && extra=(--extra-round main)
uv run python scripts/train_v1.py --round "$ROUND" --name "$PREFIX-final" --threads "$THREADS" ${extra[@]+"${extra[@]}"} \
  --final --skip-eval --model-out "$ENTRY/model"
sed -i "s/^- Round: .*/- Round: $ROUND/" "$ENTRY/README.md"
uv run python scripts/package.py --round "$ROUND" --entry "$ENTRY" --dist "$DIST"
echo
echo "Upload through the $ROUND Google Form:"
echo "  $DIST/submission.csv"
echo "  $DIST/pixelated-entry.zip"
