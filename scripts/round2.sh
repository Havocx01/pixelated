#!/usr/bin/env bash
# Round two (adaptation) in one command.
#   bash scripts/round2.sh <google-drive-file-id>   real run: fetch kit, refit, package dist/adaptation/
#   bash scripts/round2.sh path/to/kit.zip          same, from a kit ZIP downloaded by hand
#   bash scripts/round2.sh --dry-run                rehearsal on the main kit; touches nothing shipped
#
# Steps: fetch + verify kit -> safety net (main-round model's forecasts for this round, no
# training) -> cross-fit (a) adaptation data only and (b) adaptation + main histories -> keep
# whichever scores higher on adaptation validation -> final fit on all its rows -> write the
# model into the entry -> package CSV + ZIP and verify reproduction.
#
# Builds are packaged into a staging folder and replace the previous build only after the
# ZIP rerun prints MATCH, so a failed rerun never destroys a good dist/ folder.
# Everything is streamed to runs/logs/<prefix>-<timestamp>.log (follow it with `tail -f`).
# The script exits non-zero if any stage (training, selection, packaging) fails.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
THREADS="${THREADS:-14}"
export PYTHONUNBUFFERED=1

stamp() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ "${1:-}" = "--dry-run" ]]; then
  ROUND=main; PREFIX=dry; CANDIDATES=("own")
else
  ID="${1:?usage: round2.sh <google-drive-file-id> | <kit.zip> | --dry-run}"
  ROUND=adaptation; PREFIX=r2; CANDIDATES=("own" "plus-main")
fi

mkdir -p runs/logs
LOG="${ROOT}/runs/logs/${PREFIX}-$(date '+%Y%m%d-%H%M%S').log"
exec > >(tee -a "${LOG}") 2>&1
TEE_PID=$!
WORK=""
SAFE_WORK=""
finish() {
  local code=$?
  [[ -n "${WORK}" ]] && rm -rf "${WORK}"
  [[ -n "${SAFE_WORK}" ]] && rm -rf "${SAFE_WORK}"
  if [[ ${code} -eq 0 ]]; then stamp "round2: SUCCESS (log ${LOG})"; else stamp "round2: FAILED with exit code ${code} (log ${LOG})"; fi
  exec >&- 2>&-          # close the pipe so tee can flush and exit
  local tee_code=0
  wait "${TEE_PID}" 2>/dev/null || tee_code=$?
  if [[ ${code} -eq 0 && ${tee_code} -ne 0 ]]; then
    echo "round2: log writer (tee) failed with ${tee_code}; ${LOG} may be incomplete" >/dev/tty 2>/dev/null || true
    code=${tee_code}
  fi
  exit "${code}"
}
trap finish EXIT
stamp "round2: start (${ROUND}, prefix ${PREFIX}); log ${LOG}"

# Windows can briefly refuse a folder rename (antivirus, an open handle): retry a few times.
rename() {
  local i
  for i in 1 2 3 4 5 6; do
    mv "$1" "$2" 2>/dev/null && return 0
    sleep "${i}"
  done
  mv "$1" "$2"   # last try, showing the error
}

# package <entry> <dist>: build into <dist>.staging; replace <dist> only after MATCH.
package() {
  local entry="$1" dist="$2"
  # an earlier run stranded the last good build in .prev: put it back before anything else
  if [[ -d "${dist}.prev" && ! -d "${dist}" ]]; then rename "${dist}.prev" "${dist}" || return 1; fi
  rm -rf "${dist}.staging" "${dist}.prev" || return 1
  uv run python scripts/package.py --round "${ROUND}" --entry "${entry}" --dist "${dist}.staging" || return 1
  if [[ -d "${dist}" ]]; then rename "${dist}" "${dist}.prev" || return 1; fi
  if ! rename "${dist}.staging" "${dist}"; then
    if [[ -d "${dist}.prev" ]]; then rename "${dist}.prev" "${dist}" || true; fi   # roll back
    return 1
  fi
  rm -rf "${dist}.prev" || true
}

if [[ "${PREFIX}" = "dry" ]]; then
  WORK="$(mktemp -d)"
  ENTRY="${WORK}/entry"; cp -r entry "${ENTRY}"; rm -rf "${ENTRY}/model"
  DIST="${ROOT}/runs/dryrun-dist"; SAFE_DIST="${ROOT}/runs/dryrun-safety"
else
  if git rev-parse -q --verify refs/tags/main-final >/dev/null; then
    # rerun: entry/ must still be the main entry (a previous refit rewrites entry/model)
    dirty="$(git status --porcelain -- entry)"   # separate step so a git failure stops the script
    if ! git diff --quiet main-final -- entry || [[ -n "${dirty}" ]]; then
      echo "entry/ differs from main-final (a previous refit changed it)."
      echo "Restore the main entry with:  git checkout main-final -- entry   then rerun."
      exit 1
    fi
  else
    dirty="$(git status --porcelain -- entry)"
    if [[ -n "${dirty}" ]]; then
      echo "entry/ has uncommitted changes: commit the final main entry first"; exit 1
    fi
    git tag main-final
  fi
  tag_sha="$(git rev-parse --short main-final)"
  stamp "main entry preserved at tag main-final (${tag_sha})"
  stamp "stage fetch kit: start"
  bash scripts/fetch_kit.sh adaptation "${ID}"   # always re-download and hash-verify
  grep -q '"round": *"adaptation"' kit/adaptation/public/metadata.json \
    || { echo "kit/adaptation is not the adaptation round"; exit 1; }
  stamp "stage fetch kit: end"
  ENTRY="${ROOT}/entry"; DIST="${ROOT}/dist/adaptation"; SAFE_DIST="${ROOT}/dist/adaptation-safety"
fi

# Safety net: a missing round-two entry scores zero, so first package the already-trained
# main model's forecasts for this round (entry/ equals main-final here). No training, about a
# minute. Any failure in this stage is logged but doesn't stop the refit.
safety_net() {
  SAFE_WORK="$(mktemp -d)" || return 1
  cp -r entry "${SAFE_WORK}/entry" || return 1
  sed -i "s/^- Round: .*/- Round: ${ROUND} (main-round model, not refit)/" "${SAFE_WORK}/entry/README.md" || return 1
  package "${SAFE_WORK}/entry" "${SAFE_DIST}" || return 1
}
stamp "stage safety net: start"
if safety_net; then
  stamp "stage safety net: end; if you have no round-two upload yet, upload ${SAFE_DIST}/ now"
else
  stamp "WARNING: safety net failed; see the error above (the refit continues)"
fi
rm -rf "${SAFE_WORK}"; SAFE_WORK=""

for c in "${CANDIDATES[@]}"; do
  extra=(); [[ "${c}" = "plus-main" ]] && extra=(--extra-round main)
  stamp "stage cross-fit ${c}: start"
  uv run python scripts/train_v1.py --round "${ROUND}" --name "${PREFIX}-${c}" --threads "${THREADS}" ${extra[@]+"${extra[@]}"}
  stamp "stage cross-fit ${c}: end"
done
best="$(uv run python scripts/pick_best.py "${PREFIX}" "${CANDIDATES[@]}")"
stamp "using: ${best}"

extra=(); [[ "${best}" = "plus-main" ]] && extra=(--extra-round main)
stamp "stage final fit: start"
uv run python scripts/train_v1.py --round "${ROUND}" --name "${PREFIX}-final" --threads "${THREADS}" ${extra[@]+"${extra[@]}"} \
  --final --skip-eval --model-out "${ENTRY}/model"
stamp "stage final fit: end"
sed -i "s/^- Round: .*/- Round: ${ROUND}/" "${ENTRY}/README.md"
stamp "stage package: start"
package "${ENTRY}" "${DIST}"
stamp "stage package: end"
echo
echo "Upload through the ${ROUND} Google Form:"
echo "  ${DIST}/submission.csv"
echo "  ${DIST}/pixelated-entry.zip"
echo "(Safety net from the main-round model, if you need it: ${SAFE_DIST}/)"
