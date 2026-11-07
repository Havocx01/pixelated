#!/usr/bin/env bash
# Download and verify a House Edge participant kit into kit/<round>/.
#   bash scripts/fetch_kit.sh main <google-drive-file-id>
#   bash scripts/fetch_kit.sh adaptation <google-drive-file-id>
# Main-round kit id: 1WXrMzPXHv_ZuwpKvdRrcKYoJaMR5J6VV
set -euo pipefail
ROUND="${1:?round name, e.g. main or adaptation}"
ID="${2:?google drive file id}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/kit/$ROUND"
TMP="$(mktemp -d)"
curl -sSL -o "$TMP/kit.zip" "https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t"
mkdir -p "$DEST"
unzip -q -o "$TMP/kit.zip" -d "$DEST"
rm -rf "$TMP"
cd "$DEST"
py -3.13 -I -c "
import hashlib, json, pathlib
m = json.load(open('SHA256.json'))
files = m.get('files', m)
bad = [k for k, v in files.items() if pathlib.Path(k).exists() and hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest() != v]
print('kit verified' if not bad else f'HASH MISMATCH: {bad}')
"
grep -E '"round"|"max_points"|"season_id"' public/metadata.json
