#!/usr/bin/env bash
# Download and verify a House Edge participant kit into kit/<round>/.
#   bash scripts/fetch_kit.sh main <google-drive-file-id>
#   bash scripts/fetch_kit.sh adaptation <google-drive-file-id>
#   bash scripts/fetch_kit.sh adaptation path/to/downloaded-kit.zip   (if the download is blocked)
# Main-round kit id: 1WXrMzPXHv_ZuwpKvdRrcKYoJaMR5J6VV
# Extracts into a fresh staging folder, requires every file in SHA256.json to
# exist and match, and only then replaces kit/<round>/. Exits nonzero on failure.
set -euo pipefail
ROUND="${1:?round name, e.g. main or adaptation}"
ID="${2:?google drive file id}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/kit/$ROUND"

if [ -n "${PYTHON:-}" ]; then PY=("$PYTHON")
elif command -v py >/dev/null 2>&1; then PY=(py -3.13)
elif command -v python3 >/dev/null 2>&1; then PY=(python3)
else PY=(python); fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if [ -f "$ID" ]; then   # a kit ZIP downloaded by hand (e.g. the Drive link blocked curl)
  cp "$ID" "$TMP/kit.zip"
else
  curl -fsSL -o "$TMP/kit.zip" "https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t"
fi
mkdir "$TMP/stage"
unzip -q "$TMP/kit.zip" -d "$TMP/stage"

"${PY[@]}" -I - "$TMP/stage" <<'EOF'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "SHA256.json").read_text(encoding="utf-8"))
files = manifest.get("files", manifest)
problems = []
for name, digest in files.items():
    p = root / name
    if not p.is_file():
        problems.append(f"missing {name}")
    elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        problems.append(f"hash mismatch {name}")
if problems:
    sys.exit("KIT VERIFICATION FAILED:\n  " + "\n  ".join(problems))
print(f"kit verified: {len(files)} files")
EOF

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$TMP/stage" "$DEST"
grep -E '"round"|"max_points"|"season_id"' "$DEST/public/metadata.json"
