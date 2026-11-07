"""Build the two upload files for a round: the test CSV and the code ZIP.

    py -3.13 scripts/package.py [--round main]

1. Copies entry/ to a clean temp folder and runs it exactly like the organizer
   runner would (fresh folder, data path passed in), writing the test CSV.
2. Checks the CSV with the kit's checker.
3. Builds the code ZIP with the kit's build_entry.py.
4. Re-runs the code from the built ZIP and confirms it reproduces the CSV
   byte for byte.
Outputs land in dist/<round>/.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_entry(code_dir, data, out):
    subprocess.run([sys.executable, "predict.py", "--data", str(data), "--out", str(out)],
                   cwd=code_dir, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="main")
    args = ap.parse_args()

    kit = ROOT / "kit" / args.round
    data = kit / "public"
    dist = ROOT / "dist" / args.round
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)
    csv_path = dist / "submission.csv"
    zip_path = dist / "pixelated-entry.zip"

    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / "entry"
        shutil.copytree(ROOT / "entry", code, ignore=IGNORE)
        run_entry(code, data, csv_path)

    subprocess.run([sys.executable, "check_submission.py", str(csv_path), "--data", str(data)],
                   cwd=kit, check=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "entry"
        shutil.copytree(ROOT / "entry", src, ignore=IGNORE)
        subprocess.run([sys.executable, "build_entry.py", "--source", str(src), "--out", str(zip_path)],
                       cwd=kit, check=True)

    with tempfile.TemporaryDirectory() as tmp:
        unpacked = Path(tmp) / "unzipped"
        zipfile.ZipFile(zip_path).extractall(unpacked)
        again = Path(tmp) / "again.csv"
        run_entry(unpacked, data, again)
        same = sha(again) == sha(csv_path)

    print(f"\nCSV : {csv_path}  sha256 {sha(csv_path)}")
    print(f"ZIP : {zip_path}  ({zip_path.stat().st_size:,} bytes)")
    print("Reproduction from ZIP:", "MATCH" if same else "MISMATCH - do not upload")
    if not same:
        sys.exit(1)


if __name__ == "__main__":
    main()
