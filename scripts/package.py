"""Build and verify the round's forecast CSV and code ZIP."""

import argparse
import hashlib
import json
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


def run_entry(code_dir, data, out, main_data=None):
    """Run the exact command from code_dir/reproduce.json, mapping the runner's
    mount points (/data, /main_data, /output/submission.csv) to local paths."""
    command = json.loads((Path(code_dir) / "reproduce.json").read_text(encoding="utf-8"))["command"]
    mapping = {"/data": str(data), "/output/submission.csv": str(out)}
    if main_data is not None:
        mapping["/main_data"] = str(main_data)
    unmapped = [a for a in command if a.startswith("/") and a not in mapping]
    args = [mapping.get(a, a) for a in command]
    if args[0] == "python":
        args[0] = sys.executable
    if unmapped:
        sys.exit(f"reproduce.json uses paths this script can't map: {unmapped}")
    subprocess.run(args, cwd=code_dir, check=True)


def buildPackage(entry, kit, mainData, staging):
    data = kit / "public"
    csvPath = staging / "submission.csv"
    zipPath = staging / "pixelated-entry.zip"

    with tempfile.TemporaryDirectory() as tmp:
        codeDir = Path(tmp) / "entry"
        shutil.copytree(entry, codeDir, ignore=IGNORE)
        run_entry(codeDir, data, csvPath, mainData)

    subprocess.run([sys.executable, "check_submission.py", str(csvPath), "--data", str(data)], cwd=kit, check=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "entry"
        shutil.copytree(entry, source, ignore=IGNORE)
        subprocess.run([sys.executable, "build_entry.py", "--source", str(source), "--out", str(zipPath)], cwd=kit, check=True)

    with tempfile.TemporaryDirectory() as tmp:
        unpacked = Path(tmp) / "unzipped"
        with zipfile.ZipFile(zipPath) as archive:
            archive.extractall(unpacked)
        rerun = Path(tmp) / "rerun.csv"
        run_entry(unpacked, data, rerun, mainData)
        if sha(rerun) != sha(csvPath):
            sys.exit("Reproduction from ZIP: MISMATCH - do not upload")
    return csvPath, zipPath


def publishPackage(files, dist):
    dist.mkdir(parents=True, exist_ok=True)
    backups = {}
    for staged in files:
        target = dist / staged.name
        if target.exists():
            backups[target] = staged.with_name(f"previous-{staged.name}")
            shutil.copy2(target, backups[target])

    published = []
    try:
        for staged in files:
            target = dist / staged.name
            staged.replace(target)
            published.append(target)
    except OSError:
        for target in published:
            if target in backups:
                backups[target].replace(target)
            else:
                target.unlink()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="main")
    parser.add_argument("--entry", type=Path, default=ROOT / "entry", help="code folder to package")
    parser.add_argument("--dist", type=Path, help="output folder (default dist/<round>)")
    args = parser.parse_args()

    kit = ROOT / "kit" / args.round
    mainData = ROOT / "kit" / "main" / "public" if args.round != "main" else None
    dist = (args.dist or ROOT / "dist" / args.round).absolute()
    dist.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dist.parent) as staging:
        files = buildPackage(args.entry, kit, mainData, Path(staging))
        publishPackage(files, dist)

    csvPath = dist / "submission.csv"
    zipPath = dist / "pixelated-entry.zip"
    print(f"\nCSV : {csvPath}  sha256 {sha(csvPath)}")
    print(f"ZIP : {zipPath}  ({zipPath.stat().st_size:,} bytes)")
    print("Reproduction from ZIP: MATCH")


if __name__ == "__main__":
    main()
