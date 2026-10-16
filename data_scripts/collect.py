"""Phase 2 - Step 1: Data collection.

Locates (or downloads) the two raw datasets and writes an evidence log to
logs/collection_log.json (source URL, file size, SHA-256, row/column counts, timestamp).

Usage:
    python data_scripts/collect.py              # verify files already placed in data/raw/
    python data_scripts/collect.py --download   # try the Kaggle CLI first (needs ~/.kaggle/kaggle.json)
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from config import DATASETS, LOG_DIR, RAW_DIR


def find_file(name):
    """Search data/raw recursively (Kaggle zips often extract into sub-folders)."""
    hits = sorted(RAW_DIR.rglob(name))
    return hits[0] if hits else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def profile_csv(path):
    """Row count (chunked, so large files are fine) and column names."""
    rows, columns = 0, None
    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
        rows += len(chunk)
    return rows, columns or []


def kaggle_download(spec):
    cmd = ["kaggle", "datasets", "download", "-d", spec["kaggle_slug"], "-p", str(RAW_DIR), "--unzip"]
    if spec["kaggle_file"]:
        cmd += ["-f", spec["kaggle_file"]]
    print("  running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("  Kaggle CLI not installed (pip install kaggle). Falling back to manual download.")
    except subprocess.CalledProcessError as exc:
        print(f"  Kaggle download failed ({exc}). Falling back to manual download.")
    # '-f' downloads come as a zip even with --unzip on some versions
    for z in RAW_DIR.glob("*.zip"):
        import zipfile
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        z.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--download", action="store_true", help="try the Kaggle CLI for missing files")
    args = parser.parse_args()

    log = {"collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": {}}
    missing = []

    for key, spec in DATASETS.items():
        print(f"[{key}] {spec['url']}")
        path = find_file(spec["file"])
        if path is None and args.download:
            kaggle_download(spec)
            path = find_file(spec["file"])
        if path is None:
            print(f"  MISSING: {spec['file']} not found under {RAW_DIR}")
            missing.append(key)
            log["datasets"][key] = {"status": "missing", "source_url": spec["url"], "expected_file": spec["file"]}
            continue

        rows, columns = profile_csv(path)
        log["datasets"][key] = {
            "status": "ok",
            "source": "Kaggle",
            "source_url": spec["url"],
            "kaggle_slug": spec["kaggle_slug"],
            "purpose": spec["purpose"],
            "file": str(path.relative_to(RAW_DIR)),
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "sha256": sha256(path),
            "rows": rows,
            "columns": columns,
        }
        print(f"  ok: {rows:,} rows, {len(columns)} columns, {path.stat().st_size / 1e6:.1f} MB")

    out = LOG_DIR / "collection_log.json"
    out.write_text(json.dumps(log, indent=2))
    print(f"\nEvidence log written to {out}")

    if missing:
        print("\nManual download steps:")
        for key in missing:
            print(f"  - {DATASETS[key]['url']}  ->  put '{DATASETS[key]['file']}' inside {RAW_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
