#!/usr/bin/env python3
"""Zip src/molgen for upload as a Kaggle Dataset.

    python tools/pack_for_kaggle.py

Produces ``dist/molgen-src.zip`` containing ``src/molgen/``.  Upload it as a
Kaggle Dataset (suggested slug: ``molgen-src``) and attach it to the notebooks —
their bootstrap cell searches ``/kaggle/input`` for a directory containing
``molgen/__init__.py`` and puts it on ``sys.path``.

Re-upload a new version after changing the library; Kaggle keeps dataset
versions, so pin the version in a notebook if you need reproducibility.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "molgen"
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "dist" / "molgen-src.zip"))
    args = ap.parse_args()

    if not (SRC / "__init__.py").is_file():
        print(f"error: {SRC} is not a package")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in sorted(SRC.rglob("*"))
        if p.is_file()
        and p.suffix not in SKIP_SUFFIX
        and not SKIP_DIRS & set(p.relative_to(SRC).parts)
    ]

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            # arcname keeps the src/molgen/... prefix the bootstrap cell expects
            zf.write(p, Path("src") / "molgen" / p.relative_to(SRC))

    total = sum(p.stat().st_size for p in files)
    print(f"{out}  ({len(files)} files, {total / 1e3:.0f} kB -> {out.stat().st_size / 1e3:.0f} kB)")
    print("\nnext: upload as a Kaggle Dataset named 'molgen-src' and attach it to the notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
