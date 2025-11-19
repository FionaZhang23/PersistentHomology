# organize_torus_barcodes.py
from __future__ import annotations
from pathlib import Path
import re
import argparse
import shutil

# Default root (change if needed)
DEFAULT_ROOT = Path("/Users/fionazhang/PycharmProjects/PersistentHomology/graphs/torus_barcodes")

# Match: torus_a1_c3_n500_T5_sigma0.2_H1_barcode+diagram.svg
# (suffix after sigma can vary; we only care about a, c, n)
PATTERN = re.compile(
    r"^torus_a(?P<a>\d+)_c(?P<c>\d+)_n(?P<n>\d+)_T(?P<T>\d+)_sigma(?P<sigma>[\d.]+).*\.svg$"
)

VALID_N = {100, 500, 1000}

def unique_target(path: Path) -> Path:
    """If path exists, append _dupN before extension to avoid overwrite."""
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem}_dup{i}{suf}")
        if not candidate.exists():
            return candidate
        i += 1

def organize(root: Path, dry_run: bool = False) -> None:
    if not root.exists():
        print(f"[!] Root not found: {root}")
        return

    moved = 0
    skipped = 0
    for p in root.iterdir():
        if not p.is_file() or p.suffix.lower() != ".svg":
            continue
        m = PATTERN.match(p.name)
        if not m:
            print(f"[-] Skip (name pattern mismatch): {p.name}")
            skipped += 1
            continue

        n = int(m.group("n"))
        c = int(m.group("c"))
        if n not in VALID_N:
            print(f"[-] Skip n={n} (only reorganizing n in {sorted(VALID_N)}): {p.name}")
            skipped += 1
            continue
        if not (1 <= c <= 10):
            print(f"[-] Skip c={c} (expected 1..10): {p.name}")
            skipped += 1
            continue

        dest_dir = root / f"n={n}" / f"c={c}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = unique_target(dest_dir / p.name)

        print(f"[+] Move: {p.name}  ->  {dest_path.relative_to(root)}")
        if not dry_run:
            shutil.move(str(p), str(dest_path))
        moved += 1

    print(f"\nDone. Moved: {moved}, Skipped: {skipped}, Root: {root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Organize torus barcode SVGs into n=.../c=... folders.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Folder containing the SVGs.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without moving files.")
    args = ap.parse_args()
    organize(args.root, dry_run=args.dry_run)
