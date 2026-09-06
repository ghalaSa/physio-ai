"""
Phase 1 dataset inspector for MobiPhysio.

Purpose: report what the dataset ACTUALLY contains. It makes no assumptions
about exercise names, participant-ID encoding, or score ranges -- everything
printed here is read off disk. Nothing is inferred or filled in.

Usage:
    python scripts/inspect_dataset.py data/raw
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
TABLE_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".txt", ".xml", ".yaml", ".yml"}


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024:
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} PB"


def walk(root: Path):
    """Collect every file once, so we scan the tree a single time."""
    videos, tables, others = [], [], []
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXT:
            videos.append(p)
        elif ext in TABLE_EXT:
            tables.append(p)
        else:
            others.append(p)
    return videos, tables, others


def report_tree(root: Path, videos, tables, others) -> None:
    print("=" * 70)
    print(f"ROOT: {root}")
    print("=" * 70)
    total = sum(p.stat().st_size for p in videos + tables + others)
    print(f"videos: {len(videos)} | tabular/metadata: {len(tables)} | other: {len(others)}")
    print(f"total size on disk: {human(total)}")
    if videos:
        vsize = sum(p.stat().st_size for p in videos)
        print(f"video bytes: {human(vsize)}  (mean {human(vsize // max(len(videos), 1))} per clip)")

    # Directory layout: depth profile tells us how participants/exercises are nested.
    print("\n--- directory layout (relative depth -> example paths) ---")
    by_depth: dict[int, list[Path]] = {}
    for p in videos[:4000]:
        d = len(p.relative_to(root).parts) - 1
        by_depth.setdefault(d, []).append(p)
    for depth in sorted(by_depth):
        ex = by_depth[depth][:3]
        print(f"depth {depth}: {len(by_depth[depth])} videos")
        for e in ex:
            print(f"    {e.relative_to(root)}")

    print("\n--- top-level entries ---")
    for child in sorted(root.iterdir())[:40]:
        kind = "dir " if child.is_dir() else "file"
        n = sum(1 for _ in child.rglob("*")) if child.is_dir() else ""
        print(f"  [{kind}] {child.name} {f'({n} entries)' if n != '' else ''}")


def report_filenames(root: Path, videos) -> None:
    """Filename tokens are how participant/exercise/view IDs are usually encoded."""
    if not videos:
        return
    print("\n--- filename samples (for decoding participant / exercise / view IDs) ---")
    for p in videos[:15]:
        print(f"  {p.relative_to(root)}")

    seps = Counter()
    for p in videos:
        for s in ("_", "-", ".", " "):
            seps[s] += p.stem.count(s)
    print(f"\nseparator frequency across {len(videos)} filenames: {dict(seps)}")

    # Token-position analysis: a position with few distinct values is likely a
    # categorical field (exercise, camera view); many values -> participant/take.
    parts_by_pos: dict[int, Counter] = {}
    for p in videos:
        for i, tok in enumerate(p.stem.split("_")):
            parts_by_pos.setdefault(i, Counter())[tok] += 1
    print("\n'_'-token positions (distinct values -> likely field type):")
    for i in sorted(parts_by_pos):
        c = parts_by_pos[i]
        sample = list(c)[:8]
        print(f"  pos {i}: {len(c)} distinct | sample {sample}")


def report_tables(root: Path, tables) -> None:
    if not tables:
        print("\n!! No metadata/annotation files found. Labels and quality scores")
        print("   must therefore come from the directory/filename structure above.")
        return
    print("\n--- metadata / annotation files ---")
    for p in tables:
        print(f"\n### {p.relative_to(root)}  ({human(p.stat().st_size)})")
        ext = p.suffix.lower()
        try:
            if ext in {".csv", ".tsv", ".txt"}:
                import pandas as pd
                sep = "\t" if ext == ".tsv" else None
                df = pd.read_csv(p, sep=sep, engine="python", nrows=5000)
                print(f"shape (first 5000 rows): {df.shape}")
                print(f"columns: {list(df.columns)}")
                print(df.head(5).to_string())
                print("\ndtypes / nulls:")
                print(df.dtypes.to_string())
                # Numeric columns: report the true observed range. This is how we
                # confirm (not assume) the movement-quality score scale.
                num = df.select_dtypes("number")
                if not num.empty:
                    print("\nnumeric column ranges (ACTUAL observed values):")
                    print(num.describe().T[["count", "mean", "min", "max"]].to_string())
                for col in df.columns:
                    nu = df[col].nunique(dropna=True)
                    if nu <= 20:
                        print(f"\n{col!r}: {nu} distinct -> {sorted(map(str, df[col].dropna().unique()))}")
            elif ext in {".xlsx", ".xls"}:
                import pandas as pd
                xl = pd.ExcelFile(p)
                print(f"sheets: {xl.sheet_names}")
                for s in xl.sheet_names[:5]:
                    d = xl.parse(s, nrows=200)
                    print(f"  [{s}] shape={d.shape} columns={list(d.columns)}")
            elif ext == ".json":
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, list):
                    print(f"JSON list, {len(obj)} items; first item keys: "
                          f"{list(obj[0].keys()) if obj and isinstance(obj[0], dict) else type(obj[0])}")
                    print(json.dumps(obj[0], indent=2, ensure_ascii=False)[:1500])
                elif isinstance(obj, dict):
                    print(f"JSON object, top-level keys: {list(obj)[:40]}")
                    print(json.dumps(obj, indent=2, ensure_ascii=False)[:1500])
            else:
                print(p.read_text(encoding="utf-8", errors="replace")[:1200])
        except Exception as exc:  # noqa: BLE001 - inspection must never abort the report
            print(f"  !! could not parse: {type(exc).__name__}: {exc}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw").expanduser().resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist.")
        return 1

    videos, tables, others = walk(root)
    if not videos and not tables and not others:
        print(f"'{root}' is empty -- place the MobiPhysio dataset here, then re-run.")
        return 2

    report_tree(root, videos, tables, others)
    report_filenames(root, videos)
    report_tables(root, tables)

    print("\n" + "=" * 70)
    print("Inspection complete. Every value above was read from disk.")
    print("Unresolved items (exercise names, participant IDs, score scale) are")
    print("resolved ONLY from this output -- never assumed.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
