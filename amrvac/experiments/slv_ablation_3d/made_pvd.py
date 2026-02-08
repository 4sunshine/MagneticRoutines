#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


DT = 5 * 85.8746159942810  # seconds = 429.373079971405
# 5 taken from the original simulation time step, 85.8746159942810 is the time step in seconds per hour, so this is 5 hours in seconds.
# from amrvac.par. dtsave_dat=5.d0
DT = 5 * 85.8746159942810 * 1 / 3600  # hours = 0.1192702999915


def natural_key(p: Path):
    """
    Sort by trailing integer (common in simulation dumps), else lexicographic.
    Examples:
      solar_bipolar_atmb0000.vtu -> 0
      solar_bipolar_atmb0123.vtu -> 123
    """
    m = re.search(r"(\d+)\.vtu$", p.name)
    if m:
        return (0, int(m.group(1)))
    return (1, p.name)


def make_pvd(folder: Path, pattern: str, out_name: str, dt: float) -> Path:
    folder = folder.resolve()
    files = sorted(folder.glob(pattern), key=natural_key)

    if not files:
        raise SystemExit(f"No files found in {folder} matching pattern: {pattern}")

    out_path = folder / out_name

    with out_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")

        for i, fp in enumerate(files):
            t = i * dt
            # Use relative paths so moving the folder still works
            rel = fp.name
            f.write(f'    <DataSet timestep="{t:.15g}" file="{rel}"/>\n')

        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate a ParaView .pvd time-series file from a folder of .vtu files.")
    ap.add_argument("folder", nargs="?", default=".", help="Folder containing .vtu files (default: current folder)")
    ap.add_argument("--pattern", default="solar_bipolar_atmb*.vtu", help="Glob pattern (default: solar_bipolar_atmb*.vtu)")
    ap.add_argument("--out", default="solar_bipolar_series.pvd", help="Output .pvd name (default: solar_bipolar_series.pvd)")
    ap.add_argument("--dt", type=float, default=DT, help=f"Delta t in seconds (default: {DT})")
    args = ap.parse_args()

    out_path = make_pvd(Path(args.folder), args.pattern, args.out, args.dt)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
