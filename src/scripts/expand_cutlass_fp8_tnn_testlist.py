#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def split_csv_values(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expand CUTLASS heuristic testlist rows across scheduler knobs."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--raster-orders", default="along_n,along_m")
    parser.add_argument("--swizzle-sizes", default="1,2,4,8")
    return parser.parse_args()


def main():
    args = parse_args()
    raster_orders = split_csv_values(args.raster_orders)
    swizzle_sizes = split_csv_values(args.swizzle_sizes)

    with args.input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f"{args.input_csv} contains no test rows")

    fieldnames = list(rows[0].keys())
    for name in ("raster_order", "swizzle_size"):
        if name not in fieldnames:
            fieldnames.append(name)

    expanded = []
    seen = set()
    for row in rows:
        for raster_order in raster_orders:
            for swizzle_size in swizzle_sizes:
                out = dict(row)
                out["raster_order"] = raster_order
                out["swizzle_size"] = swizzle_size
                key = tuple(out.get(name, "") for name in fieldnames)
                if key not in seen:
                    seen.add(key)
                    expanded.append(out)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded)

    print(f"Expanded {len(rows)} heuristic rows to {len(expanded)} scheduler rows")


if __name__ == "__main__":
    main()
