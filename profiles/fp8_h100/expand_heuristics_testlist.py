#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def parse_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Expand a CUTLASS heuristics testlist over runtime scheduler knobs.")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--raster-orders", default="along_n,along_m")
    parser.add_argument("--swizzle-sizes", default="1,2,4,8")
    args = parser.parse_args()

    raster_orders = parse_list(args.raster_orders)
    swizzle_sizes = parse_list(args.swizzle_sizes, int)

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    for name in ("raster_order", "swizzle_size"):
        if name not in fieldnames:
            fieldnames.append(name)

    expanded = []
    for row in rows:
        for raster_order in raster_orders:
            for swizzle_size in swizzle_sizes:
                expanded_row = dict(row)
                expanded_row["raster_order"] = raster_order
                expanded_row["swizzle_size"] = str(swizzle_size)
                expanded.append(expanded_row)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded)

    print(f"Expanded {len(rows)} heuristic rows into {len(expanded)} scheduler-sweep rows")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
