#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


SUMMARY_FIELDS = [
    "m",
    "n",
    "k",
    "batch_count",
    "beta",
    "runtime_ms",
    "tflops",
    "gflops",
    "operation",
    "tile_shape",
    "cluster_shape",
    "fallback_cluster_shape",
    "kernel_schedule",
    "epilogue_schedule",
    "raster_order",
    "swizzle_size",
    "split_k_slices",
    "stages",
    "A",
    "B",
    "C",
    "D",
    "accumulator_type",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract fastest per-shape rows from CUTLASS FP8 TNN profiler CSV."
    )
    parser.add_argument("profile_csv", type=Path)
    parser.add_argument("winners_csv", type=Path)
    parser.add_argument("winners_md", type=Path)
    return parser.parse_args()


def clean_lines(path):
    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)
    return lines


def as_float(row, *names, default=0.0):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                pass
    return default


def as_int_text(row, *names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return str(int(float(value)))
            except ValueError:
                return str(value)
    return ""


def text(row, *names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def infer_shape(operation, pattern):
    match = re.search(pattern, operation)
    if match:
        return "x".join(match.groups())
    return ""


def infer_schedule(operation):
    kernel = ""
    epilogue = ""
    lower = operation.lower()
    if "warpspecialized" in lower or "warp_specialized" in lower:
        kernel = "KernelTmaWarpSpecialized"
        epilogue = "TmaWarpSpecialized"
    if "pingpong" in lower:
        kernel += "Pingpong" if kernel else "KernelTmaWarpSpecializedPingpong"
    if "cooperative" in lower:
        kernel += "Cooperative" if kernel else "KernelTmaWarpSpecializedCooperative"
    return kernel, epilogue


def shape_text(row, *names):
    return "x".join(as_int_text(row, name) for name in names).strip("x")


def row_sort_key(key):
    m, n, k, beta, batch_count = key
    return int(m or 0), int(n or 0), int(k or 0), float(beta or 0), int(batch_count or 0)


def summarize_row(key, row):
    operation = text(row, "Operation")
    kernel_schedule = text(row, "kernel_schedule", "KernelSchedule")
    epilogue_schedule = text(row, "epilogue_schedule", "EpilogueSchedule")
    if not kernel_schedule or not epilogue_schedule:
        inferred_kernel, inferred_epilogue = infer_schedule(operation)
        kernel_schedule = kernel_schedule or inferred_kernel
        epilogue_schedule = epilogue_schedule or inferred_epilogue

    tile_shape = text(row, "tile_shape", "cta_tile", "threadblock_shape")
    if not tile_shape:
        tile_shape = infer_shape(operation, r"(?:^|_)(\d+)x(\d+)x(\d+)(?:_|$)")

    gflops = as_float(row, "GFLOPs", "Gflops", "GFLOPS")
    return {
        "m": key[0],
        "n": key[1],
        "k": key[2],
        "batch_count": key[4],
        "beta": key[3],
        "runtime_ms": f"{as_float(row, 'Runtime'):.6g}",
        "tflops": f"{gflops / 1000.0:.6g}",
        "gflops": f"{gflops:.6g}",
        "operation": operation,
        "tile_shape": tile_shape,
        "cluster_shape": shape_text(row, "cluster_m", "cluster_n", "cluster_k"),
        "fallback_cluster_shape": shape_text(
            row, "cluster_m_fallback", "cluster_n_fallback", "cluster_k_fallback"
        ),
        "kernel_schedule": kernel_schedule,
        "epilogue_schedule": epilogue_schedule,
        "raster_order": text(row, "raster_order"),
        "swizzle_size": text(row, "swizzle_size"),
        "split_k_slices": text(row, "split_k_slices"),
        "stages": text(row, "stages", "StageCount"),
        "A": text(row, "A"),
        "B": text(row, "B"),
        "C": text(row, "C"),
        "D": text(row, "D"),
        "accumulator_type": text(row, "accumulator_type", "accumulator-type"),
    }


def read_winners(profile_csv):
    rows = list(csv.DictReader(clean_lines(profile_csv)))
    if not rows:
        raise SystemExit(f"{profile_csv} has no parseable CSV rows")

    valid_dispositions = {"passed", "not_verified", "notverified"}
    groups = {}
    for row in rows:
        disposition = text(row, "Disposition").lower().replace(" ", "_")
        status = text(row, "Status").lower()
        runtime = as_float(row, "Runtime", default=0.0)
        gflops = as_float(row, "GFLOPs", "Gflops", "GFLOPS", default=0.0)
        if disposition and disposition not in valid_dispositions:
            continue
        if status and status != "success":
            continue
        if runtime <= 0.0 or gflops <= 0.0:
            continue

        key = (
            as_int_text(row, "m", "M"),
            as_int_text(row, "n", "N"),
            as_int_text(row, "k", "K"),
            text(row, "beta"),
            text(row, "batch_count"),
        )
        if key not in groups or gflops > as_float(groups[key], "GFLOPs", "Gflops", "GFLOPS"):
            groups[key] = row

    summary_rows = [summarize_row(key, groups[key]) for key in sorted(groups, key=row_sort_key)]
    if not summary_rows:
        raise SystemExit("no successful profiler rows found")
    return summary_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, profile_csv):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# CUTLASS FP8 TNN Winners\n\n")
        f.write(f"Source CSV: `{profile_csv}`\n\n")
        f.write(
            "| M | N | K | beta | TFLOP/s | runtime ms | tile | cluster | "
            "fallback | raster | swizzle | kernel schedule | epilogue schedule | operation |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['m']} | {row['n']} | {row['k']} | {row['beta']} | "
                f"{row['tflops']} | {row['runtime_ms']} | {row['tile_shape']} | "
                f"{row['cluster_shape']} | {row['fallback_cluster_shape']} | "
                f"{row['raster_order']} | {row['swizzle_size']} | "
                f"{row['kernel_schedule']} | {row['epilogue_schedule']} | "
                f"`{row['operation']}` |\n"
            )


def main():
    args = parse_args()
    rows = read_winners(args.profile_csv)
    write_csv(args.winners_csv, rows)
    write_markdown(args.winners_md, rows, args.profile_csv)
    print(f"Wrote {len(rows)} per-shape winners")
    print(f"CSV: {args.winners_csv}")
    print(f"Markdown: {args.winners_md}")


if __name__ == "__main__":
    main()
