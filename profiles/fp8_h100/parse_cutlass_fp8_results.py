#!/usr/bin/env python3

import argparse
import csv
import glob
import re
from pathlib import Path


def find_column(fieldnames, *candidates):
    normalized = {name.strip().lower().replace("-", "_"): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.strip().lower().replace("-", "_")
        if key in normalized:
            return normalized[key]
    return None


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def get_first(row, *columns):
    for col in columns:
        if col and row.get(col, "") != "":
            return row.get(col)
    return ""


def parse_shape_from_operation(operation):
    match = re.search(r"_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_", operation)
    if not match:
        return {}
    return {
        "cta_m": match.group(1),
        "cta_n": match.group(2),
        "cta_k": match.group(3),
        "cluster_m": match.group(4),
        "cluster_n": match.group(5),
        "cluster_k": match.group(6),
    }


def parse_schedule_from_operation(operation):
    kernel_schedule = ""
    epilogue_schedule = ""

    suffixes = [
        ("warpspecialized_cooperative_fp8_fastaccum", "TmaWarpSpecializedCooperativeFP8FastAccum"),
        ("warpspecialized_pingpong_fp8_fastaccum", "TmaWarpSpecializedPingpongFP8FastAccum"),
        ("warpspecialized_fp8_fastaccum", "TmaWarpSpecializedFP8FastAccum"),
        ("warpspecialized_cooperative", "TmaWarpSpecializedCooperative"),
        ("warpspecialized_pingpong", "TmaWarpSpecializedPingpong"),
        ("warpspecialized", "TmaWarpSpecialized"),
        ("cpasync_warpspecialized_cooperative", "CpAsyncWarpSpecializedCooperative"),
        ("cpasync_warpspecialized", "CpAsyncWarpSpecialized"),
        ("stream_k", "StreamK"),
    ]
    for token, name in suffixes:
        if token in operation:
            kernel_schedule = name
            break

    if "epi_tma" in operation and "cooperative" in operation:
        epilogue_schedule = "TmaWarpSpecializedCooperative"
    elif "epi_tma" in operation:
        epilogue_schedule = "TmaWarpSpecialized"
    elif "epi_nosmem" in operation:
        epilogue_schedule = "NoSmemWarpSpecialized"
    elif "epi_transposed" in operation:
        epilogue_schedule = "EpilogueTransposed"

    return kernel_schedule, epilogue_schedule


def format_shape(row, prefix):
    values = [row.get(f"{prefix}_m", ""), row.get(f"{prefix}_n", ""), row.get(f"{prefix}_k", "")]
    return "x".join(values) if all(values) else ""


def summarize(input_csv, output_csv, output_md):
    fieldnames, rows = load_rows(input_csv)
    if not fieldnames:
        raise RuntimeError(f"{input_csv} has no CSV header")

    m_col = find_column(fieldnames, "m")
    n_col = find_column(fieldnames, "n")
    k_col = find_column(fieldnames, "k")
    runtime_col = find_column(fieldnames, "runtime", "runtime_ms")
    math_col = find_column(fieldnames, "math", "gflops", "tflops")
    op_col = find_column(fieldnames, "operation", "kernel", "name")
    cta_m_col = find_column(fieldnames, "cta_m", "threadblock_shape::m")
    cta_n_col = find_column(fieldnames, "cta_n", "threadblock_shape::n")
    cta_k_col = find_column(fieldnames, "cta_k", "threadblock_shape::k")
    instr_m_col = find_column(fieldnames, "instr_tile_m", "inst_m")
    instr_n_col = find_column(fieldnames, "instr_tile_n", "inst_n")
    instr_k_col = find_column(fieldnames, "instr_tile_k", "inst_k")
    cluster_m_col = find_column(fieldnames, "cluster_m", "cluster_shape::m")
    cluster_n_col = find_column(fieldnames, "cluster_n", "cluster_shape::n")
    cluster_k_col = find_column(fieldnames, "cluster_k", "cluster_shape::k")
    cluster_m_fb_col = find_column(fieldnames, "cluster_m_fallback", "cluster_shape_fallback::m")
    cluster_n_fb_col = find_column(fieldnames, "cluster_n_fallback", "cluster_shape_fallback::n")
    cluster_k_fb_col = find_column(fieldnames, "cluster_k_fallback", "cluster_shape_fallback::k")
    raster_col = find_column(fieldnames, "raster_order", "raster")
    swizzle_col = find_column(fieldnames, "swizzle_size", "swizzle")
    split_k_col = find_column(fieldnames, "split_k_slices", "split_k")
    stages_col = find_column(fieldnames, "stages")
    status_col = find_column(fieldnames, "status")
    disposition_col = find_column(fieldnames, "disposition")

    required = [m_col, n_col, k_col, runtime_col, op_col]
    if any(col is None for col in required):
        raise RuntimeError(
            "Could not find required columns. "
            f"Header was: {', '.join(fieldnames)}")

    best = {}
    for row in rows:
        status = row.get(status_col, "success").strip().lower() if status_col else "success"
        disposition = row.get(disposition_col, "passed").strip().lower() if disposition_col else "passed"
        if status and status not in {"success", "passed"}:
            continue
        if disposition and disposition not in {
            "passed", "notrun", "not run", "not_run", "not verified", "not_verified"}:
            continue

        m = as_int(row.get(m_col))
        n = as_int(row.get(n_col))
        k = as_int(row.get(k_col))
        runtime = as_float(row.get(runtime_col))
        if m is None or n is None or k is None or runtime is None:
            continue

        key = (m, n, k)
        if key not in best or runtime < best[key]["runtime_ms"]:
            math = as_float(row.get(math_col)) if math_col else None
            operation = row.get(op_col, "")
            parsed_shape = parse_shape_from_operation(operation)
            kernel_schedule, epilogue_schedule = parse_schedule_from_operation(operation)
            normalized = {
                "cta_m": get_first(row, cta_m_col) or parsed_shape.get("cta_m", ""),
                "cta_n": get_first(row, cta_n_col) or parsed_shape.get("cta_n", ""),
                "cta_k": get_first(row, cta_k_col) or parsed_shape.get("cta_k", ""),
                "instr_m": get_first(row, instr_m_col),
                "instr_n": get_first(row, instr_n_col),
                "instr_k": get_first(row, instr_k_col),
                "cluster_m": get_first(row, cluster_m_col) or parsed_shape.get("cluster_m", ""),
                "cluster_n": get_first(row, cluster_n_col) or parsed_shape.get("cluster_n", ""),
                "cluster_k": get_first(row, cluster_k_col) or parsed_shape.get("cluster_k", ""),
                "cluster_m_fallback": get_first(row, cluster_m_fb_col),
                "cluster_n_fallback": get_first(row, cluster_n_fb_col),
                "cluster_k_fallback": get_first(row, cluster_k_fb_col),
                "raster_order": get_first(row, raster_col),
                "swizzle_size": get_first(row, swizzle_col),
                "split_k_slices": get_first(row, split_k_col),
                "stages": get_first(row, stages_col),
                "kernel_schedule": kernel_schedule,
                "epilogue_schedule": epilogue_schedule,
            }
            best[key] = {
                "m": m,
                "n": n,
                "k": k,
                "runtime_ms": runtime,
                "math": math,
                "operation": operation,
                **normalized,
                "raw": row,
            }

    if not best:
        raise RuntimeError(f"No successful profiler rows found in {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "m", "n", "k", "runtime_ms", "math",
                "cta_shape", "instr_shape", "cluster_shape", "cluster_fallback_shape",
                "kernel_schedule", "epilogue_schedule",
                "raster_order", "swizzle_size", "split_k_slices", "stages",
                "operation"])
        writer.writeheader()
        for key in sorted(best):
            item = best[key]
            writer.writerow({
                "m": item["m"],
                "n": item["n"],
                "k": item["k"],
                "runtime_ms": item["runtime_ms"],
                "math": "" if item["math"] is None else item["math"],
                "cta_shape": format_shape(item, "cta"),
                "instr_shape": "x".join([item["instr_m"], item["instr_n"], item["instr_k"]]) if all(
                    [item["instr_m"], item["instr_n"], item["instr_k"]]) else "",
                "cluster_shape": format_shape(item, "cluster"),
                "cluster_fallback_shape": "x".join([
                    item["cluster_m_fallback"], item["cluster_n_fallback"], item["cluster_k_fallback"]]) if all(
                    [item["cluster_m_fallback"], item["cluster_n_fallback"], item["cluster_k_fallback"]]) else "",
                "kernel_schedule": item["kernel_schedule"],
                "epilogue_schedule": item["epilogue_schedule"],
                "raster_order": item["raster_order"],
                "swizzle_size": item["swizzle_size"],
                "split_k_slices": item["split_k_slices"],
                "stages": item["stages"],
                "operation": item["operation"],
            })

    with open(output_md, "w") as f:
        f.write("# CUTLASS FP8 H100 Winners\n\n")
        f.write("| M | N | K | Runtime ms | Math | CTA | Cluster | Kernel schedule | Raster | Swizzle | Operation |\n")
        f.write("|---:|---:|---:|---:|---:|---|---|---|---|---:|---|\n")
        for key in sorted(best):
            item = best[key]
            math = "" if item["math"] is None else f"{item['math']:.2f}"
            f.write(
                f"| {item['m']} | {item['n']} | {item['k']} | "
                f"{item['runtime_ms']:.6f} | {math} | "
                f"{format_shape(item, 'cta')} | {format_shape(item, 'cluster')} | "
                f"{item['kernel_schedule']} | {item['raster_order']} | {item['swizzle_size']} | "
                f"`{item['operation']}` |\n")


def main():
    parser = argparse.ArgumentParser(description="Extract fastest CUTLASS profiler row per GEMM shape.")
    parser.add_argument("input", help="Profiler CSV path or glob pattern")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    matches = sorted(glob.glob(args.input))
    if not matches:
        raise SystemExit(f"No CSV files matched: {args.input}")
    if len(matches) > 1:
        # CUTLASS usually appends operation names to --output; prefer Gemm output.
        gemm_matches = [path for path in matches if "gemm" in Path(path).name.lower()]
        matches = gemm_matches or matches
    input_csv = Path(matches[0])

    summarize(input_csv, Path(args.output_csv), Path(args.output_md))
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
