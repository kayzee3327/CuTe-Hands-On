#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


EXPECTED_CONTRACT = ("tnn", "e4m3", "e4m3", "bf16", "f32", "bf16")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate CUTLASS FP8 E4M3 TNN BF16 GEMM problem JSON."
    )
    parser.add_argument("problems_file", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    problems = json.loads(args.problems_file.read_text())
    if not isinstance(problems, list) or not problems:
        raise SystemExit(f"{args.problems_file} must contain a non-empty JSON list")

    for idx, problem in enumerate(problems):
        contract = (
            problem.get("layout"),
            problem.get("dtype_a"),
            problem.get("dtype_b"),
            problem.get("dtype_c"),
            problem.get("dtype_acc"),
            problem.get("dtype_d"),
        )
        if contract != EXPECTED_CONTRACT:
            layout, dtype_a, dtype_b, dtype_c, dtype_acc, dtype_d = contract
            raise SystemExit(
                f"problem {idx} is {layout}/{dtype_a}/{dtype_b}/{dtype_c}/"
                f"{dtype_acc}/{dtype_d}, but this script is for "
                "tnn/e4m3/e4m3/bf16/f32/bf16 GEMM"
            )

    print(f"Validated {len(problems)} FP8 TNN problems from {args.problems_file}")


if __name__ == "__main__":
    main()
