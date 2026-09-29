#!/usr/bin/env python3
"""Run equivalence checks, benchmarks, and matrix-structure plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numba

from assembly_case import (
    assemble_dense,
    assemble_packed,
    build_case,
    materialize_dense_v,
    storage_bytes,
    validate_topology,
)


FULL_CONFIGURATIONS = [(5, 0), (10, 0), (25, 0), (50, 0), (5, 3), (10, 3)]
QUICK_CONFIGURATIONS = [(5, 0), (10, 0), (5, 3)]


def median_time(function, warmups: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) * 1e-9)
    return statistics.median(samples), samples


def plot_matrix_structure(case, dense_v, output: Path) -> None:
    branch = 0
    packed = case.packed_v[branch].reshape(
        case.segments * case.local_dofs, -1
    )
    dense = dense_v[branch].reshape(case.segments * case.local_dofs, -1)
    packed_mask = np.abs(packed) > 1e-14
    dense_mask = np.abs(dense) > 1e-14

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    axes[0].imshow(packed_mask, cmap="Greys", interpolation="nearest", aspect="auto")
    axes[0].set_title("(a) Packed reachable columns")
    axes[0].set_xlabel("Packed column")
    axes[0].set_ylabel("Stacked segment-local rows")
    axes[1].imshow(dense_mask, cmap="Greys", interpolation="nearest", aspect="auto")
    axes[1].set_title("(b) Dense global-column reference")
    axes[1].set_xlabel("Global generalized coordinate")
    axes[1].set_ylabel("Stacked segment-local rows")
    fig.suptitle(
        f"Partial-velocity structure: {case.segments} segments, "
        f"{case.internal_modes} internal modes/segment"
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_scaling(rows: list[dict], output: Path) -> None:
    optimized_color = "#0072B2"
    dense_color = "#D55E00"
    base = [row for row in rows if row["internal_modes"] == 0]
    base.sort(key=lambda row: row["segments"])
    labels = [str(row["segments"]) for row in base]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    axes[0].plot(
        labels, [r["dense_median_ms"] for r in base], color=dense_color,
        linestyle="--", marker="o", markerfacecolor="white", label="Dense reference"
    )
    axes[0].plot(
        labels, [r["packed_median_ms"] for r in base], color=optimized_color,
        linestyle="-", marker="s", label="Topology-aware"
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Segments per blade")
    axes[0].set_ylabel("Median assembly time (ms)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        labels, [r["dense_v_mib"] for r in base], color=dense_color,
        linestyle="--", marker="o", markerfacecolor="white", label="Dense global-column V"
    )
    axes[1].plot(
        labels, [r["packed_v_mib"] for r in base], color=optimized_color,
        linestyle="-", marker="s", label="Packed reachable V"
    )
    axes[1].set_xlabel("Segments per blade")
    axes[1].set_ylabel("Partial-velocity storage (MiB)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a three-case smoke test")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    configurations = QUICK_CONFIGURATIONS if args.quick else FULL_CONFIGURATIONS
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    validation: list[dict] = []

    for segments, modes in configurations:
        case = build_case(segments, modes)
        validate_topology(case)
        dense_v = materialize_dense_v(case)

        # Compile the Numba kernel before correctness and timing measurements.
        packed_mass, packed_force = assemble_packed(case)
        dense_mass, dense_force = assemble_dense(case, dense_v)
        mass_error = float(np.max(np.abs(packed_mass - dense_mass)))
        force_error = float(np.max(np.abs(packed_force - dense_force)))
        symmetry_error = float(np.max(np.abs(packed_mass - packed_mass.T)))
        passed = mass_error <= 1e-9 and force_error <= 1e-10 and symmetry_error <= 1e-11

        packed_median, packed_samples = median_time(
            lambda: assemble_packed(case), args.warmups, args.repeats
        )
        dense_median, dense_samples = median_time(
            lambda: assemble_dense(case, dense_v), args.warmups, args.repeats
        )
        memory = storage_bytes(case, dense_v)
        row = {
            "segments": segments,
            "internal_modes": modes,
            "global_dofs": case.global_dofs,
            "tip_active_dofs": int(case.active_counts[0, -1]),
            "local_dofs": case.local_dofs,
            "packed_median_ms": 1000.0 * packed_median,
            "dense_median_ms": 1000.0 * dense_median,
            "speedup": dense_median / packed_median,
            "packed_v_mib": memory["packed_v_only"] / 2**20,
            "packed_with_indices_mib": memory["packed_with_indices"] / 2**20,
            "dense_v_mib": memory["dense_v"] / 2**20,
        }
        rows.append(row)
        validation.append({
            "segments": segments,
            "internal_modes": modes,
            "mass_max_abs_difference": mass_error,
            "force_max_abs_difference": force_error,
            "symmetry_max_abs_difference": symmetry_error,
            "topology_check": True,
            "passed": passed,
            "packed_samples_s": packed_samples,
            "dense_samples_s": dense_samples,
        })
        print(json.dumps({**row, "passed": passed}, indent=2))

        if segments == 10 and modes == 0:
            plot_matrix_structure(case, dense_v, output_dir / "matrix_structure.png")

    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "matplotlib": matplotlib.__version__,
        "configurations": configurations,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timing_note": "Wall-clock values are machine and BLAS dependent.",
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_scaling(rows, output_dir / "scaling.png")

    if not all(item["passed"] for item in validation):
        raise SystemExit("assembly validation failed")
    print(f"All {len(validation)} configurations passed. Results: {output_dir}")


if __name__ == "__main__":
    main()
