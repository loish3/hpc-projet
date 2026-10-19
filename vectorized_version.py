#!/usr/bin/env python3
"""
Vectorized version of the HPC project pipeline (role 1: Performance & Vectorization).

The three CPU-intensive functions of sequential_pipeline.py are rewritten with
NumPy operations instead of explicit Python loops. Everything else (loading,
CSV output, global aggregation) is reused from the reference implementation,
so that the only difference between both programs is the vectorization.

Usage
-----
    python vectorized_version.py              # run the vectorized pipeline
    python vectorized_version.py --compare    # time each stage, sequential vs vectorized
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

import sequential_pipeline as seq


# ---------------------------------------------------------------------------
# Vectorized processing functions
# ---------------------------------------------------------------------------

def normalize_image_vec(image: np.ndarray) -> np.ndarray:
    """Normalize one image to [0, 1] with NumPy (no Python loop)."""
    image_min = float(image.min())
    image_max = float(image.max())
    scale = image_max - image_min

    if scale == 0.0:
        return np.zeros_like(image, dtype=np.float32)

    # Same arithmetic as the reference: computed in float64, stored in float32.
    return ((image.astype(np.float64) - image_min) / scale).astype(np.float32)


def mean_filter_3x3_vec(image: np.ndarray) -> np.ndarray:
    """3 x 3 mean filter with NumPy slicing. Border pixels are copied unchanged."""
    height, width = image.shape
    output = image.copy()

    if height < 3 or width < 3:
        return output

    img = image.astype(np.float64)

    # Each slice is the image shifted by (drow, dcol). The 9 neighbours are added
    # in the same order as the reference loops, so the result is identical.
    total = np.zeros((height - 2, width - 2), dtype=np.float64)
    for drow in (-1, 0, 1):
        for dcol in (-1, 0, 1):
            total += img[1 + drow:height - 1 + drow, 1 + dcol:width - 1 + dcol]

    output[1:-1, 1:-1] = (total / 9.0).astype(np.float32)
    return output


def compute_image_statistics_vec(image: np.ndarray, threshold: float) -> dict[str, float]:
    """Mean, maximum and number of pixels above the threshold, with NumPy."""
    img = image.astype(np.float64)
    pixel_count = img.size

    above_threshold = int(np.count_nonzero(img >= threshold))

    return {
        "mean": float(img.sum()) / pixel_count,
        "max": float(img.max()),
        "above_threshold": float(above_threshold),
        "fraction_above_threshold": above_threshold / pixel_count,
    }


def process_one_image_vec(path: Path, threshold: float) -> dict[str, float | str]:
    """load -> normalize -> 3 x 3 mean filter -> statistics (vectorized)."""
    image = seq.load_image(path)
    normalized = normalize_image_vec(image)
    filtered = mean_filter_3x3_vec(normalized)
    stats = compute_image_statistics_vec(filtered, threshold=threshold)
    return {"file": path.name, **stats}


# ---------------------------------------------------------------------------
# Main vectorized application
# ---------------------------------------------------------------------------

def run_vectorized_pipeline(data_dir: Path, output_csv: Path, threshold: float) -> float:
    image_paths = sorted(data_dir.glob("*.npy"))
    if not image_paths:
        raise FileNotFoundError(
            f"No .npy files were found in '{data_dir}'.\n"
            "Generate the demo dataset first with:\n"
            "    python sequential_pipeline.py --generate-demo"
        )

    print("=" * 68)
    print("Vectorized scientific image-processing pipeline")
    print("=" * 68)
    print(f"Dataset directory : {data_dir}")
    print(f"Number of images  : {len(image_paths)}")
    print(f"Threshold         : {threshold}")
    print()

    start_time = time.perf_counter()
    results = [process_one_image_vec(path, threshold) for path in image_paths]
    elapsed_time = time.perf_counter() - start_time

    summary = seq.compute_global_summary(results)
    seq.save_results_to_csv(results, output_path=output_csv)

    print("=" * 68)
    print("Global summary")
    print("=" * 68)
    print(f"Mean of image means            : {summary['mean_of_means']:.8f}")
    print(f"Global maximum                 : {summary['global_max']:.8f}")
    print(f"Total pixels above threshold   : {summary['total_above_threshold']:.0f}")
    print(f"Average fraction above threshold: {summary['average_fraction_above_threshold']:.8f}")
    print()
    print(f"Vectorized execution time      : {elapsed_time:.6f} s")
    print(f"Results written to             : {output_csv}")
    print("=" * 68)
    return elapsed_time


# ---------------------------------------------------------------------------
# Stage-by-stage comparison: sequential vs vectorized
# ---------------------------------------------------------------------------

def time_stage(function, inputs, repeat):
    """Median over `repeat` runs of the time needed to apply `function` to all inputs."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        for args in inputs:
            function(*args)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def compare_stages(data_dir: Path, threshold: float, repeat: int) -> None:
    image_paths = sorted(data_dir.glob("*.npy"))
    images = [seq.load_image(p) for p in image_paths]

    # 1. Correctness: every stage must give the same result in both versions.
    for image in images:
        n_seq, n_vec = seq.normalize_image(image), normalize_image_vec(image)
        assert np.array_equal(n_seq, n_vec), "normalize: different results"
        f_seq, f_vec = seq.mean_filter_3x3(n_seq), mean_filter_3x3_vec(n_seq)
        assert np.array_equal(f_seq, f_vec), "filter: different results"
        s_seq = seq.compute_image_statistics(f_seq, threshold)
        s_vec = compute_image_statistics_vec(f_seq, threshold)
        for key in s_seq:
            assert np.isclose(s_seq[key], s_vec[key], rtol=0, atol=1e-12), f"stats: {key}"
    print(f"Correctness check OK on {len(images)} images (all stages).\n")

    normalized = [seq.normalize_image(img) for img in images]
    filtered = [seq.mean_filter_3x3(img) for img in normalized]

    stages = [
        ("normalize_image", [(img,) for img in images],
         seq.normalize_image, normalize_image_vec),
        ("mean_filter_3x3", [(img,) for img in normalized],
         seq.mean_filter_3x3, mean_filter_3x3_vec),
        ("compute_image_statistics", [(img, threshold) for img in filtered],
         seq.compute_image_statistics, compute_image_statistics_vec),
    ]

    print(f"Median of {repeat} runs, all {len(images)} images per run")
    print("(load_image is not vectorized: it is the same np.load in both versions)")
    print(f"{'Stage':<26}{'Sequential (s)':>16}{'Vectorized (s)':>16}{'Speedup':>10}")
    print("-" * 68)
    total_seq = total_vec = 0.0
    for name, inputs, f_seq, f_vec in stages:
        t_seq = time_stage(f_seq, inputs, repeat)
        t_vec = time_stage(f_vec, inputs, repeat)
        total_seq += t_seq
        total_vec += t_vec
        print(f"{name:<26}{t_seq:>16.6f}{t_vec:>16.6f}{t_seq / t_vec:>10.1f}")
    print("-" * 68)
    print(f"{'Total':<26}{total_seq:>16.6f}{total_vec:>16.6f}{total_seq / total_vec:>10.1f}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized version of the HPC pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("vectorized_results.csv"))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--compare", action="store_true",
                        help="Time each stage, sequential vs vectorized.")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of runs per stage for --compare.")
    args = parser.parse_args()

    if args.compare:
        compare_stages(args.data_dir, args.threshold, args.repeat)
    else:
        run_vectorized_pipeline(args.data_dir, args.output, args.threshold)


if __name__ == "__main__":
    main()