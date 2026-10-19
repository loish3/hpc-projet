#!/usr/bin/env python3
"""
Sequential baseline for the HPC project:
Parallel Processing of a Large Scientific Image Dataset

Requirements
------------
Python 3.10+
NumPy

Typical usage
-------------
1. Generate a deterministic demo dataset:

    python sequential_pipeline.py --generate-demo

2. Run the sequential baseline:

    python sequential_pipeline.py

3. Profile it:

    python -m cProfile -s cumulative sequential_pipeline.py

The goal of the project is NOT to change the numerical result.
Optimized/parallel versions should remain consistent with this baseline.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_demo_dataset(
    data_dir: Path,
    n_images: int = 24,
    height: int = 256,
    width: int = 256,
    seed: int = 42,
) -> None:
    """
    Generate a deterministic dataset of synthetic 2D scientific images.

    The images contain:
      - random background noise,
      - a smooth spatial pattern,
      - a few bright circular regions.

    Files are stored as .npy arrays so that loading is simple and reproducible.
    """

    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:height, 0:width]
    smooth_pattern = (
        0.15 * np.sin(xx / 17.0)
        + 0.10 * np.cos(yy / 23.0)
    )

    for image_id in range(n_images):
        image = rng.normal(
            loc=0.45,
            scale=0.08,
            size=(height, width),
        ).astype(np.float32)

        image += smooth_pattern.astype(np.float32)

        # Add a few bright circular regions.
        n_spots = 3 + (image_id % 4)

        min_dimension = min(height, width)
        margin = max(2, min_dimension // 8)
        max_radius = max(2, min_dimension // 16)

        for _ in range(n_spots):
            cy = int(rng.integers(margin, height - margin))
            cx = int(rng.integers(margin, width - margin))
            radius = int(rng.integers(2, max_radius + 1))
            amplitude = float(rng.uniform(0.25, 0.55))

            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            image[mask] += amplitude

        image = np.clip(image, 0.0, 1.0)

        output_path = data_dir / f"image_{image_id:04d}.npy"
        np.save(output_path, image)

    print(
        f"Generated {n_images} images in '{data_dir}' "
        f"with shape {height} x {width}."
    )


# ---------------------------------------------------------------------------
# Sequential processing functions
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """
    Load one image from disk.
    """
    return np.load(path)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize one image to the range [0, 1].

    This implementation intentionally uses explicit Python loops.
    It is correct, but students may investigate whether a different
    implementation is more efficient.
    """

    image_min = float(image.min())
    image_max = float(image.max())

    output = np.empty_like(image, dtype=np.float32)

    scale = image_max - image_min

    if scale == 0.0:
        output.fill(0.0)
        return output

    height, width = image.shape

    for row in range(height):
        for col in range(width):
            value = float(image[row, col])
            output[row, col] = (value - image_min) / scale

    return output


def mean_filter_3x3(image: np.ndarray) -> np.ndarray:
    """
    Apply a simple 3 x 3 mean filter.

    Border pixels are copied unchanged.

    The implementation intentionally uses explicit Python loops so that
    the sequential baseline contains a clear CPU-intensive section.
    """

    height, width = image.shape

    output = image.copy()

    for row in range(1, height - 1):
        for col in range(1, width - 1):

            total = 0.0

            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    total += float(image[row + drow, col + dcol])

            output[row, col] = total / 9.0

    return output


def compute_image_statistics(
    image: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Compute several statistics for one processed image.

    The returned dictionary will later be written to a CSV file.
    """

    height, width = image.shape

    total = 0.0
    maximum = float("-inf")
    above_threshold = 0

    # Explicit loops are kept here on purpose.
    for row in range(height):
        for col in range(width):

            value = float(image[row, col])

            total += value

            if value > maximum:
                maximum = value

            if value >= threshold:
                above_threshold += 1

    pixel_count = height * width

    mean_value = total / pixel_count
    fraction_above_threshold = above_threshold / pixel_count

    return {
        "mean": mean_value,
        "max": maximum,
        "above_threshold": float(above_threshold),
        "fraction_above_threshold": fraction_above_threshold,
    }


def process_one_image(
    path: Path,
    threshold: float,
) -> dict[str, float | str]:
    """
    Complete sequential processing pipeline for one image.

    Pipeline:
        load
          -> normalize
          -> 3 x 3 mean filter
          -> statistics
    """

    image = load_image(path)

    normalized = normalize_image(image)

    filtered = mean_filter_3x3(normalized)

    statistics = compute_image_statistics(
        filtered,
        threshold=threshold,
    )

    return {
        "file": path.name,
        **statistics,
    }


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def save_results_to_csv(
    results: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    """
    Save per-image results to a CSV file.
    """

    fieldnames = [
        "file",
        "mean",
        "max",
        "above_threshold",
        "fraction_above_threshold",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)


def compute_global_summary(
    results: list[dict[str, float | str]],
) -> dict[str, float]:
    """
    Aggregate the results produced for all images.
    """

    if not results:
        raise ValueError("No results to aggregate.")

    mean_of_means = sum(
        float(result["mean"])
        for result in results
    ) / len(results)

    global_max = max(
        float(result["max"])
        for result in results
    )

    total_above_threshold = sum(
        float(result["above_threshold"])
        for result in results
    )

    average_fraction_above_threshold = sum(
        float(result["fraction_above_threshold"])
        for result in results
    ) / len(results)

    return {
        "mean_of_means": mean_of_means,
        "global_max": global_max,
        "total_above_threshold": total_above_threshold,
        "average_fraction_above_threshold": (
            average_fraction_above_threshold
        ),
    }


# ---------------------------------------------------------------------------
# Main sequential application
# ---------------------------------------------------------------------------

def run_sequential_pipeline(
    data_dir: Path,
    output_csv: Path,
    threshold: float,
) -> None:
    """
    Run the complete application sequentially.

    Images are deliberately processed one after another.
    """

    image_paths = sorted(data_dir.glob("*.npy"))

    if not image_paths:
        raise FileNotFoundError(
            f"No .npy files were found in '{data_dir}'.\n"
            "Generate the demo dataset first with:\n"
            "    python sequential_pipeline.py --generate-demo"
        )

    print("=" * 68)
    print("Sequential scientific image-processing pipeline")
    print("=" * 68)
    print(f"Dataset directory : {data_dir}")
    print(f"Number of images  : {len(image_paths)}")
    print(f"Threshold         : {threshold}")
    print()

    results: list[dict[str, float | str]] = []

    start_time = time.perf_counter()

    # IMPORTANT:
    # This is the sequential baseline.
    # Only one image is processed at a time.
    for index, path in enumerate(image_paths, start=1):

        result = process_one_image(
            path,
            threshold=threshold,
        )

        results.append(result)

        print(
            f"[{index:03d}/{len(image_paths):03d}] "
            f"{path.name} processed"
        )

    elapsed_time = time.perf_counter() - start_time

    summary = compute_global_summary(results)

    save_results_to_csv(
        results,
        output_path=output_csv,
    )

    print()
    print("=" * 68)
    print("Global summary")
    print("=" * 68)
    print(
        f"Mean of image means            : "
        f"{summary['mean_of_means']:.8f}"
    )
    print(
        f"Global maximum                 : "
        f"{summary['global_max']:.8f}"
    )
    print(
        f"Total pixels above threshold   : "
        f"{summary['total_above_threshold']:.0f}"
    )
    print(
        f"Average fraction above threshold: "
        f"{summary['average_fraction_above_threshold']:.8f}"
    )
    print()
    print(f"Sequential execution time      : {elapsed_time:.6f} s")
    print(f"Results written to             : {output_csv}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Sequential baseline for the HPC parallel-computing project."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory containing .npy image files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sequential_results.csv"),
        help="CSV file in which the results will be written.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Threshold used by the statistics stage.",
    )

    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate a deterministic demo dataset and exit.",
    )

    parser.add_argument(
        "--n-images",
        type=int,
        default=24,
        help="Number of demo images to generate.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Height of generated demo images.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Width of generated demo images.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for demo dataset generation.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    if args.generate_demo:

        generate_demo_dataset(
            data_dir=args.data_dir,
            n_images=args.n_images,
            height=args.height,
            width=args.width,
            seed=args.seed,
        )

        return

    run_sequential_pipeline(
        data_dir=args.data_dir,
        output_csv=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
