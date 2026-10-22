#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import sequential_pipeline as seq
import vectorized_version as vec
from utils.benchmark import benchmark


WORKERS = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# Thread pool
# ---------------------------------------------------------------------------

def run_thread(func, items, workers: int) -> list:
    """
    Apply func to every item with a pool of threads.

    map() returns the results in the same order as the items.
    """
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(func, items))


# ---------------------------------------------------------------------------
# Work done by one thread on one item
# ---------------------------------------------------------------------------

def load_image_slow(path: Path):
    """
    Load one image after a 50 ms wait, which simulates a slow (network) disk.
    """
    time.sleep(0.05)
    return seq.load_image(path)


def cpu_python(image, threshold: float):
    """
    CPU part of the pipeline with the reference functions (Python loops).
    """
    img = seq.normalize_image(image)
    img = seq.mean_filter_3x3(img)
    return seq.compute_image_statistics(img, threshold)


def cpu_vectorized(image, threshold: float):
    """
    CPU part of the pipeline with the vectorized functions (NumPy).
    """
    img = vec.normalize_image_vec(image)
    img = vec.mean_filter_3x3_vec(img)
    return vec.compute_image_statistics_vec(img, threshold)


# ---------------------------------------------------------------------------
# Threading experiments
# ---------------------------------------------------------------------------

def run_experiment(label: str, func, items, repeat: int = 5) -> None:
    """
    Time func on all items with 1, 2, 4 and 8 threads.

    For each number of threads, print the median time, speedup, efficiency,
    coefficient of variation and the CPU time / wall time ratio.
    """

    print(f"\n=== {label} ===")

    measures = {}

    for n in WORKERS:
        r = benchmark(run_thread, func, items, n, repeat=repeat, warmup=1)

        # One more run to compare CPU time (all threads) with wall time.
        c0, w0 = time.process_time(), time.perf_counter()
        run_thread(func, items, n)
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0

        measures[n] = (r, cpu / wall)

    t1 = measures[1][0]["median"]

    for n, (r, cpu_wall) in measures.items():
        t = r["median"]
        speedup = t1 / t

        print(
            f"{n} threads : {t:.4f} s | speedup {speedup:.2f} "
            f"| efficiency {speedup / n:.2f} | CV {r['cv_percent']:.1f} % "
            f"| CPU/wall {cpu_wall:.2f}"
        )


def run_experiments(data_dir: Path, threshold: float) -> None:
    """
    A: I/O, B: CPU with Python loops, C: CPU with NumPy, D: complete pipeline.
    """

    paths = sorted(data_dir.glob("*.npy"))

    # A. I/O
    run_experiment("A. I/O (np.load)", seq.load_image, paths)
    run_experiment("A'. I/O with latency", load_image_slow, paths)

    # B and C: images are loaded before the timed region (CPU only).
    images = [seq.load_image(p) for p in paths]
    run_experiment("B. CPU, Python loops",
                   partial(cpu_python, threshold=threshold), images, repeat=3)
    run_experiment("C. CPU, vectorized",
                   partial(cpu_vectorized, threshold=threshold), images)

    # D. Complete pipeline (load + processing).
    run_experiment("D. Complete pipeline, Python loops",
                   partial(seq.process_one_image, threshold=threshold), paths, repeat=3)
    run_experiment("D'. Complete pipeline, vectorized",
                   partial(vec.process_one_image_vec, threshold=threshold), paths)


# ---------------------------------------------------------------------------
# Main threaded application
# ---------------------------------------------------------------------------

def run_threaded_pipeline(
    data_dir: Path,
    output_csv: Path,
    threshold: float,
    workers: int,
    vectorized: bool,
) -> None:
    """
    Run the complete application with a pool of threads.

    Each thread processes whole images (load -> normalize -> filter -> stats).
    Aggregation and CSV output stay sequential, as in the baseline.
    """

    image_paths = sorted(data_dir.glob("*.npy"))

    process = vec.process_one_image_vec if vectorized else seq.process_one_image
    process = partial(process, threshold=threshold)

    start_time = time.perf_counter()

    results = run_thread(process, image_paths, workers)

    elapsed_time = time.perf_counter() - start_time

    summary = seq.compute_global_summary(results)

    seq.save_results_to_csv(
        results,
        output_path=output_csv,
    )

    print("=" * 68)
    print(f"Threaded pipeline ({workers} threads, "
          f"{'vectorized' if vectorized else 'Python loops'})")
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
    print(f"Threaded execution time        : {elapsed_time:.6f} s")
    print(f"Results written to             : {output_csv}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Threading version of the HPC parallel-computing project."
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
        default=Path("threading_results.csv"),
        help="CSV file in which the results will be written.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Threshold used by the statistics stage.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of threads used by the pipeline.",
    )

    parser.add_argument(
        "--vectorized",
        action="store_true",
        help="Use the vectorized processing functions.",
    )

    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run the threading experiments (I/O vs CPU) and exit.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    if args.experiments:

        run_experiments(
            data_dir=args.data_dir,
            threshold=args.threshold,
        )

        return

    run_threaded_pipeline(
        data_dir=args.data_dir,
        output_csv=args.output,
        threshold=args.threshold,
        workers=args.workers,
        vectorized=args.vectorized,
    )


if __name__ == "__main__":
    main()
