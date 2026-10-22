#!/usr/bin/env python3

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import sequential_pipeline as seq
import vectorized_version as vec
from utils.benchmark import benchmark


WORKERS = [1, 2, 4, 8]


def run_pool(func, items, workers: int, start_method: str | None = None) -> list:
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=workers) as pool:
        return pool.map(func, items, chunksize=1)


def pool_startup_time(workers: int, start_method: str | None = None) -> float:
    ctx = mp.get_context(start_method)
    start = time.perf_counter()
    with ctx.Pool(processes=workers) as pool:
        pool.map(abs, range(workers), chunksize=1)
    return time.perf_counter() - start


def image_pipeline(vectorized: bool, threshold: float):
    func = vec.process_one_image_vec if vectorized else seq.process_one_image
    return partial(func, threshold=threshold)


def cpu_vectorized(image, threshold: float):
    img = vec.normalize_image_vec(image)
    img = vec.mean_filter_3x3_vec(img)
    return vec.compute_image_statistics_vec(img, threshold)


def run_sequential(func, items) -> list:
    return [func(item) for item in items]


def karp_flatt(speedup: float, p: int) -> float:
    return (1 / speedup - 1 / p) / (1 - 1 / p)


def print_scaling_table(times: dict[int, float], t_seq: float) -> None:
    t1 = times[1]

    print(f"{'p':>3} | {'Tp (s)':>9} | {'speedup':>7} | {'effic.':>6} "
          f"| {'Karp-Flatt':>10} | {'vs seq.':>7}")
    print("-" * 60)

    for p, t in times.items():
        speedup = t1 / t
        kf = karp_flatt(speedup, p) if p > 1 else 0.0
        print(f"{p:>3} | {t:>9.4f} | {speedup:>7.2f} | {speedup / p:>6.2f} "
              f"| {kf:>10.3f} | {t_seq / t:>7.2f}")


def run_experiment(label, func, items, workers, start_method, repeat=3) -> None:
    print(f"\n=== {label} ===")

    t_seq = benchmark(run_sequential, func, items, repeat=repeat, warmup=1)["median"]
    print(f"Plain sequential loop : {t_seq:.4f} s")

    times = {}
    for p in workers:
        r = benchmark(run_pool, func, items, p, start_method, repeat=repeat, warmup=1)
        times[p] = r["median"]

    print_scaling_table(times, t_seq)


def check_results(data_dir: Path, threshold: float, workers: int,
                  start_method: str | None) -> None:
    paths = sorted(data_dir.glob("*.npy"))

    reference = run_sequential(image_pipeline(False, threshold), paths)

    for vectorized in (False, True):
        results = run_pool(image_pipeline(vectorized, threshold), paths,
                           workers, start_method)
        name = "vectorized" if vectorized else "Python loops"
        status = "OK" if results == reference else "DIFFERENT"
        print(f"Correctness ({name}, {workers} processes) : {status}")


def run_experiments(data_dir: Path, threshold: float, start_method: str | None,
                    workers: list[int]) -> None:
    paths = sorted(data_dir.glob("*.npy"))
    method = mp.get_context(start_method).get_start_method()

    print(f"Images: {len(paths)} | logical CPUs: {os.cpu_count()} "
          f"| start method: {method}\n")

    check_results(data_dir, threshold, max(workers), start_method)

    print("\n=== Cost of an empty pool (start + stop) ===")
    for p in workers:
        t = min(pool_startup_time(p, start_method) for _ in range(5))
        print(f"{p} processes : {t * 1e3:.1f} ms")

    run_experiment("A. Complete pipeline, Python loops",
                   image_pipeline(False, threshold), paths, workers, start_method)
    run_experiment("B. Complete pipeline, vectorized",
                   image_pipeline(True, threshold), paths, workers, start_method,
                   repeat=10)

    images = [seq.load_image(p) for p in paths]
    run_experiment("C. Vectorized, images sent to the processes",
                   partial(cpu_vectorized, threshold=threshold), images,
                   workers, start_method, repeat=10)


def print_summary(summary: dict[str, float], title: str, elapsed: float,
                  output_csv: Path) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)
    print(f"Mean of image means            : {summary['mean_of_means']:.8f}")
    print(f"Global maximum                 : {summary['global_max']:.8f}")
    print(f"Total pixels above threshold   : {summary['total_above_threshold']:.0f}")
    print(f"Average fraction above threshold: "
          f"{summary['average_fraction_above_threshold']:.8f}")
    print()
    print(f"Multiprocessing execution time : {elapsed:.6f} s")
    print(f"Results written to             : {output_csv}")
    print("=" * 68)


def run_multiprocessing_pipeline(
    data_dir: Path,
    output_csv: Path,
    threshold: float,
    workers: int,
    vectorized: bool,
    start_method: str | None,
) -> None:
    image_paths = sorted(data_dir.glob("*.npy"))

    start_time = time.perf_counter()
    results = run_pool(image_pipeline(vectorized, threshold), image_paths,
                       workers, start_method)
    elapsed_time = time.perf_counter() - start_time

    summary = seq.compute_global_summary(results)
    seq.save_results_to_csv(results, output_path=output_csv)

    kind = "vectorized" if vectorized else "Python loops"
    print_summary(summary, f"Multiprocessing pipeline ({workers} processes, {kind})",
                  elapsed_time, output_csv)


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Multiprocessing version of the HPC parallel-computing project."
    )

    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("multiprocessing_results.csv"))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vectorized", action="store_true")
    parser.add_argument("--start-method", default=None)
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--workers-list", type=int, nargs="+", default=WORKERS)

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    if args.experiments:
        run_experiments(args.data_dir, args.threshold, args.start_method,
                        args.workers_list)
        return

    run_multiprocessing_pipeline(
        data_dir=args.data_dir,
        output_csv=args.output,
        threshold=args.threshold,
        workers=args.workers,
        vectorized=args.vectorized,
        start_method=args.start_method,
    )


if __name__ == "__main__":
    main()
