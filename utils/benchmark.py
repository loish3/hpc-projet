import csv
import statistics
import time

def benchmark(function, *args, repeat=10, warmup=0):
    for _ in range(warmup):
        function(*args)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
 
    mean = statistics.mean(times)
    # stdev needs at least 2 values
    std = statistics.stdev(times) if len(times) > 1 else 0.0
 
    return {
        "name": function.__name__,
        "args": args,
        "times": times,
        "min": min(times),
        "max": max(times),
        "mean": mean,
        "median": statistics.median(times),
        "std": std,
        "cv_percent": std / mean * 100 if mean > 0 else 0.0,
    }


def print_benchmark_report(results):
    if isinstance(results, dict):
        results = [results]
 
    for r in results:
        args = ", ".join(repr(a) for a in r["args"])
        print("Benchmark Report:")
        print(f"min    : {r['min'] * 1e3:10.3f} ms")
        print(f"max    : {r['max'] * 1e3:10.3f} ms")
        print(f"mean   : {r['mean'] * 1e3:10.3f} ms")
        print(f"median : {r['median'] * 1e3:10.3f} ms")
        print(f"std    : {r['std'] * 1e3:10.3f} ms")
        print(f"CV     : {r['cv_percent']:10.2f} %")
 
 
def save_benchmark_csv(results, filename):
    # One row per benchmark with its statistics (in seconds)
    if isinstance(results, dict):
        results = [results]
 
    fields = ["name", "args", "runs", "min", "max", "mean",
              "median", "std", "cv_percent"]
 
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "name": r["name"],
                "args": " ".join(repr(a) for a in r["args"]),
                "runs": len(r["times"]),
                "min": r["min"],
                "max": r["max"],
                "mean": r["mean"],
                "median": r["median"],
                "std": r["std"],
                "cv_percent": r["cv_percent"],
            })


