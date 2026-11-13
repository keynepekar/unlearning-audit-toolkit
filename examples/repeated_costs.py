"""Measure ridge update costs and process high-water marks in isolated workers."""

import argparse
import ctypes
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from block_ridge import BlockRidge
from ridge_benchmark import ridge_svd
from ridge_control import RidgeStats


def peak_memory_bytes():
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                *((name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )),
            ]

        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        read_counters = ctypes.windll.psapi.GetProcessMemoryInfo
        read_counters.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not read_counters(current_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def measure(method, rows, features, fraction, pattern, repeats):
    rng = np.random.default_rng(42)
    with threadpool_limits(limits=1):
        x = rng.normal(size=(rows, features))
        y = x @ rng.normal(size=features) + rng.normal(size=rows)
        removed = np.arange(round(rows * fraction))
        if pattern == "dispersed":
            removed = rng.permutation(rows)[:len(removed)]
        keep = np.ones(rows, dtype=bool)
        keep[removed] = False
        alpha = 4.0
        timings = []
        for repetition in range(repeats + 1):
            start = perf_counter()
            if method == "gram":
                model = RidgeStats.from_rows(x, y)
            elif method == "qr":
                model = BlockRidge(x, y, alpha=alpha, block_size=256)
            setup_ms = 0.0 if method == "svd" else (perf_counter() - start) * 1000
            start = perf_counter()
            if method == "gram":
                model.subtract(RidgeStats.from_rows(x[removed], y[removed])).solve(alpha)
            elif method == "qr":
                model.remove(removed)
            else:
                ridge_svd(x[keep], y[keep], alpha)
            update_ms = (perf_counter() - start) * 1000
            if repetition:
                timings.append({"setup_ms": setup_ms, "update_ms": update_ms})
            if method != "svd":
                del model
    return {
        "method": method, "rows": rows, "features": features, "fraction": fraction, "pattern": pattern,
        "timings": timings, "peak_process_bytes": peak_memory_bytes(),
    }


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(measure(*json.loads(sys.argv[2]))))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    results = []
    for rows, features in ([(2000, 16)] if args.quick else [(12000, 32), (24000, 64)]):
        for fraction in ((0.1,) if args.quick else (0.01, 0.1, 0.5)):
            for pattern in ("concentrated", "dispersed"):
                for method in ("gram", "qr", "svd"):
                    config = [method, rows, features, fraction, pattern, 2 if args.quick else 11]
                    command = [sys.executable, __file__, "--worker", json.dumps(config)]
                    row = json.loads(subprocess.check_output(command, text=True))
                    results.append(row)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"platform": sys.platform, "results": results}, stream, indent=2)
    print(f"Wrote {len(results)} isolated cost measurements to {args.output}")


if __name__ == "__main__":
    main()
