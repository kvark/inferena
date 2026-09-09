#!/usr/bin/env python3
"""Rank measured CUDA kernels or Vulkan submissions in an Nsight SQLite export."""

import argparse
from pathlib import Path
import sqlite3


def report(database, top):
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        ranges = db.execute(
            "SELECT start, end, coalesce(text, value) FROM NVTX_EVENTS "
            "LEFT JOIN StringIds ON textId=StringIds.id ORDER BY start"
        ).fetchall()
        phases = [row for row in ranges if row[2] and row[2].endswith("/measure")]
        if not phases:
            raise ValueError("no measurement ranges; API-only traces are insufficient")
        for start, end, name in phases:
            if end is None:
                raise ValueError(f"{name}: incomplete measurement range")
            engine = name.split("/")[0]
            samples = [(a, b) for a, b, label in ranges
                       if label == f"{engine}/sample" and b is not None and start <= a < b <= end]
            if not samples or any(b > c for (_, b), (c, _) in zip(samples, samples[1:])):
                raise ValueError(f"{name}: missing or overlapping samples")
            count = len(samples)
            print(f"\n{name}: {count} calls; mean host sample "
                  f"{sum(b - a for a, b in samples) / count / 1e6:.3f} ms")
            for label in ("step", "wait"):
                spans = [(a, b) for a, b, text in ranges if text == f"{engine}/{label}"
                         and b is not None and start <= a < b <= end]
                if spans:
                    print(f"  host {label}: {sum(b-a for a,b in spans) / count / 1e6:.3f} ms/call")

            table = {"pytorch": "CUPTI_ACTIVITY_KIND_KERNEL", "meganeura": "VULKAN_WORKLOAD"}[engine]
            events = db.execute(f"SELECT start, end FROM {table} WHERE end>? AND start<?",
                                (start, end)).fetchall()
            if not events or any(not any(a <= x < y <= b for a, b in samples) for x, y in events):
                raise ValueError(f"{name}: GPU events missing or outside complete samples")
            print(f"  {len(events) / count:g} GPU intervals/call; summed duration "
                  f"{sum(b-a for a,b in events) / count / 1e6:.3f} ms/call")
            if engine == "meganeura":
                print("  Vulkan intervals are grouped submissions, not individual shaders.")
                continue
            graph_nodes = db.execute(
                f"SELECT count(*) FROM {table} WHERE start>=? AND end<=? AND graphNodeId>0",
                (start, end),
            ).fetchone()[0]
            print(f"  CUDA graph-node events: {graph_nodes}/{len(events)}")
            print("  ms/call   launches/call   registers/thread   shared bytes/block   kernel")
            rows = db.execute(
                f"SELECT value, count(*), sum(end-start), min(registersPerThread), "
                f"max(registersPerThread), min(staticSharedMemory+dynamicSharedMemory), "
                f"max(staticSharedMemory+dynamicSharedMemory) FROM {table} "
                "JOIN StringIds ON demangledName=StringIds.id WHERE start>=? AND end<=? "
                "GROUP BY value ORDER BY sum(end-start) DESC LIMIT ?", (start, end, top),
            )
            for kernel, calls, duration, low_reg, high_reg, low_shared, high_shared in rows:
                print(f"  {duration / count / 1e6:7.3f}   {calls / count:13g}   "
                      f"{low_reg}-{high_reg:<14}   {low_shared}-{high_shared:<16}   {kernel}")
        print("\nDiagnostic durations only. Summed intervals may overlap; host waits are elapsed time.")
        print("Do not subtract these quantities to estimate busy CPU time or removable barrier cost.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    report(args.database, args.top)


if __name__ == "__main__":
    main()
