#!/usr/bin/env python3
"""Rank measured CUDA kernels or Vulkan submissions in an Nsight SQLite export."""

import argparse
from pathlib import Path
import sqlite3


def report(database, top, launches=False, setup=False):
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        ranges = db.execute(
            "SELECT start, end, coalesce(text, value) FROM NVTX_EVENTS "
            "LEFT JOIN StringIds ON textId=StringIds.id ORDER BY start"
        ).fetchall()
        phases = [row for row in ranges if row[2] and row[2].endswith("/measure")]
        if not phases:
            raise ValueError("no measurement ranges; API-only traces are insufficient")
        if setup:
            preparation = [(a, b) for a, b, name in ranges if name == "meganeura/parameter_preparation"]
            if not preparation or any(b is None for _, b in preparation):
                raise ValueError("setup reporting needs complete native parameter-preparation ranges")
            for index, (a, b) in enumerate(preparation, 1):
                print(f"\nParameter preparation {index}: {(b-a)/1e6:.3f} ms host elapsed")
                for label in ("tensor_preparation", "parameter_upload"):
                    spans = [(x, y) for x, y, name in ranges
                             if name == f"meganeura/{label}" and y is not None and a <= x < y <= b]
                    if spans:
                        print(f"  {label}: {len(spans)} calls, {sum(y-x for x,y in spans)/1e6:.3f} ms host elapsed")
                    else:
                        print(f"  {label}: not captured")
                print("  Vulkan API calls, summed host ms, name (nested in preparation):")
                for count, elapsed, name in db.execute(
                    "SELECT count(*), sum(end-start)/1e6, value FROM VULKAN_API "
                    "JOIN StringIds ON nameId=StringIds.id WHERE start>=? AND end<=? "
                    "GROUP BY nameId ORDER BY sum(end-start) DESC LIMIT ?", (a, b, top),
                ):
                    print(f"  {count:6} {elapsed:10.3f} {name}")
                count, elapsed = db.execute(
                    "SELECT count(*), coalesce(sum(end-start), 0)/1e6 FROM VULKAN_WORKLOAD "
                    "WHERE start>=? AND end<=?", (a, b),
                ).fetchone()
                print(f"  {count} grouped GPU intervals, summed {elapsed:.3f} ms (not additive to host)")
        for start, end, name in phases:
            if end is None:
                raise ValueError(f"{name}: incomplete measurement range")
            engine = name.split("/")[0]
            samples = [(a, b) for a, b, label in ranges
                       if label in (f"{engine}/sample", name.removesuffix("/measure") + "/sample")
                       and b is not None and start <= a < b <= end]
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
            per_call = [[(x, y) for x, y in events if a <= x < y <= b] for a, b in samples]
            if any(not part for part in per_call) or sum(map(len, per_call)) != len(events):
                raise ValueError(f"{name}: GPU events missing or outside complete samples")
            print(f"  {len(events) / count:g} GPU intervals/call; summed duration "
                  f"{sum(b-a for a,b in events) / count / 1e6:.3f} ms/call")
            spans = [max(b for _, b in part) - min(a for a, _ in part) for part in per_call]
            print(f"  First observed GPU start to last end: {sum(spans) / count / 1e6:.3f} ms/call")
            print("  This span includes inter-event gaps, not host time before/after GPU work.")
            if engine == "meganeura":
                print("  Vulkan intervals are grouped submissions, not individual shaders.")
                continue
            graph_nodes = db.execute(
                f"SELECT count(*) FROM {table} WHERE start>=? AND end<=? AND graphNodeId>0",
                (start, end),
            ).fetchone()[0]
            print(f"  CUDA graph-node events: {graph_nodes}/{len(events)}")
            print("  CUDA spans cover kernel events only; memcpy/memset events are not included.")
            if launches and graph_nodes != len(events):
                raise ValueError("launch-level reporting requires captured CUDA graph nodes")
            print("  ms/call   launches/call   registers/thread   shared bytes/block   kernel")
            grouping = "value, graphId, graphNodeId" if launches else "value"
            rows = db.execute(
                f"SELECT value, count(*), sum(end-start), min(registersPerThread), "
                f"max(registersPerThread), min(staticSharedMemory+dynamicSharedMemory), "
                f"max(staticSharedMemory+dynamicSharedMemory), min(graphId), min(graphNodeId), "
                "min(gridX*gridY*gridZ), max(gridX*gridY*gridZ), "
                "min(blockX*blockY*blockZ), max(blockX*blockY*blockZ), "
                "group_concat(DISTINCT gridX||'x'||gridY||'x'||gridZ), "
                "group_concat(DISTINCT blockX||'x'||blockY||'x'||blockZ), "
                f"max(localMemoryPerThread) FROM {table} "
                "JOIN StringIds ON demangledName=StringIds.id WHERE start>=? AND end<=? "
                f"GROUP BY {grouping} ORDER BY sum(end-start) DESC LIMIT ?", (start, end, top),
            )
            for (kernel, calls, duration, low_reg, high_reg, low_shared, high_shared,
                 graph, node, low_blocks, high_blocks, low_threads, high_threads,
                 grid, block, local_bytes) in rows:
                print(f"  {duration / count / 1e6:7.3f}   {calls / count:13g}   "
                      f"{low_reg}-{high_reg:<14}   {low_shared}-{high_shared:<16}   {kernel}")
                geometry = (f"graph {graph}, node {node}; grid {grid}, block {block}" if launches
                            else f"blocks/launch {low_blocks}-{high_blocks}; "
                                 f"threads/block {low_threads}-{high_threads}")
                print(f"    {geometry}; max reported local bytes/thread {local_bytes}")
        print("\nDiagnostic durations only. Summed intervals may overlap; host waits are elapsed time.")
        print("Do not subtract these quantities to estimate busy CPU time or removable barrier cost.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--launches", action="store_true", help="rank CUDA graph nodes separately")
    parser.add_argument("--setup", action="store_true", help="native parameter preparation and Vulkan allocation calls")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    report(args.database, args.top, args.launches, args.setup)


if __name__ == "__main__":
    main()
