"""Print one CTA's IKET ranges in time order.

`analyze_trace.py` aggregates; this shows the sequence, which is what is needed
to tell "a role is slow" from "a role starts late and then waits".

Usage:
  python tests/kernel_profile/iket_cta_timeline.py TRACE.json \
      --launch 1 --cta 0 --warps 0,12,13,8,14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--launch", type=int, default=-1)
    parser.add_argument("--cta", type=int, default=0)
    parser.add_argument("--warps", type=str, default="0,4,8,12,13,14,15")
    parser.add_argument("--max-rows", type=int, default=400)
    args = parser.parse_args()

    trace = json.loads(Path(args.trace).read_text())
    strings = trace["stringTable"]
    locations = trace["locationTable"]
    launch = trace["launches"][args.launch]
    wanted = {int(value) for value in args.warps.split(",")}

    def location_key(loc_idx: int):
        loc = locations[loc_idx]
        return tuple(loc["ctaId"]), loc["warpId"], loc["smId"]

    origin = min(item["startTs"] for item in launch["warpLifetimes"])

    print(
        f"launch {args.launch} {launch['kernelName'][:60]}"
        f" grid=({launch['gridDimX']},{launch['gridDimY']},{launch['gridDimZ']})"
    )
    lifetimes = {}
    for item in launch["warpLifetimes"]:
        cta, warp, sm = location_key(item["locIdx"])
        if cta[0] != args.cta or warp not in wanted:
            continue
        lifetimes[warp] = (
            (item["startTs"] - origin) / 1e3,
            (item["endTs"] - origin) / 1e3,
            sm,
        )
    print("\nwarp lifetimes on cta", args.cta)
    for warp in sorted(lifetimes):
        start, end, sm = lifetimes[warp]
        print(
            f"  warp{warp:<3} sm{sm:<4} {start:9.3f} -> {end:9.3f} us"
            f"   ({end - start:8.3f} us)"
        )

    rows = []
    for item in launch["ranges"]:
        cta, warp, _ = location_key(item["warpLocIdxs"][0])
        if cta[0] != args.cta or warp not in wanted:
            continue
        rows.append(
            (
                (item["startTs"] - origin) / 1e3,
                (item["endTs"] - origin) / 1e3,
                warp,
                strings[item["rangeNameIdx"]],
            )
        )
    rows.sort()
    print("\nranges in time order (us from first warp start)")
    for start, end, warp, name in rows[: args.max_rows]:
        print(
            f"  {start:9.3f} -> {end:9.3f}  ({end - start:8.3f})"
            f"  warp{warp:<3} {name}"
        )
    if len(rows) > args.max_rows:
        print(f"  ... {len(rows) - args.max_rows} more")


if __name__ == "__main__":
    main()
