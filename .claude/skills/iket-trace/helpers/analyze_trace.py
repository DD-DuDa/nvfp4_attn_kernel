#!/usr/bin/env python3
"""Turn an IKET trace.json into warp-role critical path and wait attribution.

The Perfetto timeline is for orientation; this is for conclusions. Output answers
three questions: which warp role is the critical path, where each role spends its
life, and whether work is spread across the SMs.

Usage:
    analyze_trace.py TRACE.json
    analyze_trace.py TRACE.json --roles "0-3=softmax0,12=mma,14=load"
    analyze_trace.py TRACE.json --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_roles(spec: str) -> dict[int, str]:
    """Parse "0-3=softmax0,12=mma" into {0: 'softmax0', ..., 12: 'mma'}."""
    roles: dict[int, str] = {}
    for clause in filter(None, (c.strip() for c in spec.split(","))):
        if "=" not in clause:
            raise ValueError(f"bad --roles clause {clause!r}, expected WARPS=NAME")
        warps, name = clause.split("=", 1)
        for part in warps.split("+"):
            part = part.strip()
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                span = range(lo, hi + 1)
            else:
                span = range(int(part), int(part) + 1)
            for w in span:
                roles[w] = name.strip()
    return roles


def fmt_ns(ns: float) -> str:
    if ns >= 1e6:
        return f"{ns / 1e6:.3f}ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f}us"
    return f"{ns:.0f}ns"


def table(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return "  (none)\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        out.append("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
    return "\n".join(out) + "\n"


class LaunchReport:
    def __init__(self, launch: dict, loc_table: list[dict], strings: list[str],
                 roles: dict[int, str]) -> None:
        self.launch = launch
        self.loc = loc_table
        self.strings = strings
        self.roles = roles

    def role_of(self, loc_idx: int) -> str:
        warp = self.loc[loc_idx]["warpId"]
        return self.roles.get(warp, f"warp{warp}")

    def lifetimes_by_role(self) -> dict[str, list[int]]:
        by_role: dict[str, list[int]] = defaultdict(list)
        for wl in self.launch.get("warpLifetimes", []):
            by_role[self.role_of(wl["locIdx"])].append(wl["endTs"] - wl["startTs"])
        return by_role

    def role_life_total(self) -> dict[str, int]:
        return {r: sum(v) for r, v in self.lifetimes_by_role().items()}

    def ranges_by_role(self) -> dict[tuple[str, str], list[int]]:
        acc: dict[tuple[str, str], list[int]] = defaultdict(list)
        for rng in self.launch.get("ranges", []):
            locs = rng.get("warpLocIdxs") or []
            if not locs:
                continue
            name = self.strings[rng["rangeNameIdx"]]
            acc[(self.role_of(locs[0]), name)].append(rng["endTs"] - rng["startTs"])
        return acc

    def markers_by_role(self) -> dict[tuple[str, str], int]:
        acc: dict[tuple[str, str], int] = defaultdict(int)
        for mk in self.launch.get("markers", []):
            acc[(self.role_of(mk["locIdx"]), self.strings[mk["markerNameIdx"]])] += 1
        return acc

    def malformed_ranges(self) -> int:
        bad = 0
        for rng in self.launch.get("ranges", []):
            locs = rng.get("warpLocIdxs") or []
            if rng["endTs"] < rng["startTs"]:
                bad += 1
            elif len(locs) >= 2 and locs[0] != locs[1]:
                bad += 1
        return bad

    def occupancy(self) -> dict[str, Any]:
        sms: set[int] = set()
        ctas: set[tuple[int, ...]] = set()
        cta_per_sm: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        touched = {wl["locIdx"] for wl in self.launch.get("warpLifetimes", [])}
        if not touched:
            touched = set(range(len(self.loc)))
        for idx in touched:
            entry = self.loc[idx]
            cta = tuple(entry["ctaId"])
            sms.add(entry["smId"])
            ctas.add(cta)
            cta_per_sm[entry["smId"]].add(cta)
        grid = (self.launch.get("gridDimX", 1), self.launch.get("gridDimY", 1),
                self.launch.get("gridDimZ", 1))
        return {
            "grid": grid,
            "grid_ctas": grid[0] * grid[1] * grid[2],
            "traced_ctas": len(ctas),
            "sms_used": len(sms),
            "max_ctas_per_sm": max((len(v) for v in cta_per_sm.values()), default=0),
        }

    def wall(self) -> int:
        wls = self.launch.get("warpLifetimes", [])
        if not wls:
            return 0
        return max(w["endTs"] for w in wls) - min(w["startTs"] for w in wls)

    def to_dict(self) -> dict[str, Any]:
        life = self.role_life_total()
        longest = max(life.values(), default=0)
        return {
            "kernel": self.launch.get("kernelName"),
            "wall_ns": self.wall(),
            "occupancy": self.occupancy(),
            "malformed_ranges": self.malformed_ranges(),
            "roles": {
                role: {
                    "warps": len(self.lifetimes_by_role()[role]),
                    "total_ns": total,
                    "mean_ns": statistics.mean(self.lifetimes_by_role()[role]),
                    "max_ns": max(self.lifetimes_by_role()[role]),
                    "pct_of_critical": 100.0 * total / longest if longest else 0.0,
                }
                for role, total in sorted(life.items(), key=lambda kv: -kv[1])
            },
            "ranges": [
                {
                    "role": role,
                    "name": name,
                    "count": len(durs),
                    "total_ns": sum(durs),
                    "mean_ns": statistics.mean(durs),
                    "pct_of_role_life": (100.0 * sum(durs) / life[role]) if life.get(role) else 0.0,
                }
                for (role, name), durs in sorted(
                    self.ranges_by_role().items(), key=lambda kv: -sum(kv[1])
                )
            ],
        }

    def render(self, top: int) -> str:
        d = self.to_dict()
        occ = d["occupancy"]
        out = [
            f"kernel   {d['kernel']}",
            f"grid     {occ['grid']}  ->  {occ['grid_ctas']} CTAs, {occ['traced_ctas']} traced",
            f"spread   {occ['sms_used']} SMs used, max {occ['max_ctas_per_sm']} CTA/SM",
            f"wall     {fmt_ns(d['wall_ns'])} (first warp start to last warp end)",
            "",
        ]
        if d["malformed_ranges"]:
            out.append(
                f"WARNING  {d['malformed_ranges']} malformed ranges "
                f"(negative duration or crossing warps) -- trace is suspect\n"
            )

        out.append("WARP LIFETIME BY ROLE  (longest = critical path)")
        rows = []
        for role, r in d["roles"].items():
            mark = "  <== critical" if abs(r["pct_of_critical"] - 100.0) < 1e-9 else ""
            rows.append([
                role, str(r["warps"]), fmt_ns(r["total_ns"]),
                fmt_ns(r["mean_ns"]), fmt_ns(r["max_ns"]),
                f"{r['pct_of_critical']:.1f}%" + mark,
            ])
        out.append(table(rows, ["role", "warps", "total", "mean", "max", "vs critical"]))

        out.append("RANGE TIME  (pct is of that role's total warp lifetime)")
        rows = []
        for r in d["ranges"][:top]:
            rows.append([
                r["role"], r["name"], str(r["count"]),
                fmt_ns(r["total_ns"]), fmt_ns(r["mean_ns"]),
                f"{r['pct_of_role_life']:.1f}%",
            ])
        out.append(table(rows, ["role", "range", "count", "total", "mean", "pct"]))

        mk = self.markers_by_role()
        if mk:
            out.append("MARKERS")
            rows = [[role, name, str(n)]
                    for (role, name), n in sorted(mk.items(), key=lambda kv: -kv[1])[:top]]
            out.append(table(rows, ["role", "marker", "count"]))

        out.append(
            "Nested ranges overlap, so percentages may exceed 100%. A role whose wait\n"
            "range dominates its lifetime is starved by its producer; a producer that\n"
            "waits is blocked by slow consumers."
        )
        return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path, help="path to iket_pid_*.trace.json")
    ap.add_argument("--roles", default="",
                    help='warp-to-role map, e.g. "0-3=softmax0,12=mma,14=load"')
    ap.add_argument("--top", type=int, default=25, help="max rows per table")
    ap.add_argument("--launch", type=int, default=None, help="only this launch index")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit machine-readable output")
    args = ap.parse_args()

    if not args.trace.is_file():
        print(f"no such trace: {args.trace}", file=sys.stderr)
        return 2

    data = json.loads(args.trace.read_text())
    launches = data.get("launches", [])
    if not launches:
        print("trace contains no launches -- the kernel probably did not JIT-compile\n"
              "inside the profiled run (see reference/04-pitfalls.md)", file=sys.stderr)
        return 1

    roles = parse_roles(args.roles) if args.roles else {}
    loc = data.get("locationTable", [])
    strings = data.get("stringTable", [])

    selected = ([launches[args.launch]] if args.launch is not None else launches)
    reports = [LaunchReport(l, loc, strings, roles) for l in selected]

    if args.as_json:
        json.dump([r.to_dict() for r in reports], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    for i, rep in enumerate(reports):
        if len(reports) > 1:
            print(f"{'=' * 72}\nLAUNCH {i}\n{'=' * 72}")
        print(rep.render(args.top))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
