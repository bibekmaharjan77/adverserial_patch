#!/usr/bin/env python3
import csv, sys, math, argparse
import statistics as stats

def is_high(v):
    s = str(v).strip()
    return 0 if s in ("0","0.0","") else 1

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--rows", type=int, default=200000, help="rows to scan")
args = ap.parse_args()

with open(args.csv, newline="") as f:
    rdr = csv.DictReader(f)
    headers = rdr.fieldnames
    print("Headers:", headers)

    # Load a chunk
    rows = []
    for i, row in enumerate(rdr):
        rows.append(row)
        if i+1 >= args.rows: break
    print(f"Scanned rows: {len(rows)}")

    # Compute flips & duty cycle for each column (except time)
    cols = [h for h in headers if h.lower() != "time(s)"]
    stats_table = []
    for c in cols:
        vals = [is_high(r[c]) for r in rows]
        if len(vals) < 2:
            flips = 0
        else:
            flips = sum(1 for a,b in zip(vals, vals[1:]) if a!=b)
        ones = sum(vals)
        ratio = ones / max(1,len(vals))
        stats_table.append((c, flips, ratio))
    stats_table.sort(key=lambda x: x[1], reverse=True)

    print("\nTop by flips (likely PCLK first):")
    for c, flips, ratio in stats_table[:10]:
        print(f"  {c:>8}  flips={flips:7d}  high_ratio={ratio:0.3f}")

    # Heuristics
    if stats_table:
        pclk_guess = stats_table[0][0]
        print(f"\nPCLK candidate: {pclk_guess}")

    print("\nColumns with moderate flips and ~line-like duty (0.05–0.95):")
    for c, flips, ratio in stats_table:
        if 0.05 < ratio < 0.95 and flips>0 and c!=stats_table[0][0]:
            print(f"  {c:>8}  flips={flips:7d}  high_ratio={ratio:0.3f}")

    print("\nColumns stuck mostly low or mostly high (possible VSYNC/HREF if pulsing rarely):")
    for c, flips, ratio in stats_table:
        if flips>0 and (ratio<0.02 or ratio>0.98):
            print(f"  {c:>8}  flips={flips:7d}  high_ratio={ratio:0.3f}  (rare pulses)")
