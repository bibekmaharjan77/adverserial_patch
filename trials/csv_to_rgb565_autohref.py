#!/usr/bin/env python3
"""
csv_to_rgb565_autohref.py

Reconstruct an RGB565 image from a DSView CSV/XLSX of a DVP camera, when we
know PCLK but are unsure which column is HREF/VSYNC. This script:
- Takes a fixed PCLK column and a *list* of HREF candidates.
- Optionally a list of data columns (exactly 8) — otherwise it will try a best-guess set.
- Brute-forces common PCLK/HREF polarities, byte order, and bit order.
- Scores each combo by how many rows produce ~width pixels and saves the best image.

Usage (based on our probe results):
  python3 csv_to_rgb565_autohref.py \
    --in "/path/to/640x480.dsl-la-251016-230216.csv" \
    --width 640 --height 480 \
    --pclk 0 \
    --href_candidates 4 1 8 9 6 2 10 3 5 \
    --data 1 2 3 4 6 8 9 10 \
    --out best_autohref.png
"""

import argparse, csv, sys, itertools
from typing import List, Dict, Tuple
from PIL import Image

def is_high(x) -> int:
    s = str(x).strip()
    return 0 if s in ('0','','0.0') else 1

def to_rgb565(msb, lsb):
    r = ((msb & 0xF8) >> 3) * 255 // 31
    g = (((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)) * 255 // 63
    b = (lsb & 0x1F) * 255 // 31
    return (r, g, b)

def load_iter_rows(path: str):
    if path.lower().endswith('.xlsx'):
        import pandas as pd
        df = pd.read_excel(path, engine='openpyxl')
        headers = list(df.columns)
        for _, row in df.iterrows():
            yield headers, {h: row[h] for h in headers}
    else:
        f = open(path, newline='')
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames
        for row in rdr:
            yield headers, row

def normalize_headers(headers: List[str], cols: List[str]) -> List[str]:
    lower = {h.lower(): h for h in headers}
    out = []
    for v in cols:
        if v in headers:
            out.append(v)
        elif v.lower() in lower:
            out.append(lower[v.lower()])
        else:
            raise KeyError(f"Column '{v}' not found. Headers: {headers}")
    return out

def sample_frame(path: str, width: int, height: int,
                 pclk_col: str, href_col: str, data_cols: List[str],
                 pclk_edge: str, href_active: str, byte_order: str, bit_order: str,
                 max_rows:int=1000000) -> Tuple[int, List[List[tuple]]]:
    """Return (score, image_rows). Score is how many rows matched width exactly."""
    rows_built = 0
    exact_rows = 0
    prev_pclk = prev_href = 0
    byte_buf: List[int] = []
    row_pixels: List[tuple] = []
    image_rows: List[List[tuple]] = []

    href_act = 1 if href_active == 'high' else 0
    sample_on = 1 if pclk_edge == 'rising' else 0
    reversed_bits = (bit_order == 'reversed')

    for headers, row in load_iter_rows(path):
        # Stream; stop if we already collected a full frame
        if rows_built >= height:
            break

        pclk = is_high(row[pclk_col])
        hr   = is_high(row[href_col])

        active_row = (hr == href_act)
        edge_ok = (pclk != prev_pclk) and (pclk == sample_on)

        if active_row and edge_ok:
            # Build byte from data_cols
            if len(data_cols) != 8:
                return 0, []
            ordered = (data_cols if not reversed_bits else list(reversed(data_cols)))
            val = 0
            for j, key in enumerate(ordered):
                b = is_high(row[key])
                val |= (b << j)
            byte_buf.append(val)
            if len(byte_buf) == 2:
                if byte_order == 'msb_first':
                    msb, lsb = byte_buf[0], byte_buf[1]
                else:
                    lsb, msb = byte_buf[0], byte_buf[1]
                byte_buf.clear()
                row_pixels.append(to_rgb565(msb, lsb))
                if len(row_pixels) == width:
                    image_rows.append(row_pixels)
                    row_pixels = []
                    rows_built += 1
                    exact_rows += 1

        # row end when HREF deasserts
        if prev_href == href_act and hr != href_act:
            if row_pixels:
                # Pad/truncate and count if width matched
                if len(row_pixels) == width:
                    exact_rows += 1
                row_pixels = []

        prev_pclk, prev_href = pclk, hr

    return exact_rows, image_rows

def save_image(rows: List[List[tuple]], width: int, height: int, path: str):
    while len(rows) < height:
        rows.append([(0,0,0)]*width)
    rows = rows[:height]
    img = Image.new('RGB', (width, height))
    for y in range(height):
        rp = rows[y] if y < len(rows) else [(0,0,0)]*width
        if len(rp) < width:
            rp = rp + [(0,0,0)]*(width - len(rp))
        for x in range(width):
            img.putpixel((x,y), rp[x])
    img.save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--width', type=int, required=True)
    ap.add_argument('--height', type=int, required=True)
    ap.add_argument('--pclk', required=True, help='PCLK column name/number')
    ap.add_argument('--href_candidates', nargs='+', required=True, help='List of columns to try as HREF')
    ap.add_argument('--data', nargs='+', help='Exactly 8 data columns (order assumed D0..D7). If omitted, guesses will be used.')
    ap.add_argument('--out', default='best_autohref.png')
    args = ap.parse_args()

    # Prime headers
    headers, first_row = next(load_iter_rows(args.inp))
    pclk = normalize_headers(headers, [args.pclk])[0]
    hrefs = normalize_headers(headers, args.href_candidates)

    # Choose data set
    if args.data:
        data_cols = normalize_headers(headers, args.data)
        if len(data_cols) != 8:
            print("ERROR: --data must list exactly 8 columns.")
            sys.exit(2)
        data_sets = [data_cols]
    else:
        # Guess: use the 8 most "active" excluding pclk and href candidates (simple heuristic)
        active = [h for h in headers if h.lower() != 'time(s)' and h not in [pclk] + hrefs]
        # just pick first 8 for now (user can provide --data to be precise)
        data_sets = [active[:8]]

    best = None
    for href in hrefs:
        for data_cols in data_sets:
            for pclk_edge in ['rising','falling']:
                for href_active in ['high','low']:
                    for byte_order in ['msb_first','lsb_first']:
                        for bit_order in ['normal','reversed']:
                            score, rows = sample_frame(
                                args.inp, args.width, args.height,
                                pclk, href, data_cols,
                                pclk_edge, href_active, byte_order, bit_order
                            )
                            print(f"href={href} data={data_cols} pclk_edge={pclk_edge} href_active={href_active} byte_order={byte_order} bit_order={bit_order} -> rows_exact={score}")
                            if best is None or score > best[0]:
                                best = (score, href, tuple(data_cols), pclk_edge, href_active, byte_order, bit_order, rows)

    if best is None or best[0] == 0:
        print("No combination produced valid rows. Re-check candidates or provide explicit --data in correct order.")
        sys.exit(3)

    score, href, data_cols, pclk_edge, href_active, byte_order, bit_order, rows = best
    print("\nBEST:")
    print(f"  href={href}")
    print(f"  data={list(data_cols)}")
    print(f"  pclk_edge={pclk_edge}  href_active={href_active}")
    print(f"  byte_order={byte_order} bit_order={bit_order}")
    print(f"  rows_exact={score}/{args.height}")

    save_image(rows, args.width, args.height, args.out)
    print("Saved:", args.out)

if __name__ == '__main__':
    main()
