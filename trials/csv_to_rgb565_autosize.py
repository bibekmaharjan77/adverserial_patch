#!/usr/bin/env python3
"""
csv_to_rgb565_autosize.py
Auto-detect width/height from a DSView CSV/XLSX DVP capture and reconstruct RGB565.

- we give PCLK + HREF + 8 data lines (mapping can be numeric strings like "0", "5", …).
- It scans the whole file, collects byte counts per HREF-high segment (row),
  tries both byte alignments (offset 0/1) and both byte orders (msb/lsb first),
  builds a histogram of pixels/row, chooses the most likely width,
  sets height = number of rows, and writes the image.
- Flags let us override polarity/edge/bit-order if needed.

Example (our mapping discovered earlier):
  python3 csv_to_rgb565_autosize.py \
    --in "/path/to/640x480.dsl-la-251016-230216.csv" \
    --map PCLK=0 HREF=5 D0=1 D1=2 D2=3 D3=4 D4=6 D5=8 D6=9 D7=10 \
    --pclk_edge rising --href_active high \
    --out autosized.png
"""

import argparse, csv, sys, math, collections
from typing import List, Dict, Tuple
from PIL import Image

def is_high(x) -> int:
    s = str(x).strip()
    return 0 if s in ('0', '', '0.0') else 1

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

def norm_headers(headers: List[str], mapping: Dict[str,str]) -> Dict[str,str]:
    lower = {h.lower(): h for h in headers}
    out = {}
    for k, v in mapping.items():
        if v in headers:
            out[k] = v
        elif v.lower() in lower:
            out[k] = lower[v.lower()]
        else:
            raise KeyError(f"Column '{v}' not found. Headers: {headers}")
    return out

def parse_map(pairs) -> Dict[str,str]:
    m = {}
    for p in pairs:
        k, v = p.split('=', 1)
        m[k.strip().upper()] = v.strip()
    need = {'PCLK','HREF','D0','D1','D2','D3','D4','D5','D6','D7'}
    miss = need - set(m.keys())
    if miss: raise ValueError(f"Missing in --map: {sorted(miss)}")
    return m

def build_row_bytes(path, fmap, pclk_edge, href_active, bit_order, invert_data):
    """Yield list-of-bytes for each HREF-high segment."""
    sample_on = 1 if pclk_edge == 'rising' else 0
    href_act  = 1 if href_active == 'high' else 0
    reversed_bits = (bit_order == 'reversed')

    prev_pclk = prev_href = 0
    row_bytes: List[int] = []

    for headers, row in load_iter_rows(path):
        pclk = is_high(row[fmap['PCLK']])
        hr   = is_high(row[fmap['HREF']])

        active_row = (hr == href_act)
        edge_ok = (pclk != prev_pclk) and (pclk == sample_on)

        if active_row and edge_ok:
            order = (['D0','D1','D2','D3','D4','D5','D6','D7'] if not reversed_bits
                     else ['D7','D6','D5','D4','D3','D2','D1','D0'])
            v = 0
            for j, key in enumerate(order):
                b = is_high(row[fmap[key]])
                if invert_data: b ^= 1
                v |= (b << j)
            row_bytes.append(v)

        # End-of-row on HREF falling edge
        if prev_href == href_act and hr != href_act:
            yield row_bytes
            row_bytes = []

        prev_pclk, prev_href = pclk, hr

    # flush if last row didn't close
    if row_bytes:
        yield row_bytes

def bytes_to_pixels_exact(bb: List[int], byte_order: str, offset: int) -> List[Tuple[int,int,int]]:
    b = bb[offset:]
    if len(b) % 2 == 1:
        b = b[:-1]
    pix = []
    for i in range(0, len(b), 2):
        if byte_order == 'msb_first':
            msb, lsb = b[i], b[i+1]
        else:
            lsb, msb = b[i], b[i+1]
        pix.append(to_rgb565(msb, lsb))
    return pix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--map', nargs='+', required=True)
    ap.add_argument('--pclk_edge', choices=['rising','falling'], default='rising')
    ap.add_argument('--href_active', choices=['high','low'], default='high')
    ap.add_argument('--byte_order', choices=['msb_first','lsb_first'], default=None, help='If omitted, try both')
    ap.add_argument('--bit_order', choices=['normal','reversed'], default='normal')
    ap.add_argument('--invert_data', action='store_true')
    ap.add_argument('--out', default='autosized.png')
    args = ap.parse_args()

    mapping = parse_map(args.map)

    # Prime and map headers
    try:
        headers, _ = next(load_iter_rows(args.inp))
    except StopIteration:
        print("Input empty."); sys.exit(2)
    fmap = norm_headers(headers, mapping)

    # Pass 1: collect per-row byte counts; try offsets + byte orders to estimate width.
    hist = collections.Counter()
    rows_bytes = list(build_row_bytes(args.inp, fmap, args.pclk_edge, args.href_active, args.bit_order, args.invert_data))
    if not rows_bytes:
        print("No HREF-high segments found. Check HREF mapping/polarity."); sys.exit(3)

    # Try both byte orders if not specified
    byte_orders = [args.byte_order] if args.byte_order else ['msb_first','lsb_first']
    best_choice = None  # (width, byte_order, offset)
    for bo in byte_orders:
        for offset in (0,1):
            for rb in rows_bytes:
                pix = bytes_to_pixels_exact(rb, bo, offset)
                hist[len(pix)] += 1
            # top width candidate
            if hist:
                width_candidate, count = hist.most_common(1)[0]
                if best_choice is None or count > best_choice[3]:
                    best_choice = (width_candidate, bo, offset, count)
    if best_choice is None or best_choice[0] <= 1:
        print("Could not infer width. Try toggling polarity/bit order, or provide known width/height.")
        sys.exit(4)

    width, chosen_bo, offset, votes = best_choice
    height = len(rows_bytes)

    print(f"Inferred width ≈ {width} pixels (votes={votes}), height ≈ {height} rows")
    print(f"Using byte_order={chosen_bo}, byte_offset={offset}, pclk_edge={args.pclk_edge}, href_active={args.href_active}, bit_order={args.bit_order}")

    # Pass 2: build image using inferred params; pad/crop uniformly.
    rows_pixels = []
    for rb in rows_bytes:
        pix = bytes_to_pixels_exact(rb, chosen_bo, offset)
        if len(pix) < width:
            pix = pix + [(0,0,0)]*(width - len(pix))
        else:
            # center-trim to width
            start = max(0, (len(pix)-width)//2)
            pix = pix[start:start+width]
        rows_pixels.append(pix)

    img = Image.new('RGB', (width, height))
    for y in range(height):
        row = rows_pixels[y] if y < len(rows_pixels) else [(0,0,0)]*width
        for x in range(width):
            img.putpixel((x,y), row[x])
    img.save(args.out)
    print(f"Saved {args.out} ({width}x{height}).")
    print("Tip: If colors look off, rerun with --bit_order reversed or --invert_data; if geometry looks wrong, try --href_active low or --pclk_edge falling.")

if __name__ == '__main__':
    main()
