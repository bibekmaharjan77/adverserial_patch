#!/usr/bin/env python3
"""
csv_to_rgb565_row_robust.py

Robust RGB565 reconstruction from DSView CSV/XLSX when rows contain extra/missing bytes.
- Uses PCLK + a single HREF (no VSYNC required).
- Collects ALL bytes while HREF is active, then rescues the row:
  * fixes byte alignment (tries offsets),
  * trims or pads to exactly `width` pixels.
- Much more tolerant than "exact width or discard".

Usage (fill in your mapping and size):
  python3 csv_to_rgb565_row_robust.py \
    --in capture.csv --width 640 --height 480 \
    --map PCLK=0 HREF=5 D0=1 D1=2 D2=3 D3=4 D4=6 D5=8 D6=9 D7=10 \
    --pclk_edge rising --href_active high --byte_order msb_first --bit_order normal \
    --out frame.png
"""

import argparse, csv, sys
from typing import List, Tuple
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

def normalize_headers(headers: List[str], mapping: dict) -> dict:
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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--width', type=int, required=True)
    ap.add_argument('--height', type=int, required=True)
    # Only PCLK, HREF and 8 data lines are needed
    ap.add_argument('--map', nargs='+', required=True,
                    help='PCLK=<col> HREF=<col> D0=<col> ... D7=<col>')
    ap.add_argument('--pclk_edge', choices=['rising','falling'], default='rising')
    ap.add_argument('--href_active', choices=['high','low'], default='high')
    ap.add_argument('--byte_order', choices=['msb_first','lsb_first'], default='msb_first')
    ap.add_argument('--bit_order', choices=['normal','reversed'], default='normal')
    ap.add_argument('--invert_data', action='store_true')
    ap.add_argument('--out', default='frame.png')
    return ap.parse_args()

def parse_mapping(pairs) -> dict:
    m = {}
    for p in pairs:
        k, v = p.split('=', 1)
        m[k.strip().upper()] = v.strip()
    need = {'PCLK','HREF','D0','D1','D2','D3','D4','D5','D6','D7'}
    if not need.issubset(m.keys()):
        missing = need - set(m.keys())
        raise ValueError(f"Missing in --map: {sorted(missing)}")
    return m

def reconstruct(path, fmap, width, height, pclk_edge, href_active, byte_order, bit_order, invert_data):
    sample_on = 1 if pclk_edge == 'rising' else 0
    href_act  = 1 if href_active == 'high' else 0
    reversed_bits = (bit_order == 'reversed')

    prev_pclk = prev_href = 0
    row_bytes: List[int] = []
    rows_pixels: List[List[tuple]] = []

    # helper: turn a list of *bytes* for a row into exactly `width` pixels
    def bytes_to_pixels(bts: List[int]) -> List[tuple]:
        if not bts:
            return []
        # try both alignments (offset 0 or 1) so pixels are paired correctly
        candidates = []
        for offset in (0, 1):
            bb = bts[offset:]
            # If too many bytes, try to center-trim to 2*width
            if len(bb) >= 2*width:
                start = max(0, (len(bb) - 2*width)//2)
                bb = bb[start:start + 2*width]
            # If too few bytes, pad zeros to reach an even length
            if len(bb) % 2 == 1:
                bb = bb[:-1]  # drop last odd byte
            # Form pixels
            pix = []
            for i in range(0, min(len(bb), 2*width), 2):
                if byte_order == 'msb_first':
                    msb, lsb = bb[i], bb[i+1]
                else:
                    lsb, msb = bb[i], bb[i+1]
                pix.append(to_rgb565(msb, lsb))
            # pad/truncate to width
            if len(pix) < width:
                pix += [(0,0,0)]*(width - len(pix))
            else:
                pix = pix[:width]
            candidates.append(pix)

        # choose the alignment with higher color variance (less likely to be wrong)
        import math
        def score(pix):
            if not pix: return 0.0
            gs = [0.3*r+0.59*g+0.11*b for (r,g,b) in pix]
            mu = sum(gs)/len(gs)
            var = sum((x-mu)**2 for x in gs)/len(gs)
            return var
        s0, s1 = score(candidates[0]), score(candidates[1])
        return candidates[0] if s0 >= s1 else candidates[1]

    # streaming decode
    for headers, row in load_iter_rows(path):
        pclk = is_high(row[fmap['PCLK']])
        hr   = is_high(row[fmap['HREF']])

        active_row = (hr == href_act)
        edge_ok = (pclk != prev_pclk) and (pclk == sample_on)

        if active_row and edge_ok:
            # sample one byte from D0..D7
            order = (['D0','D1','D2','D3','D4','D5','D6','D7'] if not reversed_bits
                     else ['D7','D6','D5','D4','D3','D2','D1','D0'])
            val = 0
            for j, key in enumerate(order):
                b = is_high(row[fmap[key]])
                if invert_data: b ^= 1
                val |= (b << j)
            row_bytes.append(val)

        # end-of-row
        if prev_href == href_act and hr != href_act:
            # rescue row into exactly `width` pixels
            px = bytes_to_pixels(row_bytes)
            if px:
                rows_pixels.append(px)
            row_bytes = []
            if len(rows_pixels) >= height:
                break

        prev_pclk, prev_href = pclk, hr

    # pad/crop to requested height
    while len(rows_pixels) < height:
        rows_pixels.append([(0,0,0)]*width)
    rows_pixels = rows_pixels[:height]
    return rows_pixels

def main():
    args = parse_args()
    mapping = parse_mapping(args.map)

    # Prime headers and normalize mapping
    try:
        headers, first_row = next(load_iter_rows(args.inp))
    except StopIteration:
        print("Input empty.")
        sys.exit(2)
    fmap = normalize_headers(headers, mapping)

    rows = reconstruct(
        args.inp, fmap, args.width, args.height,
        args.pclk_edge, args.href_active, args.byte_order, args.bit_order, args.invert_data
    )

    img = Image.new('RGB', (args.width, args.height))
    for y in range(args.height):
        rp = rows[y]
        for x in range(args.width):
            img.putpixel((x,y), rp[x])
    img.save(args.out)
    print(f"Saved {args.out} with {len(rows)} rows.")

if __name__ == '__main__':
    main()
