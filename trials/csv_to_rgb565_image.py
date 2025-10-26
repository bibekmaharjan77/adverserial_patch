#!/usr/bin/env python3
"""
csv_to_rgb565_image.py

Reconstruct an image from a DSView logic-analyzer export (CSV or XLSX)
captured from a DVP camera (VSYNC/HREF/PCLK + D0..D7) in RGB565 mode.

- Supports CSV and XLSX inputs (auto-detected by extension).
- we provide a mapping for PCLK/HREF/VSYNC/D0..D7 (column names or numbers).
- Samples one byte on each chosen PCLK edge while HREF is active.
- Pairs two bytes -> one RGB565 pixel (msb_first by default).
- Lets us flip polarity/edge/byte-order/bit-order if the image is black or weird.

Usage example (with your mapping guess, 96×96):
  python csv_to_rgb565_image.py --in capture.csv --width 96 --height 96 \
    --map PCLK=0 HREF=7 VSYNC=5 D0=6 D1=3 D2=4 D3=1 D4=2 D5=8 D6=9 D7=10 \
    --out frame.png

If the result is dark/scrambled, try toggles:
  --byte_order lsb_first   --pclk_edge falling
  --href_active low        --vsync_active low
  --bit_order reversed     --invert_data
"""

import argparse, csv, sys
from typing import Dict, List
from PIL import Image

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input DSView export (.csv or .xlsx)')
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--height', type=int, default=96)
    ap.add_argument('--map', nargs='+', required=True,
                    help='Mapping like PCLK=0 HREF=7 VSYNC=5 D0=6 D1=3 D2=4 D3=1 D4=2 D5=8 D6=9 D7=10')
    ap.add_argument('--pclk_edge', choices=['rising','falling'], default='rising')
    ap.add_argument('--vsync_active', choices=['high','low'], default='high')
    ap.add_argument('--href_active', choices=['high','low'], default='high')
    ap.add_argument('--byte_order', choices=['msb_first','lsb_first'], default='msb_first')
    ap.add_argument('--bit_order', choices=['normal','reversed'], default='normal',
                    help='If reversed, treat D7 as least significant input bit')
    ap.add_argument('--invert_data', action='store_true', help='Invert D0..D7 polarity')
    ap.add_argument('--skip_rows', type=int, default=0, help='Skip first N rows (warm-up region)')
    ap.add_argument('--max_pixels', type=int, default=0, help='Stop after this many pixels (0=no cap)')
    ap.add_argument('--out', default='frame.png')
    return ap.parse_args()

def parse_mapping(pairs) -> Dict[str, str]:
    m = {}
    for p in pairs:
        if '=' not in p:
            raise ValueError(f"Bad mapping '{p}'. Use KEY=COLNAME.")
        k, v = p.split('=', 1)
        m[k.strip().upper()] = v.strip()
    required = {'PCLK','HREF','VSYNC','D0','D1','D2','D3','D4','D5','D6','D7'}
    missing = required - set(m.keys())
    if missing:
        raise ValueError(f"Missing mappings for: {sorted(missing)}")
    return m

def is_high(x) -> int:
    s = str(x).strip()
    return 0 if s in ('0', '', '0.0') else 1

def normalize_headers(headers: List[str], mapping: Dict[str, str]) -> Dict[str, str]:
    # Case-insensitive match; allow numeric names as strings
    fixed = {}
    lower = {h.lower(): h for h in headers}
    for k, v in mapping.items():
        if v in headers:
            fixed[k] = v
        else:
            vv = v.lower()
            if vv in lower:
                fixed[k] = lower[vv]
            else:
                raise KeyError(f"CSV/XLSX has no column '{v}'. Headers: {headers}")
    return fixed

def load_iter_rows(path: str, skip_rows: int):
    if path.lower().endswith('.xlsx'):
        import pandas as pd
        df = pd.read_excel(path, engine='openpyxl')
        if skip_rows:
            df = df.iloc[skip_rows:]
        headers = list(df.columns)
        for _, row in df.iterrows():
            yield headers, {h: row[h] for h in headers}
    else:
        f = open(path, newline='')
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames
        for _ in range(skip_rows):
            try:
                next(rdr)
            except StopIteration:
                break
        for row in rdr:
            yield headers, row

def to_rgb565(msb, lsb):
    r = ((msb & 0xF8) >> 3) * 255 // 31
    g = (((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)) * 255 // 63
    b = (lsb & 0x1F) * 255 // 31
    return (r, g, b)

def main():
    args = parse_args()
    mapping = parse_mapping(args.map)
    width, height = args.width, args.height

    # State
    prev_pclk = prev_vsync = prev_href = 0
    byte_buf: List[int] = []
    row_pixels: List[tuple] = []
    image_rows: List[List[tuple]] = []
    pixels_written = 0

    pclk_sample_on = 1 if args.pclk_edge == 'rising' else 0
    vsync_act = 1 if args.vsync_active == 'high' else 0
    href_act  = 1 if args.href_active == 'high' else 0
    reversed_bits = (args.bit_order == 'reversed')

    frame_started = False
    fmap = None

    # Prime headers and mapping
    first_iter = load_iter_rows(args.inp, args.skip_rows)
    try:
        headers, first_row = next(first_iter)
    except StopIteration:
        print("Input is empty after skipping rows.")
        sys.exit(2)
    fmap = normalize_headers(headers, mapping)

    # Process first row plus the rest
    def row_gen():
        yield headers, first_row
        for h, r in load_iter_rows(args.inp, args.skip_rows+1):
            yield h, r

    for headers, row in row_gen():
        # Remap if DSView changed headers (rare)
        if fmap is None or set(fmap.values()) - set(headers):
            fmap = normalize_headers(headers, mapping)

        pclk = is_high(row[fmap['PCLK']])
        vs   = is_high(row[fmap['VSYNC']])
        hr   = is_high(row[fmap['HREF']])

        # Frame start on VSYNC deassert (typical)
        if prev_vsync == vsync_act and vs != vsync_act:
            frame_started = True
            byte_buf.clear()
            row_pixels.clear()
            image_rows.clear()

        if frame_started:
            active_row = (hr == href_act)
            edge_ok = (pclk != prev_pclk) and (pclk == pclk_sample_on)

            if active_row and edge_ok:
                # Build 8-bit sample from D0..D7
                order = (['D0','D1','D2','D3','D4','D5','D6','D7'] if not reversed_bits
                         else ['D7','D6','D5','D4','D3','D2','D1','D0'])
                val = 0
                for j, key in enumerate(order):
                    b = is_high(row[fmap[key]])
                    if args.invert_data:
                        b ^= 1
                    val |= (b << j)
                byte_buf.append(val)
                if len(byte_buf) == 2:
                    if args.byte_order == 'msb_first':
                        msb, lsb = byte_buf[0], byte_buf[1]
                    else:
                        lsb, msb = byte_buf[0], byte_buf[1]
                    byte_buf.clear()
                    row_pixels.append(to_rgb565(msb, lsb))
                    pixels_written += 1
                    if 0 < args.max_pixels <= pixels_written:
                        break
                    if len(row_pixels) == width:
                        image_rows.append(row_pixels)
                        row_pixels = []

            # Row end when HREF deasserts
            if prev_href == href_act and hr != href_act:
                if row_pixels:
                    if len(row_pixels) < width:
                        row_pixels += [(0,0,0)]*(width - len(row_pixels))
                    image_rows.append(row_pixels[:width])
                    row_pixels = []

            # Frame end when VSYNC asserts
            if prev_vsync != vsync_act and vs == vsync_act and frame_started:
                break

        prev_pclk, prev_vsync, prev_href = pclk, vs, hr

    # Build final image (pad if needed)
    if not image_rows:
        print("No image rows decoded. Try adjusting polarity/edge/byte order/bit order, and verify mapping.")
        sys.exit(3)

    while len(image_rows) < height:
        image_rows.append([(0,0,0)]*width)
    image_rows = image_rows[:height]

    img = Image.new('RGB', (width, height))
    for y in range(height):
        row = image_rows[y]
        if len(row) < width:
            row = row + [(0,0,0)]*(width - len(row))
        for x in range(width):
            img.putpixel((x,y), row[x])
    img.save(args.out)
    print(f"Saved {args.out} ({width}x{height}), rows={len(image_rows)}")
    if pixels_written == 0:
        print("Warning: 0 pixels written; likely wrong polarity/edge/byte order or no data during HREF.")

if __name__ == "__main__":
    main()
