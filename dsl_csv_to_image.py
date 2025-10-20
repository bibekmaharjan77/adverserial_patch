
#!/usr/bin/env python3
"""
dsl_csv_to_image.py  (user-defaults version)

Convert a DSView/DSL exported CSV of a DVP camera (ESP32-CAM/OV2640 style) to a PNG image.

Defaults (override with CLI flags):
- width x height: 640 x 480
- mapping: PCLK=0, HREF=7, VSYNC=5, D0=6, D1=3, D2=4, D3=1, D4=2, D5=8, D6=9, D7=10
- pclk_edge=rising, vsync_active=high, href_active=high, byte_order=msb_first

Usage:
    python dsl_csv_to_image.py --csv path/to/export.csv --out frame.png

To override mapping (example):
    python dsl_csv_to_image.py --csv export.csv --map PCLK=P0 HREF=P7 VSYNC=P5 D0=P6 D1=P3 D2=P4 D3=P1 D4=P2 D5=P8 D6=P9 D7=P10

"""
import argparse, csv, sys
from typing import Dict
from PIL import Image

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_MAPPING = {
    'PCLK': '0',
    'HREF': '7',
    'VSYNC': '5',
    'D0': '6',
    'D1': '3',
    'D2': '4',
    'D3': '1',
    'D4': '2',
    'D5': '8',
    'D6': '9',
    'D7': '10',
}

def parse_mapping(pairs) -> Dict[str, str]:
    if not pairs:
        return DEFAULT_MAPPING.copy()
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

def bitbus(row, m):
    val = 0
    for i in range(8):
        bit = 1 if row[m[f"D{i}"]].strip() not in ('0','','0.0') else 0
        val |= (bit << i)  # LSB=D0
    return val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    ap.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    ap.add_argument('--map', nargs='+', help='Channel mapping like PCLK=0 HREF=7 VSYNC=5 D0=6 ... D7=10')
    ap.add_argument('--pclk_edge', choices=['rising','falling'], default='rising')
    ap.add_argument('--vsync_active', choices=['high','low'], default='high')
    ap.add_argument('--href_active', choices=['high','low'], default='high')
    ap.add_argument('--byte_order', choices=['msb_first','lsb_first'], default='msb_first')
    ap.add_argument('--out', default='frame.png')
    args = ap.parse_args()

    mapping = parse_mapping(args.map)

    with open(args.csv, newline='') as f:
        rdr = csv.DictReader(f)

        # normalize mapping values to actual header names (case-insensitive match)
        for k, v in list(mapping.items()):
            if v not in rdr.fieldnames:
                cand = [h for h in rdr.fieldnames if h.lower() == v.lower()]
                if cand:
                    mapping[k] = cand[0]

        # basic validation: ensure all mapping headers exist
        missing_headers = [v for v in mapping.values() if v not in rdr.fieldnames]
        if missing_headers:
            print(f"[!] These mapped headers were not found in the CSV: {missing_headers}")
            print(f"    CSV headers: {rdr.fieldnames}")
            sys.exit(2)

        prev_pclk = prev_vsync = prev_href = 0
        def is_high(raw): return 0 if raw.strip() in ('0','','0.0') else 1

        frame_started = False
        row_pixels, image_rows = [], []
        w, h = args.width, args.height

        def to_rgb565(msb, lsb):
            r = ((msb & 0xF8) >> 3) * 255 // 31
            g = (((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)) * 255 // 63
            b = (lsb & 0x1F) * 255 // 31
            return (r, g, b)

        PCLK_sample_on = 1 if args.pclk_edge == 'rising' else 0
        VSYNC_active = 1 if args.vsync_active == 'high' else 0
        HREF_active = 1 if args.href_active == 'high' else 0
        pending = []

        for row in rdr:
            pclk = is_high(row[mapping['PCLK']])
            vs = is_high(row[mapping['VSYNC']])
            hr = is_high(row[mapping['HREF']])

            # frame begins when VSYNC transitions from active to inactive (typical OV2640)
            if prev_vsync == VSYNC_active and vs != VSYNC_active:
                frame_started = True
                image_rows.clear()
                row_pixels.clear()
                pending.clear()

            if frame_started:
                active_row = (hr == HREF_active)
                if active_row and (pclk != prev_pclk) and (pclk == PCLK_sample_on):
                    byte = bitbus(row, mapping)
                    pending.append(byte)
                    if len(pending) == 2:
                        if args.byte_order == 'msb_first':
                            msb, lsb = pending[0], pending[1]
                        else:
                            lsb, msb = pending[0], pending[1]
                        row_pixels.append(to_rgb565(msb, lsb))
                        pending.clear()
                        if len(row_pixels) == w:
                            image_rows.append(row_pixels)
                            row_pixels = []

                # end of a row: HREF deasserts
                if prev_href == HREF_active and hr != HREF_active:
                    if row_pixels:
                        if len(row_pixels) < w:
                            row_pixels += [(0,0,0)] * (w - len(row_pixels))
                        image_rows.append(row_pixels[:w])
                        row_pixels = []

                # frame ends when VSYNC becomes active again
                if prev_vsync != VSYNC_active and vs == VSYNC_active and frame_started:
                    if len(image_rows) < h:
                        for _ in range(h - len(image_rows)):
                            image_rows.append([(0,0,0)]*w)
                    img = Image.new('RGB', (w, h))
                    for y in range(h):
                        rowpix = image_rows[y] if y < len(image_rows) else [(0,0,0)]*w
                        for x in range(w):
                            img.putpixel((x, y), rowpix[x] if x < len(rowpix) else (0,0,0))
                    img.save(args.out)
                    print(f"Saved image to {args.out} ({w}x{h})")
                    return

            prev_pclk, prev_vsync, prev_href = pclk, vs, hr

        print("Reached end of CSV without completing a frame. Try toggling polarity/edge or byte order, and confirm mapping.")
        sys.exit(2)

if __name__ == "__main__":
    main()
