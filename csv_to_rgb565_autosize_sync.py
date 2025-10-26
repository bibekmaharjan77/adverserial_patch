#!/usr/bin/env python3
"""
csv_to_rgb565_autosize_sync.py
--------------------------------

Purpose
=======
Auto-detect the image width/height from a DSView CSV/XLSX capture of a DVP
camera (PCLK/HREF + D0..D7) and reconstruct an RGB565 image.

What’s new vs. the basic autosizer?
-----------------------------------
Implements *row-level PCLK re-sync* :
  - After each HREF rising edge (row start), we **wait until PCLK has been LOW
    at least once**, then accept sampling **only on the chosen edge** (0→1 for
    rising, or 1→0 for falling). This avoids starting mid-bit when PCLK was
    already high as the row began.

How autosize works
------------------
1) We parse the stream and, for each HREF-high “row”, collect all **bytes**
   (each byte comes from D0..D7 sampled on each valid PCLK edge).
2) For each row, we try both **byte alignments** (offset 0/1) and both **byte
   orders** (msb_first/lsb_first, unless one is forced with --byte_order).
3) We convert byte pairs to RGB565 pixels and build a **histogram** of
   “pixels per row”. The histogram’s peak is the inferred width.
4) Height is simply the number of HREF-high rows observed.
5) We then write the reconstructed image with the inferred width/height.

Options
-------
--in            : DSView-exported CSV or XLSX
--map           : PCLK=<col> HREF=<col> D0=<col> ... D7=<col>  (strings/numbers)
--pclk_edge     : 'rising' or 'falling' (default: rising)
--href_active   : 'high' or 'low' (default: high)
--byte_order    : 'msb_first' or 'lsb_first' (default: try both)
--bit_order     : 'normal' (D0=LSB) or 'reversed' (D7=LSB)
--invert_data   : invert D0..D7 if analyzer polarity is inverted
--no_pclk_gating: DISABLE the row-level gating (not recommended; default is ON)
--out           : output PNG filename

Example (our mapping)
----------------------
python3 csv_to_rgb565_autosize_sync.py \
  --in "/path/to/640x480.dsl-la-251016-230216.csv" \
  --map PCLK=0 HREF=5 D0=1 D1=2 D2=3 D3=4 D4=6 D5=8 D6=9 D7=10 \
  --pclk_edge rising --href_active high \
  --out autosized.png
"""

import argparse, csv, sys, math, collections
from typing import List, Dict, Tuple
from PIL import Image


# ------------------------ Low-level helpers ------------------------

def is_high(x) -> int:
    """
    Normalize DSView cell values to 0/1.

    DSView commonly outputs '0', '1', '0.0', or ''.
    Treat any nonzero / nonempty string as logic-high (1).
    """
    s = str(x).strip()
    return 0 if s in ('0', '', '0.0') else 1


def to_rgb565(msb, lsb):
    """
    Convert a 16-bit RGB565 value (two bytes) into an (R,G,B) 8-bit tuple.

    RGB565 bit layout (msb_first):
      msb = RRRRRGGG (R: 5 bits, G: upper 3 bits)
      lsb = GGGGBBBB (G: lower 3 bits, B: 5 bits)
    """
    r = ((msb & 0xF8) >> 3) * 255 // 31
    g = (((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)) * 255 // 63
    b = (lsb & 0x1F) * 255 // 31
    return (r, g, b)


def load_iter_rows(path: str):
    """
    Yield (headers, row_dict) for each sample row from CSV or XLSX.

    - CSV: streamed with csv.DictReader (low memory).
    - XLSX: read via pandas (simple; higher memory). Prefer CSV for very big files.
    """
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


def norm_headers(headers: List[str], mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Map user-provided column identifiers (numbers or names) to actual headers.
    Supports case-insensitive matches.
    """
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


def parse_map(pairs) -> Dict[str, str]:
    """
    Parse --map arguments like "PCLK=0", "HREF=5", "D0=1" ... "D7=10".
    Ensure all required keys are present.
    """
    m = {}
    for p in pairs:
        k, v = p.split('=', 1)
        m[k.strip().upper()] = v.strip()
    need = {'PCLK','HREF','D0','D1','D2','D3','D4','D5','D6','D7'}
    miss = need - set(m.keys())
    if miss:
        raise ValueError(f"Missing in --map: {sorted(miss)}")
    return m


# ------------------------ Sampling logic ------------------------

def build_row_bytes_sync(
    path: str,
    fmap: Dict[str, str],
    pclk_edge: str,
    href_active: str,
    bit_order: str,
    invert_data: bool,
    use_pclk_gating: bool = True,
):
    """
    Yield a list of *bytes* for each HREF-high row, sampling D0..D7 on PCLK edges.

    Row-level PCLK gating:
    -------------------------------------------
    After HREF rises (row start), if `use_pclk_gating` is True (default):
      - Wait until PCLK has been LOW at least once.
      - Then only sample on the configured edge (0→1 for rising, 1→0 for falling).
    This avoids starting mid-cycle when PCLK is already high at HREF rising.

    Parameters
    ----------
    fmap            : mapping for 'PCLK', 'HREF', and 'D0'..'D7'
    pclk_edge       : 'rising' or 'falling' (which transition to sample)
    href_active     : 'high' or 'low' (HREF polarity)
    bit_order       : 'normal' (D0=LSB) or 'reversed' (D7=LSB)
    invert_data     : True to XOR each data bit (inverted analyzer threshold)
    use_pclk_gating : enable/disable row-level gating (default True)
    """
    sample_on = 1 if pclk_edge == 'rising' else 0
    href_act  = 1 if href_active == 'high' else 0
    reversed_bits = (bit_order == 'reversed')

    prev_pclk = prev_href = 0
    row_bytes: List[int] = []

    # Row-gating state: has PCLK been low since row start?
    pclk_seen_low = True  # set each row; default True for safety if gating disabled

    for headers, row in load_iter_rows(path):
        # Read current logic levels
        pclk = is_high(row[fmap['PCLK']])
        hr   = is_high(row[fmap['HREF']])

        active_row = (hr == href_act)

        # Detect row start (HREF rising to active polarity)
        if prev_href != href_act and hr == href_act:
            row_bytes = []
            # For gating: require a LOW before we accept sampling edges
            pclk_seen_low = (pclk == 0) if use_pclk_gating else True

        # Track PCLK low since row start (enables sampling after a 'true' period low)
        if use_pclk_gating and pclk == 0:
            pclk_seen_low = True

        # Accept sampling only on the desired edge and (if gating) only after a low was seen
        edge_ok = (pclk != prev_pclk) and (pclk == (1 if sample_on == 1 else 0))
        if active_row and edge_ok and (pclk_seen_low or not use_pclk_gating):
            # Pack one 8-bit data sample from D0..D7 in the chosen bit order
            order = (['D0','D1','D2','D3','D4','D5','D6','D7'] if not reversed_bits
                     else ['D7','D6','D5','D4','D3','D2','D1','D0'])
            v = 0
            for j, key in enumerate(order):
                b = is_high(row[fmap[key]])
                if invert_data:
                    b ^= 1
                v |= (b << j)
            row_bytes.append(v)

        # End of row on HREF falling edge → yield all collected bytes
        if prev_href == href_act and hr != href_act:
            yield row_bytes
            row_bytes = []

        prev_pclk, prev_href = pclk, hr

    # Flush last row if HREF never fell at end-of-file
    if row_bytes:
        yield row_bytes


def bytes_to_pixels_exact(bb: List[int], byte_order: str, offset: int) -> List[Tuple[int,int,int]]:
    """
    Turn a raw row of bytes into pixels using a fixed alignment offset (0 or 1)
    and a given byte order (msb_first/lsb_first). Drops the last byte if odd length.
    """
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


# ------------------------ Main (autosize + build) ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='DSView export (.csv or .xlsx)')
    ap.add_argument('--map', nargs='+', required=True, help='PCLK=.. HREF=.. D0=.. .. D7=..')
    ap.add_argument('--pclk_edge', choices=['rising','falling'], default='rising')
    ap.add_argument('--href_active', choices=['high','low'], default='high')
    ap.add_argument('--byte_order', choices=['msb_first','lsb_first'], default=None,
                    help='If omitted, try both during autosize.')
    ap.add_argument('--bit_order', choices=['normal','reversed'], default='normal')
    ap.add_argument('--invert_data', action='store_true', help='Invert D0..D7 bits')
    ap.add_argument('--no_pclk_gating', action='store_true',
                    help='Disable row-level PCLK gating (NOT recommended).')
    ap.add_argument('--out', default='autosized.png')
    args = ap.parse_args()

    mapping = parse_map(args.map)

    # Prime & bind mapping to actual headers
    try:
        headers, _ = next(load_iter_rows(args.inp))
    except StopIteration:
        print("Input empty."); sys.exit(2)
    fmap = norm_headers(headers, mapping)

    # Pass 1: collect row byte streams with robust row-level PCLK gating
    rows_bytes = list(build_row_bytes_sync(
        path=args.inp,
        fmap=fmap,
        pclk_edge=args.pclk_edge,
        href_active=args.href_active,
        bit_order=args.bit_order,
        invert_data=args.invert_data,
        use_pclk_gating=not args.no_pclk_gating,
    ))
    if not rows_bytes:
        print("No HREF-high rows found. Check HREF mapping/polarity."); sys.exit(3)

    # Build histogram of pixels-per-row to infer width
    hist = collections.Counter()
    byte_orders = [args.byte_order] if args.byte_order else ['msb_first', 'lsb_first']
    best_choice = None  # (width, byte_order, offset, votes)

    for bo in byte_orders:
        for offset in (0, 1):
            for rb in rows_bytes:
                pix_count = len(bytes_to_pixels_exact(rb, bo, offset))
                hist[pix_count] += 1
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
    print(f"Using byte_order={chosen_bo}, byte_offset={offset}, "
          f"pclk_edge={args.pclk_edge}, href_active={args.href_active}, "
          f"bit_order={args.bit_order}, pclk_gating={'ON' if not args.no_pclk_gating else 'OFF'}")

    # Pass 2: build and save the image
    rows_pixels = []
    for rb in rows_bytes:
        pix = bytes_to_pixels_exact(rb, chosen_bo, offset)
        # pad or center-trim to inferred width (uniform output)
        if len(pix) < width:
            pix = pix + [(0,0,0)] * (width - len(pix))
        else:
            start = max(0, (len(pix) - width) // 2)
            pix = pix[start:start + width]
        rows_pixels.append(pix)

    img = Image.new('RGB', (width, height))
    for y in range(height):
        row = rows_pixels[y] if y < len(rows_pixels) else [(0,0,0)] * width
        for x in range(width):
            img.putpixel((x, y), row[x])
    img.save(args.out)
    print(f"Saved {args.out} ({width}x{height}).")
    print("Tips: If colors look off, try --bit_order reversed or specify --byte_order explicitly. "
          "If geometry looks wrong, try --href_active low or --pclk_edge falling.")
    

if __name__ == '__main__':
    main()
