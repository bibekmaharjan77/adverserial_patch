#!/usr/bin/env python3
"""
csv_to_rgb565_row_robust_sync.py

Goal
----
Reconstruct an RGB565 image from a DSView export (CSV or XLSX) of a DVP camera
interface (PCLK/HREF + 8-bit data). This version is more tolerant of real-world
captures where:
  - PCLK may already be high when HREF rises (row start).
  - Rows may have extra/missing bytes due to sampling jitter.

Key ideas
---------
1) Row-level PCLK re-sync:
   After each HREF rising edge (row start), ignore edges until the clock has
   been observed LOW at least once. Only then begin sampling *true* 0→1 edges.
   This implements the advice "only sample when previous PCLK was 0 and now 1".

2) Row rescue / alignment:
   For each row we collected a stream of bytes. We don't assume they start on
   an exact pixel boundary. Instead, we try both alignments (offset 0 and 1),
   convert to pixels, and choose the alignment with higher grayscale variance
   (a simple, robust heuristic that prefers 'real' image structure over flat rows).
   Then we trim/pad to exactly `width` pixels.

Inputs
------
--in           : CSV/XLSX path from DSView export
--map          : PCLK=<col> HREF=<col> D0=<col> ... D7=<col> (column names or numbers as strings)
--width/height : desired output geometry (e.g., 640x89 if your capture has 89 rows)
--pclk_edge    : which PCLK edge defines sampling (rising/falling)
--href_active  : HREF polarity (high/low)
--byte_order   : order of RGB565 byte pair on the bus (msb_first/lsb_first)
--bit_order    : bit significance across the 8 data columns (normal/reversed)
--invert_data  : invert the data bits if the analyzer captured inverted logic
--out          : output image filename

Outputs
-------
A PNG image written to --out with the requested size, built from the rows
decoded using the re-sync + row-rescue strategy.
"""

import argparse, csv, sys
from typing import List, Tuple
from PIL import Image


def is_high(x) -> int:
    """
    Normalize a DSView cell to a binary 0/1.

    DSView exports '0', '1', sometimes '0.0', or empty strings.
    Any nonzero value is treated as logic-high (1).
    """
    s = str(x).strip()
    return 0 if s in ('0', '', '0.0') else 1


def to_rgb565(msb, lsb):
    """
    Convert a single RGB565 pixel (two bytes) into an 8-bit/channel RGB tuple.

    RGB565 layout (msb_first):
      msb = RRRRRGGG (R:5 bits, G: upper 3 bits)
      lsb = GGGGBBBB (G: lower 3 bits, B:5 bits)

    We expand 5-bit channels to 8-bit by scaling 0..31 -> 0..255,
    and the 6-bit green 0..63 -> 0..255.
    """
    r = ((msb & 0xF8) >> 3) * 255 // 31
    g = (((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)) * 255 // 63
    b = (lsb & 0x1F) * 255 // 31
    return (r, g, b)


def load_iter_rows(path: str):
    """
    Unified row iterator for CSV or XLSX.

    Yields tuples: (headers, row_dict)

    Notes
    -----
    - XLSX is loaded fully (simpler but uses more memory). For very large XLSX
      consider saving to CSV in DSView and using the CSV path (streaming).
    - CSV is streamed via csv.DictReader for lower memory footprint.
    """
    if path.lower().endswith('.xlsx'):
        import pandas as pd
        df = pd.read_excel(path, engine='openpyxl')
        headers = list(df.columns)
        for _, row in df.iterrows():
            # Normalize to dict(header -> value) to match the CSV branch
            yield headers, {h: row[h] for h in headers}
    else:
        f = open(path, newline='')
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames
        for row in rdr:
            yield headers, row


def normalize_headers(headers, mapping):
    """
    Map user-provided column identifiers to actual headers in the file.

    Accepts:
      - Exact header names (e.g., "7", "HREF", "P0")
      - Case-insensitive matches

    Raises a KeyError if any mapping entry isn't found in headers.
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


def parse_mapping(pairs):
    """
    Parse --map entries like "PCLK=0", "HREF=5", "D0=1" ... "D7=10".

    Validates that all required keys are provided.
    """
    m = {}
    for p in pairs:
        k, v = p.split('=', 1)
        m[k.strip().upper()] = v.strip()
    need = {'PCLK','HREF','D0','D1','D2','D3','D4','D5','D6','D7'}
    missing = need - set(m.keys())
    if missing:
        raise ValueError(f"Missing in --map: {sorted(missing)}")
    return m


def bytes_to_pixels_rescued(bts: List[int], width: int, byte_order: str) -> List[tuple]:
    """
    Convert a row's *byte* stream into exactly `width` RGB pixels.

    Why "rescued"?
    --------------
    Real captures frequently have 1 stray byte at the beginning or end of a row.
    If we naively pair bytes starting at index 0, every color channel is wrong.
    So we:
      1) Try pairing from offset=0 and offset=1.
      2) Convert to pixels (respecting byte_order).
      3) For each candidate, compute grayscale variance; choose the higher one
         (tends to prefer better-aligned, more structured image rows).
      4) Trim/pad to exactly `width`.

    This recovers most lines without needing perfect timing.
    """
    import math
    candidates = []
    for offset in (0, 1):
        bb = bts[offset:]

        # If odd number of bytes, drop the last one to keep pairs aligned.
        if len(bb) % 2 == 1:
            bb = bb[:-1]

        # If we captured too many bytes, center-trim to the nearest 2*width.
        if len(bb) > 2 * width:
            start = max(0, (len(bb) - 2 * width) // 2)
            bb = bb[start:start + 2 * width]

        # Build pixels by pairing bytes
        pix = []
        for i in range(0, min(len(bb), 2 * width), 2):
            if byte_order == 'msb_first':
                msb, lsb = bb[i], bb[i + 1]
            else:
                lsb, msb = bb[i], bb[i + 1]
            pix.append(to_rgb565(msb, lsb))

        # Enforce exact row width (pad with black or truncate)
        if len(pix) < width:
            pix += [(0, 0, 0)] * (width - len(pix))
        else:
            pix = pix[:width]

        # Score by grayscale variance: higher = more structured = better alignment
        gs = [0.3 * r + 0.59 * g + 0.11 * b for (r, g, b) in pix]
        mu = sum(gs) / len(gs)
        var = sum((x - mu) ** 2 for x in gs) / len(gs)
        candidates.append((var, pix))

    # Choose the alignment with the larger variance (i.e., likely correct)
    return max(candidates, key=lambda t: t[0])[1]


def reconstruct(path, fmap, width, height, pclk_edge, href_active, byte_order, bit_order, invert_data):
    """
    Core streaming decoder.

    Process:
      - Walk the DSView rows once (streaming).
      - While HREF is active, sample one byte on PCLK edges.
      - Enforce row-level PCLK "arming":
          After HREF rises, require that PCLK has been seen low at least once
          before accepting any rising edges for sampling. This avoids starting
          mid-bit if PCLK was held high when HREF asserted.
      - At HREF falling edge, "rescue" the row (align/trim/pad) and store it.

    Parameters
    ----------
    path         : CSV/XLSX path
    fmap         : dict mapping logical names ('PCLK','HREF','D0'..'D7') -> actual headers
    width,height : output geometry
    pclk_edge    : 'rising' or 'falling' (which clock edge to sample)
    href_active  : 'high' or 'low' (active polarity of HREF)
    byte_order   : 'msb_first' or 'lsb_first' (RGB565 byte order on the bus)
    bit_order    : 'normal' (D0 is LSB) or 'reversed' (D7 is LSB) – affects byte packing
    invert_data  : if True, XOR each data bit (handles inverted analyzer polarity)
    """
    sample_on = 1 if pclk_edge == 'rising' else 0
    href_act  = 1 if href_active == 'high' else 0
    reversed_bits = (bit_order == 'reversed')

    prev_pclk = prev_href = 0
    row_bytes: List[int] = []
    rows_pixels: List[List[tuple]] = []

    # Row-level clock gating state:
    # - pclk_seen_low: has PCLK been low at least once since this row began?
    # - armed_for_rise: optional state to mark we've started sampling (kept for clarity).
    pclk_seen_low = True
    armed_for_rise = False

    for headers, row in load_iter_rows(path):
        # Current logic values for this sample
        pclk = is_high(row[fmap['PCLK']])
        hr   = is_high(row[fmap['HREF']])

        # Detect HREF rising: start a new row context
        if prev_href != href_act and hr == href_act:
            row_bytes = []
            # Require we see PCLK low at least once before accepting an edge.
            # If pclk is already 0 at this exact sample, great; otherwise it will
            # be set once we observe a 0 in subsequent samples.
            pclk_seen_low = (pclk == 0)
            armed_for_rise = False

        # Track whether PCLK has been low since row start
        if pclk == 0:
            pclk_seen_low = True

        # Accept a sampling edge only when:
        #   - HREF is active,
        #   - we saw a transition (prev != current),
        #   - the transition matches the configured sampling edge,
        #   - AND we've seen PCLK low since the row began.
        edge_ok = (prev_pclk == 0 and pclk == 1) if sample_on == 1 else (prev_pclk == 1 and pclk == 0)
        if hr == href_act and edge_ok and pclk_seen_low:
            armed_for_rise = True

            # Build an 8-bit value from the 8 data lines in the chosen bit order
            order = (['D0','D1','D2','D3','D4','D5','D6','D7'] if not reversed_bits
                     else ['D7','D6','D5','D4','D3','D2','D1','D0'])
            val = 0
            for j, key in enumerate(order):
                b = is_high(row[fmap[key]])
                if invert_data:
                    b ^= 1
                val |= (b << j)
            row_bytes.append(val)

        # End-of-row on HREF deassertion: convert collected bytes to pixels
        if prev_href == href_act and hr != href_act:
            if row_bytes:
                rows_pixels.append(bytes_to_pixels_rescued(row_bytes, width, byte_order))
                if len(rows_pixels) >= height:
                    break  # stop once we have enough rows for the requested output

        # Remember previous sample for transition detection
        prev_pclk, prev_href = pclk, hr

    # If the capture had fewer rows than requested, pad with black rows
    while len(rows_pixels) < height:
        rows_pixels.append([(0, 0, 0)] * width)

    # Crop if we somehow collected too many rows
    return rows_pixels[:height]


def main():
    """
    CLI entry point: parse args, build header mapping, run reconstruction,
    and write the final PNG.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='DSView export (.csv or .xlsx)')
    ap.add_argument('--width', type=int, required=True, help='Output image width in pixels')
    ap.add_argument('--height', type=int, required=True, help='Output image height (rows)')
    ap.add_argument('--map', nargs='+', required=True, help='PCLK=<col> HREF=<col> D0=<col> ... D7=<col>')
    ap.add_argument('--pclk_edge', choices=['rising','falling'], default='rising', help='Which PCLK edge to sample')
    ap.add_argument('--href_active', choices=['high','low'], default='high', help='HREF active polarity')
    ap.add_argument('--byte_order', choices=['msb_first','lsb_first'], default='lsb_first',
                    help='RGB565 byte order on the bus (your autosize result picked lsb_first)')
    ap.add_argument('--bit_order', choices=['normal','reversed'], default='normal',
                    help='normal: D0=LSB .. D7=MSB; reversed: D7=LSB .. D0=MSB')
    ap.add_argument('--invert_data', action='store_true', help='Invert D0..D7 bits (rare; analyzer polarity issue)')
    ap.add_argument('--out', default='frame.png', help='Output PNG path')
    args = ap.parse_args()

    # Parse and bind user mapping to actual headers
    mapping = parse_mapping(args.map)
    headers, _ = next(load_iter_rows(args.inp))
    fmap = normalize_headers(headers, mapping)

    # Do the reconstruction
    rows = reconstruct(
        args.inp, fmap, args.width, args.height,
        args.pclk_edge, args.href_active, args.byte_order, args.bit_order, args.invert_data
    )

    # Paint and save the final image
    img = Image.new('RGB', (args.width, args.height))
    for y in range(args.height):
        for x in range(args.width):
            img.putpixel((x, y), rows[y][x])
    img.save(args.out)
    print(f"Saved {args.out}  ({args.width}x{args.height})")


if __name__ == '__main__':
    main()
