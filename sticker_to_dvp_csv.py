#!/usr/bin/env python3
"""
sticker_to_dvp_csv.py
Encode a PNG sticker into a DSView-like CSV of DVP signals (PCLK/HREF/VSYNC + D0..D7).

Model:
- 16-bit 565 per pixel (RGB or BGR layout).
- 2 bytes per pixel, 1 byte is presented and sampled on each *PCLK rising edge*.
- HREF is HIGH during each active row; VSYNC pulses at frame boundaries (optional).
- Bit order on bus: "normal" (D0=LSB .. D7=MSB) or "reversed" (D7=LSB .. D0=MSB).
- Byte order on bus: "msb_first" ([MSB][LSB]) or "lsb_first" ([LSB][MSB]).

Output matches DSView CSV conventions:
; CSV ...
; Channels (16/16)
; Sample rate: <rate> MHz
; Sample count: <N> Samples
Time(s),0,1,...,15
<rows of 0/1>
"""

import argparse, csv
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image

# ---------- helpers ----------

def load_image_rgba(path: str) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    return img

def rgba_to_rgb565_bytes(r: int, g: int, b: int, a: int, layout: str, byte_order: str) -> Tuple[int, int]:
    """
    Convert one RGBA pixel to two bytes (RGB/BGR565).
    Transparent (a==0) is treated as black (0) per your friend's instruction.
    """
    if a == 0:
        r = g = b = 0

    # clamp
    r5 = (r * 31) // 255
    g6 = (g * 63) // 255
    b5 = (b * 31) // 255

    if layout == "rgb":
        msb = ((r5 & 0x1F) << 3) | ((g6 >> 3) & 0x07)
        lsb = ((g6 & 0x07) << 5) | (b5 & 0x1F)
    else:  # "bgr"
        # B in high five, then G6, then R in low five
        msb = ((b5 & 0x1F) << 3) | ((g6 >> 3) & 0x07)
        lsb = ((g6 & 0x07) << 5) | (r5 & 0x1F)

    if byte_order == "msb_first":
        return msb, lsb
    else:
        return lsb, msb

def byte_to_bits(byte: int, bit_order: str) -> List[int]:
    """
    Map one byte onto D0..D7 lines for one sample (edge).
    - "normal": D0=LSB ... D7=MSB
    - "reversed": D7=LSB ... D0=MSB
    """
    if bit_order == "normal":
        return [(byte >> i) & 1 for i in range(8)]           # D0..D7
    else:
        return [(byte >> i) & 1 for i in range(7, -1, -1)]   # D7..D0

def init_row(num_channels: int) -> List[int]:
    return [0]*num_channels

# ---------- encoder ----------

def encode_png_to_csv(
    png_path: str,
    out_csv: str,
    mapping: Dict[str, str],
    layout: str = "rgb",
    byte_order: str = "lsb_first",
    bit_order: str = "normal",
    sample_rate_mhz: float = 20.0,
    include_vsync: bool = True,
    idle_edges_before: int = 4,
    idle_edges_between_rows: int = 2,
    meta_prefix: bool = False,
):
    """
    Create a DSView-like CSV that, when decoded with your existing pipeline,
    reconstructs the original sticker.

    Timing model: one CSV row per "sample". We produce:
      - PCLK toggles 0/1/0/1 ... (rising edge rows carry data)
      - Data lines (D0..D7) are *valid on rising-edge rows*.
      - HREF=1 during a row's active pixel bytes; 0 otherwise.
      - VSYNC pulses high for a few edges before the first row and after the last row (optional).
    """
    # Resolve channel indices from mapping (strings "0".."15" allowed)
    # We emit 16 columns 0..15 like your DSView exports.
    ch_count = 16
    def ci(key: str) -> int:
        v = mapping[key]
        return int(v)

    idx_pclk = ci("PCLK")
    idx_href = ci("HREF")
    idx_vsync = int(mapping.get("VSYNC", "15"))  # default use ch15 if not provided
    # data line indices in order D0..D7
    data_keys = ["D0","D1","D2","D3","D4","D5","D6","D7"]
    data_idx = [ci(k) for k in data_keys]

    # Load image
    im = load_image_rgba(png_path)
    W, H = im.size
    rgba = np.array(im, dtype=np.uint8).reshape(H, W, 4)

    # Precompute the 2 bytes per pixel
    # (flatten row-major so HREF boundaries are clean)
    pix_bytes: List[int] = []
    for y in range(H):
        row = rgba[y]
        for x in range(W):
            r, g, b, a = map(int, row[x])
            b0, b1 = rgba_to_rgb565_bytes(r, g, b, a, layout, byte_order)
            pix_bytes.append(b0)
            pix_bytes.append(b1)

    # Optional 4-byte meta prefix: [W_hi,W_lo,H_hi,H_lo]
    prefix_bytes: List[int] = []
    if meta_prefix:
        prefix_bytes = [(W >> 8) & 0xFF, W & 0xFF, (H >> 8) & 0xFF, H & 0xFF]

    # Build CSV rows
    rows = []  # list of channel states per sample
    pclk = 0
    href = 0
    vsync = 0

    def push_sample(data_byte: int = None, href_on: bool = False, vsync_on: bool = False):
        nonlocal pclk, href, vsync
        # rising edge will happen after we write this "low" sample
        # We emit two samples per "edge": LOW then HIGH carrying data.
        # LOW sample
        s_low = init_row(ch_count)
        s_low[idx_pclk] = 0
        s_low[idx_href] = 1 if href_on else 0
        if include_vsync:
            s_low[idx_vsync] = 1 if vsync_on else 0
        rows.append(s_low)
        # HIGH sample (rising edge) – data valid here
        s_high = init_row(ch_count)
        s_high[idx_pclk] = 1
        s_high[idx_href] = 1 if href_on else 0
        if include_vsync:
            s_high[idx_vsync] = 1 if vsync_on else 0
        if data_byte is not None:
            bits = byte_to_bits(data_byte, bit_order)
            # map bits onto the physical channel indices
            for bit_pos, chan_idx in enumerate(data_idx):
                s_high[chan_idx] = bits[bit_pos]
        rows.append(s_high)

    # Idle preamble
    for _ in range(idle_edges_before):
        push_sample(data_byte=None, href_on=False, vsync_on=False)

    # Optional VSYNC high for a few edges before frame
    if include_vsync:
        for _ in range(2):
            push_sample(data_byte=None, href_on=False, vsync_on=True)

    # Stream bytes:
    # If meta_prefix is on, send 4 bytes with HREF low (out-of-band metadata)
    for b in prefix_bytes:
        push_sample(data_byte=b, href_on=False, vsync_on=False)

    # Now rows with HREF=1. Walk the pixel bytes sequentially by row.
    i = 0
    for y in range(H):
        # a couple idle edges with HREF low between rows
        for _ in range(idle_edges_between_rows):
            push_sample(data_byte=None, href_on=False, vsync_on=False)
        # active row: output 2*W bytes with HREF high
        for _ in range(2*W):
            b = pix_bytes[i]; i += 1
            push_sample(data_byte=b, href_on=True, vsync_on=False)

    # Optional VSYNC tail
    if include_vsync:
        for _ in range(2):
            push_sample(data_byte=None, href_on=False, vsync_on=True)

    # DSView-like preamble + header
    dt = 1.0 / (sample_rate_mhz * 1_000_000.0)  # seconds per sample
    header = [str(i) for i in range(ch_count)]
    # Compose CSV file
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"; CSV generated by sticker_to_dvp_csv.py"])
        w.writerow([f"; Channels (16/16)"])
        w.writerow([f"; Sample rate: {sample_rate_mhz} MHz"])
        w.writerow([f"; Sample count: {len(rows)} Samples"])
        w.writerow(["Time(s)"] + header)
        # emit samples
        t = 0.0
        for s in rows:
            w.writerow([f"{t:.8E}"] + s)
            t += dt

    print(f"Saved CSV: {out_csv}")
    print(f"Sticker {W}x{H} → samples: {len(rows)}  (approx {len(rows)*dt:.6f}s @ {sample_rate_mhz} MHz)")
    if meta_prefix:
        print("Meta prefix inserted: [W_hi,W_lo,H_hi,H_lo] before frame bytes (HREF low).")

# ---------- CLI ----------

def parse_map(pairs: List[str]) -> Dict[str,str]:
    need = {"PCLK","HREF","D0","D1","D2","D3","D4","D5","D6","D7"}
    m = {}
    for p in pairs:
        k, v = p.split("=",1)
        m[k.strip().upper()] = v.strip()
    miss = need - set(m.keys())
    if miss:
        raise ValueError(f"Missing in --map: {sorted(miss)}")
    # VSYNC optional; user can add VSYNC=<col>
    if "VSYNC" in m:
        pass
    return m

def main():
    ap = argparse.ArgumentParser(description="Encode a PNG sticker into a DSView-like DVP CSV.")
    ap.add_argument("--png", required=True, help="input sticker (PNG)")
    ap.add_argument("--out_csv", required=True, help="output CSV path")
    ap.add_argument("--map", nargs="+", required=True,
                    help="PCLK=<ch> HREF=<ch> D0=<ch> ... D7=<ch> [VSYNC=<ch>]")
    ap.add_argument("--layout", choices=["rgb","bgr"], default="rgb")
    ap.add_argument("--byte_order", choices=["msb_first","lsb_first"], default="lsb_first")
    ap.add_argument("--bit_order", choices=["normal","reversed"], default="normal")
    ap.add_argument("--sample_rate_mhz", type=float, default=20.0)
    ap.add_argument("--no_vsync", action="store_true", help="do not pulse VSYNC")
    ap.add_argument("--idle_before", type=int, default=4, help="idle edges before frame")
    ap.add_argument("--idle_between_rows", type=int, default=2, help="idle edges between rows")
    ap.add_argument("--meta_prefix", action="store_true",
                    help="insert 4-byte width/height prefix before pixels with HREF low")
    args = ap.parse_args()

    mapping = parse_map(args.map)
    include_vsync = not args.no_vsync

    encode_png_to_csv(
        png_path=args.png,
        out_csv=args.out_csv,
        mapping=mapping,
        layout=args.layout,
        byte_order=args.byte_order,
        bit_order=args.bit_order,
        sample_rate_mhz=args.sample_rate_mhz,
        include_vsync=include_vsync,
        idle_edges_before=args.idle_before,
        idle_edges_between_rows=args.idle_between_rows,
        meta_prefix=args.meta_prefix,
    )

if __name__ == "__main__":
    main()



'''
I ran it in terminal:

python3 sticker_to_dvp_csv.py \
  --png images/cat.jpg \
  --out_csv sticker_encoded2.csv \
  --map PCLK=0 HREF=5 D0=6 D1=3 D2=4 D3=1 D4=2 D5=8 D6=9 D7=10 VSYNC=7 \
  --layout rgb \
  --byte_order lsb_first \
  --bit_order normal \
  --sample_rate_mhz 20 \
  --meta_prefix
'''
