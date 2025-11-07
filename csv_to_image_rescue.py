#!/usr/bin/env python3
import csv, argparse
import numpy as np
from PIL import Image

def buffer_to_image(raw: bytes, width: int, height: int,
                    byte_order: str = "msb_first", layout: str = "rgb"):
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2)
    if byte_order == "msb_first":
        msb, lsb = buf[:, 0].astype(np.uint16), buf[:, 1].astype(np.uint16)
    else:
        lsb, msb = buf[:, 0].astype(np.uint16), buf[:, 1].astype(np.uint16)
    five_r = (msb & 0xF8) >> 3
    six_g  = ((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)
    five_b = (lsb & 0x1F)
    R = (five_r * 255 // 31).astype(np.uint8)
    G = (six_g  * 255 // 63).astype(np.uint8)
    B = (five_b * 255 // 31).astype(np.uint8)
    img = np.stack([R, G, B], axis=1) if layout == "rgb" else np.stack([B, G, R], axis=1)
    return Image.fromarray(img.reshape(height, width, 3), mode="RGB")

def is_high(x):
    s = str(x).strip()
    return 0 if s in ("", "0", "0.0") else 1

def open_dsview_dictreader(path, skip_lines=0, skip_comment_preamble=True):
    f = open(path, newline='')
    for _ in range(max(0, skip_lines)):
        f.readline()
    if skip_comment_preamble:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                f.seek(pos); break
            if not line.lstrip().startswith(';'):
                f.seek(pos); break
    rdr = csv.DictReader(f, skipinitialspace=True)
    if rdr.fieldnames:
        rdr.fieldnames = [(h or "").strip().lstrip("\ufeff") for h in rdr.fieldnames]
    return rdr

def bytes_to_pixels_row(bb: list[int], width: int, byte_order: str):
    """Rescue a *row* by trying offsets 0/1 and picking higher grayscale variance."""
    def make_pix(offset):
        b = bb[offset:]
        if len(b) % 2: b = b[:-1]
        if len(b) > 2*width:
            s = max(0, (len(b)-2*width)//2)
            b = b[s:s+2*width]
        # form pixels (just count here; channel mapping later)
        # we return packed bytes so buffer_to_image can do channel layout uniformly
        return np.asarray(b, dtype=np.uint8)

    cands = [make_pix(0), make_pix(1)]
    scores = []
    for c in cands:
        # quick “structure” score using 565 expand on the fly
        if byte_order == "msb_first":
            msb, lsb = c[0::2].astype(np.uint16), c[1::2].astype(np.uint16)
        else:
            lsb, msb = c[0::2].astype(np.uint16), c[1::2].astype(np.uint16)
        five_r = (msb & 0xF8) >> 3
        six_g  = ((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)
        five_b = (lsb & 0x1F)
        R = five_r * 255 // 31
        G = six_g  * 255 // 63
        B = five_b * 255 // 31
        gs = 0.3*R + 0.59*G + 0.11*B
        mu = gs.mean() if gs.size else 0.0
        scores.append(((gs - mu)**2).mean() if gs.size else 0.0)
    best = cands[int(scores[1] > scores[0])]
    # pad/trim to exactly 2*width bytes
    if best.size < 2*width:
        best = np.pad(best, (0, 2*width-best.size), mode="constant")
    elif best.size > 2*width:
        s = max(0, (best.size - 2*width)//2)
        best = best[s:s+2*width]
    return best

def main():
    ap = argparse.ArgumentParser(description="CSV → RGB/BGR565 with row rescue.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--map", nargs="+", required=True, help="PCLK=.. HREF=.. D0=.. .. D7=..")
    ap.add_argument("--pclk_edge", choices=["rising","falling"], default="rising")
    ap.add_argument("--href_active", choices=["high","low"], default="high")
    ap.add_argument("--bit_order", choices=["normal","reversed"], default="normal")
    ap.add_argument("--invert_data", action="store_true")
    ap.add_argument("--byte_order", choices=["msb_first","lsb_first"], default="lsb_first")
    ap.add_argument("--layout", choices=["rgb","bgr"], default="rgb")
    ap.add_argument("--skip_lines", type=int, default=0)
    ap.add_argument("--out", default="frame.png")
    args = ap.parse_args()

    # parse mapping
    m = {}
    for p in args.map:
        k, v = p.split("=", 1)
        m[k.upper()] = v

    rdr = open_dsview_dictreader(args.csv, skip_lines=args.skip_lines, skip_comment_preamble=True)
    hdr = rdr.fieldnames
    if not hdr: raise RuntimeError("No CSV header found.")

    lookup = {h.strip().lower(): h for h in hdr}
    def col(name):
        key = str(name).strip()
        if key in hdr: return key
        lk = key.lower()
        if lk in lookup: return lookup[lk]
        for h in hdr:
            if h.strip() == key: return h
        raise KeyError(f"Column '{name}' not in {hdr}")

    PCLK = col(m["PCLK"])
    HREF = col(m["HREF"])
    order = (["D0","D1","D2","D3","D4","D5","D6","D7"]
             if args.bit_order=="normal" else
             ["D7","D6","D5","D4","D3","D2","D1","D0"])
    data_cols = [col(m[k]) for k in order]

    samp_on = 1 if args.pclk_edge=="rising" else 0
    href_act = 1 if args.href_active=="high" else 0

    prev_clk = prev_href = 0
    pclk_seen_low = True

    rows_bytes = []
    row_acc = []

    for row in rdr:
        clk = is_high(row[PCLK])
        hr  = is_high(row[HREF])

        # row start: arm gating
        if prev_href != href_act and hr == href_act:
            row_acc = []
            pclk_seen_low = (clk == 0)

        if clk == 0:
            pclk_seen_low = True

        edge_ok = (clk != prev_clk) and (clk == samp_on)
        if hr == href_act and edge_ok and pclk_seen_low:
            v = 0
            for j, c in enumerate(data_cols):
                b = is_high(row[c])
                v |= ((b ^ args.invert_data) << j)
            row_acc.append(v)

        # row end: rescue this row
        if prev_href == href_act and hr != href_act:
            if row_acc:
                rows_bytes.append(bytes_to_pixels_row(row_acc, args.width, args.byte_order))
            row_acc = []

        prev_clk, prev_href = clk, hr

    # If file ended mid-row, finalize it too
    if row_acc:
        rows_bytes.append(bytes_to_pixels_row(row_acc, args.width, args.byte_order))

    # Assemble final frame buffer (2*width bytes per row), pad/crop to height
    row_len = 2 * args.width
    while len(rows_bytes) < args.height:
        rows_bytes.append(np.zeros(row_len, dtype=np.uint8))
    rows_bytes = rows_bytes[:args.height]
    frame = np.concatenate(rows_bytes, axis=0)

    img = buffer_to_image(frame.tobytes(), args.width, args.height,
                          byte_order=args.byte_order, layout=args.layout)
    img.save(args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()


'''
current best image generated after running command:

python3 csv_to_image_rescue.py \
  --csv "/Users/bibekmaharjan/Downloads/640x480.dsl-la-251029-124819.csv" \
  --width 640 --height 480 \
  --map PCLK=0 HREF=5 D0=6 D1=3 D2=4 D3=1 D4=2 D5=8 D6=9 D7=10 \
  --pclk_edge rising --href_active high \
  --byte_order lsb_first --layout rgb \
  --out frame_rgb_lsb3.png
'''