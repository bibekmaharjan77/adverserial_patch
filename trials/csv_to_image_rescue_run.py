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

def rescue_row_bytes(bb: list[int], width: int, byte_order: str) -> np.ndarray:
    """Try offsets 0/1, choose higher grayscale variance; return exactly 2*width bytes."""
    def candidate(offset):
        b = bb[offset:]
        if len(b) % 2: b = b[:-1]
        if len(b) > 2*width:
            s = max(0, (len(b)-2*width)//2)
            b = b[s:s+2*width]
        return np.asarray(b, dtype=np.uint8)

    c0 = candidate(0); c1 = candidate(1)
    def score(c):
        if byte_order == "msb_first":
            msb, lsb = c[0::2].astype(np.uint16), c[1::2].astype(np.uint16)
        else:
            lsb, msb = c[0::2].astype(np.uint16), c[1::2].astype(np.uint16)
        R = ((msb & 0xF8) >> 3) * 255 // 31
        G = ((((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)) * 255 // 63)
        B = ((lsb & 0x1F) * 255 // 31)
        gs = 0.3*R + 0.59*G + 0.11*B
        mu = gs.mean() if gs.size else 0.0
        return float(((gs - mu)**2).mean()) if gs.size else 0.0
    best = c1 if score(c1) > score(c0) else c0

    # pad/trim to exactly 2*width bytes
    need = 2*width
    if best.size < need:
        best = np.pad(best, (0, need-best.size), mode="constant")
    elif best.size > need:
        s = max(0, (best.size-need)//2)
        best = best[s:s+need]
    return best

def longest_good_run(good_flags):
    """Return (start, length) of the longest contiguous True run."""
    best_s = best_len = 0
    cur_s = cur_len = 0
    for i, g in enumerate(good_flags + [False]):
        if g:
            if cur_len == 0: cur_s = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_s, best_len = cur_s, cur_len
            cur_len = 0
    return best_s, best_len

def main():
    ap = argparse.ArgumentParser(description="CSV → RGB/BGR565 with row rescue + good-run selection.")
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
    ap.add_argument("--row_tol_pct", type=float, default=10.0,
                    help="Allowed % deviation from 2*width bytes to mark a row as good (default 10%%).")
    ap.add_argument("--out", default="frame.png")
    args = ap.parse_args()

    # mapping
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

    # collect raw rows (byte counts + data)
    raw_rows = []
    row_acc = []

    for row in rdr:
        clk = is_high(row[PCLK])
        hr  = is_high(row[HREF])

        # row start
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
                if args.invert_data: b ^= 1
                v |= (b << j)
            row_acc.append(v)

        # row end
        if prev_href == href_act and hr != href_act:
            raw_rows.append(row_acc)
            row_acc = []

        prev_clk, prev_href = clk, hr

    if row_acc:
        raw_rows.append(row_acc)

    # mark good rows by byte count
    expected = 2 * args.width
    tol = int(round(expected * args.row_tol_pct / 100.0))
    good = [abs(len(r) - expected) <= tol for r in raw_rows]

    # choose longest contiguous good run (one clean frame)
    start, length = longest_good_run(good)
    if length == 0:
        # fallback: use all rows with rescue (may still band)
        use_rows = raw_rows
        print("Warning: no good contiguous run found; using all rows.")
    else:
        use_rows = raw_rows[start:start+min(length, args.height)]
        print(f"Selected good run: rows {start}..{start+length-1} (len={length}), expected per-row bytes={expected}±{tol}")

    # rescue and assemble
    rows_bytes = [rescue_row_bytes(bb, args.width, args.byte_order) for bb in use_rows]
    # pad/crop to height
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
