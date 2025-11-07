# file: csv_to_image_min.py
import csv, argparse
import numpy as np
from PIL import Image

def buffer_to_image(raw: bytes, width: int, height: int,
                    byte_order: str = "msb_first",
                    layout: str = "rgb"):
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
    img = img.reshape(height, width, 3)
    return Image.fromarray(img, mode="RGB")

def is_high(x):
    s = str(x).strip()
    return 0 if s in ("", "0", "0.0") else 1

def open_dsview_dictreader(path, skip_lines=0, skip_comment_preamble=True):
    """
    Position at the real header row and return a DictReader.
    - Skips exactly skip_lines lines first (if >0)
    - Then skips any leading lines starting with ';' (DSView preamble)
    - Strips whitespace from fieldnames
    """
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
    # Normalize header names aggressively (strip spaces & BOMs)
    if rdr.fieldnames:
        rdr.fieldnames = [ (h or "").strip().lstrip("\ufeff") for h in rdr.fieldnames ]
    return rdr

def main():
    ap = argparse.ArgumentParser(description="Reconstruct RGB/BGR565 from DSView CSV (minimal).")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--map", nargs="+", required=True,
                    help="PCLK=<col> HREF=<col> D0=<col> ... D7=<col>  (use header names like 0,1,2...)")
    ap.add_argument("--pclk_edge", choices=["rising","falling"], default="rising")
    ap.add_argument("--href_active", choices=["high","low"], default="high")
    ap.add_argument("--bit_order", choices=["normal","reversed"], default="normal")
    ap.add_argument("--invert_data", action="store_true")
    ap.add_argument("--byte_order", choices=["msb_first","lsb_first"], default="lsb_first")
    ap.add_argument("--layout", choices=["rgb","bgr"], default="rgb")
    ap.add_argument("--skip_lines", type=int, default=0)
    ap.add_argument("--print_headers", action="store_true",
                    help="Print detected CSV headers then exit")
    ap.add_argument("--out", default="frame.png")
    args = ap.parse_args()

    # Parse mapping
    m = {}
    for p in args.map:
        k, v = p.split("=", 1)
        m[k.upper()] = v

    # Open CSV and normalize headers
    rdr = open_dsview_dictreader(args.csv, skip_lines=args.skip_lines, skip_comment_preamble=True)
    hdr = rdr.fieldnames
    if not hdr:
        raise RuntimeError("No CSV header found after skipping preamble; check --skip_lines or file format.")
    # Build a case-insensitive, space-stripped lookup
    hdr_lookup = {h.strip().lower(): h for h in hdr}

    if args.print_headers:
        print("Headers:", hdr)
        return

    # Resolve a mapping token (e.g., "0", "HREF") to the actual header string
    def col(name: str):
        key = str(name).strip()
        # exact
        if key in hdr: return key
        # case/space-insensitive
        lk = key.lower()
        if lk in hdr_lookup: return hdr_lookup[lk]
        # be forgiving if there are stray spaces in header
        for h in hdr:
            if h.strip() == key: return h
        raise KeyError(f"Column '{name}' not found. Available headers: {hdr}")

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
    row_bytes = []
    all_bytes = []

    for row in rdr:
        clk = is_high(row[PCLK])
        hr  = is_high(row[HREF])

        # Start-of-row: arm gating (require a LOW first)
        if prev_href != href_act and hr == href_act:
            row_bytes = []
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
            row_bytes.append(v)

        # End-of-row → append to full-frame buffer
        if prev_href == href_act and hr != href_act:
            all_bytes.extend(row_bytes)
            row_bytes = []

        prev_clk, prev_href = clk, hr

    # Expect exactly width*height*2 bytes (pad or center-trim)
    b = np.array(all_bytes, dtype=np.uint8)
    need = args.width * args.height * 2
    if b.size < need:
        b = np.pad(b, (0, need - b.size), mode="constant")
    elif b.size > need:
        start = max(0, (b.size - need)//2)
        b = b[start:start+need]

    img = buffer_to_image(b.tobytes(), args.width, args.height,
                          byte_order=args.byte_order, layout=args.layout)
    img.save(args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
