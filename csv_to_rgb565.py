# file: buffer_to_image.py
import numpy as np
from PIL import Image
import csv, argparse

def buffer_to_image(raw: bytes, width: int, height: int,
                    byte_order: str = "msb_first",    # or "lsb_first"
                    layout: str = "rgb"):              # or "bgr"
    """
    Convert a packed 16-bit 565 frame into an 8-bit RGB PIL image.

    raw        : bytes-like of length == width*height*2
    byte_order : "msb_first"  -> [MSB][LSB]
                 "lsb_first"  -> [LSB][MSB]
    layout     : "rgb" (RGB565) or "bgr" (BGR565)
                 - RGB565:   RRRRRGGG GGGGBBBB
                 - BGR565:   BBBBBGGG GGGGRRRR  (same packing, channels swapped)
    """
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 2)
    if byte_order == "msb_first":
        msb, lsb = buf[:, 0].astype(np.uint16), buf[:, 1].astype(np.uint16)
    else:
        lsb, msb = buf[:, 0].astype(np.uint16), buf[:, 1].astype(np.uint16)

    # Extract 5:6:5 channels (bit math identical for RGB/BGR; we just map names later)
    five_r  =  (msb & 0xF8) >> 3
    six_g   = ((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)
    five_b  =  (lsb & 0x1F)

    # Scale to 8-bit per channel
    R = (five_r * 255 // 31).astype(np.uint8)
    G = (six_g  * 255 // 63).astype(np.uint8)
    B = (five_b * 255 // 31).astype(np.uint8)

    # Channel layout
    if layout == "rgb":
        img = np.stack([R, G, B], axis=1)
    else:  # "bgr"
        img = np.stack([B, G, R], axis=1)

    img = img.reshape(height, width, 3)
    return Image.fromarray(img, mode="RGB")

# file: csv_to_image_min.py
def is_high(x):  # DSView normalization
    s = str(x).strip()
    return 0 if s in ("", "0", "0.0") else 1

def main():
    ap = argparse.ArgumentParser(description="Reconstruct RGB/BGR565 from DSView CSV (minimal).")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--map", nargs="+", required=True,
                    help="PCLK=<col> HREF=<col> D0=<col> ... D7=<col>")
    ap.add_argument("--pclk_edge", choices=["rising","falling"], default="rising")
    ap.add_argument("--href_active", choices=["high","low"], default="high")
    ap.add_argument("--bit_order", choices=["normal","reversed"], default="normal")
    ap.add_argument("--invert_data", action="store_true")
    ap.add_argument("--byte_order", choices=["msb_first","lsb_first"], default="lsb_first")
    ap.add_argument("--layout", choices=["rgb","bgr"], default="rgb")  # note: may be bgr
    ap.add_argument("--out", default="frame.png")
    args = ap.parse_args()

    # Parse mapping
    m = {}
    for p in args.map:
        k, v = p.split("=", 1)
        m[k.upper()] = v

    with open(args.csv, newline="") as f:
        rdr = csv.DictReader(f)
        hdr = rdr.fieldnames
        # resolve headers case-insensitively
        def col(name):
            if name in hdr: return name
            low = {h.lower(): h for h in hdr}
            return low[name.lower()]

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

            # Start-of-row: arm clock gating (require a LOW first)
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

    # Expect exactly width*height*2 bytes (if not, we’ll center-trim or pad)
    b = np.array(all_bytes, dtype=np.uint8)
    need = args.width * args.height * 2
    if b.size < need:                 # pad with zeros (black)
        b = np.pad(b, (0, need - b.size), mode="constant")
    elif b.size > need:               # center-trim
        start = max(0, (b.size - need)//2)
        b = b[start:start+need]

    img = buffer_to_image(b.tobytes(), args.width, args.height,
                          byte_order=args.byte_order, layout=args.layout)
    img.save(args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()


"""
I ran the code using the following command in the terminal:

python3 csv_to_rgb565.py \
  --csv "/Users/bibekmaharjan/Desktop/640x480.dsl-la-251016-230216.csv" \
  --width 640 --height 89 \ 
  --map PCLK=0 HREF=5 D0=6 D1=3 D2=4 D3=1 D4=2 D5=8 D6=9 D7=10 \
  --pclk_edge rising --href_active high \
  --byte_order lsb_first --layout rgb \
  --out frame_rgb_lsb.png
"""