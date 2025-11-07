
#!/usr/bin/env python3
"""
dsl_csv_to_image_autotune.py
(see previous description; brute-forces common signal options to recover a non-black frame)
"""
import argparse, csv, sys, itertools, statistics
from PIL import Image

def parse_mapping(pairs):
    m = {}
    for p in pairs:
        k, v = p.split('=',1)
        m[k.strip().upper()] = v.strip()
    req = {'PCLK','HREF','VSYNC','D0','D1','D2','D3','D4','D5','D6','D7'}
    miss = req - set(m.keys())
    if miss:
        raise ValueError(f"Missing mappings: {sorted(miss)}")
    return m

def get_fieldmap(fieldnames, mapping):
    fmap = {}
    for k, v in mapping.items():
        if v in fieldnames:
            fmap[k] = v
        else:
            low = [h for h in fieldnames if h.lower() == v.lower()]
            if low:
                fmap[k] = low[0]
            else:
                raise KeyError(f"CSV has no column '{v}'. Headers: {fieldnames}")
    return fmap

def is_high_str(s, invert=False):
    bit = 0 if str(s).strip() in ('0','','0.0') else 1
    return 1-bit if invert else bit

def to_rgb(msb, lsb):
    r = ((msb & 0xF8) >> 3) * 255 // 31
    g = (((msb & 0x07) << 3) | ((lsb & 0xE0) >> 5)) * 255 // 63
    b = (lsb & 0x1F) * 255 // 31
    return (r, g, b)

def read_rows(rdr, fmap, w, h, pclk_edge, vsync_active, href_active, byte_order, bit_order_reversed, invert_data, limit_rows=None):
    prev_pclk=prev_vsync=prev_href=0
    image_rows=[]; row_pixels=[]; pending=[]; rows_seen=0

    for row in rdr:
        pclk = is_high_str(row[fmap['PCLK']])
        vs   = is_high_str(row[fmap['VSYNC']])
        hr   = is_high_str(row[fmap['HREF']])

        if prev_vsync == (1 if vsync_active=='high' else 0) and vs != (1 if vsync_active=='high' else 0):
            image_rows.clear(); row_pixels.clear(); pending.clear()

        active_row = (hr == (1 if href_active=='high' else 0))
        edge_ok = (pclk != prev_pclk) and (pclk == (1 if pclk_edge=='rising' else 0))

        if active_row and edge_ok:
            order = (['D0','D1','D2','D3','D4','D5','D6','D7'] if not bit_order_reversed
                     else ['D7','D6','D5','D4','D3','D2','D1','D0'])
            val=0
            for j, key in enumerate(order):
                b = is_high_str(row[fmap[key]], invert=invert_data)
                val |= (b << j)
            pending.append(val)
            if len(pending)==2:
                if byte_order=='msb_first':
                    msb, lsb = pending[0], pending[1]
                else:
                    lsb, msb = pending[0], pending[1]
                row_pixels.append(to_rgb(msb, lsb))
                pending.clear()
                if len(row_pixels) == w:
                    image_rows.append(row_pixels); row_pixels=[]; rows_seen+=1
                    if limit_rows and rows_seen>=limit_rows:
                        break

        if prev_href == (1 if href_active=='high' else 0) and hr != (1 if href_active=='high' else 0):
            if row_pixels:
                if len(row_pixels) < w:
                    row_pixels += [(0,0,0)]*(w-len(row_pixels))
                image_rows.append(row_pixels[:w]); row_pixels=[]; rows_seen+=1
                if limit_rows and rows_seen>=limit_rows:
                    break

        prev_pclk, prev_vsync, prev_href = pclk, vs, hr

    return image_rows

def score_rows(rows):
    import math
    flat = [pix for r in rows for pix in r]
    if not flat:
        return 0.0, 0, 0.0
    nonblack = sum(1 for (r,g,b) in flat if (r|g|b)!=0)
    gs = [ (r*0.3 + g*0.59 + b*0.11) for (r,g,b) in flat ]
    mean = sum(gs)/len(gs)
    var = sum((x-mean)**2 for x in gs) / len(gs)
    st = math.sqrt(var)
    return st + nonblack/10000.0, nonblack, st

def build_image(rows, w, h, path):
    while len(rows) < h:
        rows.append([(0,0,0)]*w)
    rows = rows[:h]
    img = Image.new('RGB', (w, h))
    for y in range(h):
        r = rows[y]
        if len(r) < w:
            r = r + [(0,0,0)]*(w-len(r))
        for x in range(w):
            img.putpixel((x,y), r[x])
    img.save(path)

def main():
    import argparse, itertools
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--map', nargs='+', required=True)
    ap.add_argument('--save_best', default='best_autotuned.png')
    ap.add_argument('--limit_rows', type=int, default=200)
    ap.add_argument('--pclk_edge', choices=['rising','falling'])
    ap.add_argument('--vsync_active', choices=['high','low'])
    ap.add_argument('--href_active', choices=['high','low'])
    ap.add_argument('--byte_order', choices=['msb_first','lsb_first'])
    ap.add_argument('--bit_order', choices=['normal','reversed'])
    ap.add_argument('--invert_data', choices=['off','on'])
    args = ap.parse_args()

    with open(args.csv, newline='') as f:
        rdr = csv.DictReader(f)
        fmap = get_fieldmap(rdr.fieldnames, parse_mapping(args.map))

    opt_space = {
        'pclk_edge':     [args.pclk_edge] if args.pclk_edge else ['rising','falling'],
        'vsync_active':  [args.vsync_active] if args.vsync_active else ['high','low'],
        'href_active':   [args.href_active] if args.href_active else ['high','low'],
        'byte_order':    [args.byte_order] if args.byte_order else ['msb_first','lsb_first'],
        'bit_order':     [args.bit_order] if args.bit_order else ['normal','reversed'],
        'invert_data':   [args.invert_data] if args.invert_data else ['off','on'],
    }
    combos = list(itertools.product(*opt_space.values()))
    keys = list(opt_space.keys())

    best = None
    for combo in combos:
        params = dict(zip(keys, combo))
        with open(args.csv, newline='') as f:
            rdr = csv.DictReader(f)
            rows = read_rows(
                rdr, fmap, args.width, args.height,
                params['pclk_edge'], params['vsync_active'], params['href_active'],
                params['byte_order'], params['bit_order']=='reversed', params['invert_data']=='on',
                limit_rows=args.limit_rows
            )
        score, nonblack, st = score_rows(rows)
        print(f"score={score:.3f} nonblack={nonblack} stdev={st:.2f} params={params}")
        if best is None or score > best[0]:
            best = (score, nonblack, st, params, rows)

    if best and best[4]:
        print("\nBest params:", best[3], f"score={best[0]:.3f}, nonblack={best[1]}, stdev={best[2]:.2f}")
        build_image(best[4], args.width, args.height, args.save_best)
        print("Saved:", args.save_best)
    else:
        print("No good combination found. Check mapping or whether the capture is RGB565 vs JPEG.")

if __name__ == "__main__":
    main()
