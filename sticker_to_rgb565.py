#!/usr/bin/env python3
"""
sticker_to_rgb565.py

Usage:
    python sticker_to_rgb565.py /path/to/sticker.png

What it does:
- Loads the input image (will be resized to 480x320 if different)
- For each pixel:
    - produce 8-bit binary strings for R, G, B (e.g. '11111111')
    - compute RGB565 16-bit value and store as uint16
- Outputs:
    - saves RGB565 numpy array to 'sticker_rgb565.npy'
    - saves RGB565 binary stream (big-endian) to 'sticker_rgb565_be.bin'
    - saves a small CSV with hex RGB565 per pixel (optional, for inspection)
    - saves 'sticker_rgb888_binary.npy' (optional) containing (H,W,3) binary strings
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import csv

TARGET_W = 480
TARGET_H = 320

def to_8bit_bin_str(value: int) -> str:
    """Return 8-bit binary string for 0-255 value."""
    return format(int(value) & 0xFF, '08b')

def rgb888_to_rgb565_value(r: int, g: int, b: int) -> np.uint16:
    """
    Convert 8-bit r,g,b to a 16-bit RGB565 value.
    Using bit truncation (>>), equivalent to floor(value * max_new / 255).
    """
    r5 = (r >> 3) & 0x1F       # keep top 5 bits
    g6 = (g >> 2) & 0x3F       # keep top 6 bits
    b5 = (b >> 3) & 0x1F       # keep top 5 bits
    val = (r5 << 11) | (g6 << 5) | b5
    return np.uint16(val)

def image_to_rgb565_and_binaries(img: Image.Image, force_size=(TARGET_W, TARGET_H)):
    """
    Convert PIL image to:
      - rgb565_array: (H,W) uint16 numpy array
      - rgb888_bin_array: (H,W,3) numpy array of 8-bit binary strings for R,G,B
    If force_size is provided, the image is resized (bilinear).
    """
    if force_size is not None:
        img = img.convert('RGB').resize(force_size, Image.BILINEAR)
    else:
        img = img.convert('RGB')

    arr = np.array(img, dtype=np.uint8)   # shape (H,W,3)
    H, W, _ = arr.shape

    rgb565 = np.zeros((H, W), dtype=np.uint16)
    # We'll store binary strings in an object dtype array (can be large)
    rgb888_bin = np.empty((H, W, 3), dtype=object)

    # Vectorized-ish approach: get channels arrays
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Compute RGB565 using integer arithmetic (vectorized)
    r5 = (r >> 3).astype(np.uint16)   # 5 bits
    g6 = (g >> 2).astype(np.uint16)   # 6 bits
    b5 = (b >> 3).astype(np.uint16)   # 5 bits
    rgb565 = (r5 << 11) | (g6 << 5) | b5

    # Create per-channel 8-bit binary strings (looping; strings can't be vectorized easily)
    # This is somewhat expensive for large images but fine for 480x320 (~153k pixels).
    for y in range(H):
        for x in range(W):
            rr = int(r[y, x])
            gg = int(g[y, x])
            bb = int(b[y, x])
            rgb888_bin[y, x, 0] = to_8bit_bin_str(rr)
            rgb888_bin[y, x, 1] = to_8bit_bin_str(gg)
            rgb888_bin[y, x, 2] = to_8bit_bin_str(bb)

    return rgb565, rgb888_bin, img

def save_outputs(out_dir: Path, input_path: Path, rgb565_arr: np.ndarray, rgb888_bin_arr: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Numpy save of uint16 array
    np.save(out_dir / 'sticker_rgb565.npy', rgb565_arr)
    # Save big-endian binary stream (common bus packing)
    with open(out_dir / 'sticker_rgb565_be.bin', 'wb') as f:
        # ensure big-endian uint16
        f.write(rgb565_arr.astype('>u2').tobytes())

    # Also save little-endian variant (useful in practice)
    with open(out_dir / 'sticker_rgb565_le.bin', 'wb') as f:
        f.write(rgb565_arr.astype('<u2').tobytes())

    # Save a CSV of hex values (rows=W columns=H optionally flattened)
    csv_path = out_dir / 'sticker_rgb565_hex.csv'
    H, W = rgb565_arr.shape
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x','y','rgb565_hex'])  # header
        for y in range(H):
            for x in range(W):
                writer.writerow([x, y, format(int(rgb565_arr[y,x]), '#06x')])  # e.g. 0x7e0

    # Save the binary strings array (object dtype) as a numpy file (inspectable)
    np.save(out_dir / 'sticker_rgb888_binary.npy', rgb888_bin_arr)

    print(f"[+] Saved outputs to: {out_dir}")
    print(f"    - sticker_rgb565.npy (uint16 array)")
    print(f"    - sticker_rgb565_be.bin (big-endian stream)")
    print(f"    - sticker_rgb565_le.bin (little-endian stream)")
    print(f"    - sticker_rgb565_hex.csv (inspectable hex per-pixel)")
    print(f"    - sticker_rgb888_binary.npy (H,W,3) 8-bit string array")

def main():
    if len(sys.argv) < 2:
        print("Usage: python sticker_to_rgb565.py /path/to/sticker.png")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print("Input file not found:", input_path)
        sys.exit(1)

    # Load image
    img = Image.open(input_path)
    print("[*] Loaded image:", input_path, "size:", img.size)

    # Convert
    rgb565_arr, rgb888_bin_arr, resized_img = image_to_rgb565_and_binaries(img, force_size=(TARGET_W, TARGET_H))
    print("[*] Converted to RGB565 shape:", rgb565_arr.shape)

    # Output directory
    out_dir = input_path.parent / (input_path.stem + "_rgb565_out")
    save_outputs(out_dir, input_path, rgb565_arr, rgb888_bin_arr)

    # Optionally save a reconstructed preview image from rgb565 (to verify packing)
    # reconstruct back to 24-bit RGB for visual check:
    H, W = rgb565_arr.shape
    recon = np.zeros((H, W, 3), dtype=np.uint8)
    r5 = (rgb565_arr >> 11) & 0x1F
    g6 = (rgb565_arr >> 5) & 0x3F
    b5 = rgb565_arr & 0x1F
    recon[:,:,0] = (r5 * 255) // 31
    recon[:,:,1] = (g6 * 255) // 63
    recon[:,:,2] = (b5 * 255) // 31
    Image.fromarray(recon, 'RGB').save(out_dir / 'reconstructed_from_rgb565.png')
    # Also save resized original for comparison
    resized_img.save(out_dir / 'resized_input_480x320.png')
    print("[*] Also saved reconstructed preview and resized input.")

if __name__ == '__main__':
    main()
