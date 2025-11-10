command in the terminal to run csv_to_image_rescue:

python3 csv_to_image_rescue.py \
  --csv "/path/to/csv" \
  --width 640 --height 480 \
  --map PCLK=0 HREF=5 D0=6 D1=3 D2=4 D3=1 D4=2 D5=8 D6=9 D7=10 \
  --pclk_edge rising --href_active high \
  --byte_order lsb_first --layout rgb \
  --out frame_rgb_lsb3.png

command in terminal to run sticker_to_dvp_csv:

python3 sticker_to_dvp_csv.py \
  --png images/cat.jpg \
  --out_csv sticker_encoded2.csv \
  --map PCLK=0 HREF=5 D0=6 D1=3 D2=4 D3=1 D4=2 D5=8 D6=9 D7=10 VSYNC=7 \
  --layout rgb \
  --byte_order lsb_first \
  --bit_order normal \
  --sample_rate_mhz 20 \
  --meta_prefix






