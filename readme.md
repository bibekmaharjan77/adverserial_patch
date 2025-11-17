Adverserial patch attack

Entire image is used as an adverserial patch and it is inserted into a new image that is captured so that it gets misclassified.

The image of a cat under the folder images is used as a sticker (adverserial patch) that would be inserted into the picture. This will happen when we insert the sticker's pixel data into the image's pixel data. So we are trying to do adverserial attack during the RAW pixel formation of an image (before a final image is formed in the form of jpeg, jpg, png or any other formats).

The dsl files are actually signal files of images where we are working on for this adverserial patch. I got this data from my friend. You can checkout his repo to know further about how the data was acquired and from which hardware here: https://github.com/TNeutron/ESP32-CAM---DVP-Image-Signal-in-RGB565-and-JPEG



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






