

src_font='font_data/simkai.ttf'
dst_font='font_data/HYRuiYiSongW.otf'

python font2img.py \
--src_font ${src_font} \
--dst_font ${dst_font} \
--charset=CN \
--sample_count 1000 \
--sample_dir image_dir \
--label 0 \
--filter \
--shuffle \
--mode 'font2font'
