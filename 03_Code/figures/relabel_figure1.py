"""Reapply the two editorial headings of Figure 1; all other pixels are retained.
Usage: python 03_Code/figures/relabel_figure1.py --font /path/to/arialbd.ttf
The supplied final artwork was rendered with Arial Bold, 55 px.
"""
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
p=argparse.ArgumentParser();p.add_argument('--font',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
im=Image.open(root/'06_Artwork/label_sources/Figure_1_before_titles.png').convert('RGB')
draw=ImageDraw.Draw(im);font=ImageFont.truetype(a.font,55)
for box,center,title in [((240,38,2210,130),(1220,80),'Query–document matching'),((210,1018,2240,1115),(1220,1061),'Task development in AI dialogue')]:
    draw.rectangle(box,fill=im.getpixel((100,box[1])));draw.text(center,title,font=font,fill='#222222',anchor='mm')
out=root/'outputs/figures';out.mkdir(parents=True,exist_ok=True);im.save(out/'Figure_1.png',dpi=(400,400))
