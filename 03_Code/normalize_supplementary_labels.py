"""Normalize supplementary figure terminology without changing numerical marks."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parents[1]
im=Image.open(P/'06_Artwork/label_sources/Figure_S5.png').convert('RGB')
d=ImageDraw.Draw(im)
try:font=ImageFont.truetype('arial.ttf',40)
except OSError:
    from matplotlib.font_manager import findfont
    font=ImageFont.truetype(findfont('DejaVu Sans'),40)
d.rectangle((390,1505,1390,1664),fill='white')
for x,y,text,color in [(410,1520,'Achieved','#3B6E8F'),(855,1520,'Unknown','#D5DADD'),(410,1585,'Explicitly unmet','#C47A35'),(855,1585,'Topic-switch inferred','#8A64A0')]:
    d.rectangle((x,y+5,x+44,y+36),fill=color)
    d.text((x+80,y),text,font=font,fill='#222222')
out=P/'outputs/figures';out.mkdir(exist_ok=True,parents=True)
im.save(out/'Figure_S5.png',dpi=(400,400))

# Retain the original schematic; change only the legacy outcome label.
im=Image.open(P/'06_Artwork/label_sources/Figure_S2.png').convert('RGB')
d=ImageDraw.Draw(im)
font=ImageFont.truetype('arial.ttf',44)
d.rectangle((158,2055,1050,2103),fill=im.getpixel((150,2045)))
d.text((166,2055),'• Overall goal achieved',font=font,fill='#222222')
im.save(out/'Figure_S2.png',dpi=(400,400))
