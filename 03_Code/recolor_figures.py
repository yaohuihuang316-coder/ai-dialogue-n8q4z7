"""Palette-only adaptation of retained raster masters, with unchanged geometry.
Run from any directory. No data are fitted or redrawn; grayscale text is untouched.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFont
P=Path(__file__).resolve().parents[1]
outdir=P/'outputs/figures';outdir.mkdir(parents=True,exist_ok=True)
oldcols=['#3B6E8F','#C47A35','#4A8A7A','#8A64A0','#8C939A']
newcols=['#3B6E8F','#B03030','#527F9C','#C06162','#A8BECC']
rgb=lambda h:np.array([int(h[i:i+2],16) for i in [1,3,5]],float)
report=[]
for n in [3,4,8]:
    src=P/f'06_Artwork/palette_sources/Figure_{n}.png'
    im=Image.open(src);a=np.array(im.convert('RGB'))
    colors,inv=np.unique(a.reshape(-1,3),axis=0,return_inverse=True)
    out=colors.copy();best=np.full(len(colors),1000.);chroma=colors.max(1)-colors.min(1)
    for oh,nh in zip(oldcols,newcols):
        o=rgb(oh);v=rgb(nh)
        for bg in [255.,238.,252.]:
            direction=o-bg;alpha=np.clip((colors-bg)@direction/(direction@direction),0,1)
            fit=bg+alpha[:,None]*direction;err=np.max(np.abs(colors-fit),axis=1)
            take=(err<best)&(err<=1.6)&(alpha>=.025)&(chroma>3)
            out[take]=np.clip(np.rint(bg+alpha[take,None]*(v-bg)),0,255).astype('uint8');best[take]=err[take]
    b=out[inv].reshape(a.shape);gray=np.ptp(a,axis=2)==0
    assert np.array_equal(a[gray],b[gray])
    if n==4:
        ink=Image.fromarray(b);draw=ImageDraw.Draw(ink)
        try:font=ImageFont.truetype('arial.ttf',42)
        except OSError:
            from matplotlib.font_manager import findfont
            font=ImageFont.truetype(findfont('DejaVu Sans'),42)
        draw.rectangle((400,1770,2350,1870),fill='white')
        labels=[('Achieved','#3B6E8F'),('Explicitly unmet','#B03030'),('Not determinable','#D5DADD')]
        widths=[100+32+draw.textlength(t,font=font) for t,c in labels]
        x=(im.width-sum(widths)-2*100)/2
        for (text,color),width in zip(labels,widths):
            draw.rectangle((round(x),1810,round(x)+99,1848),fill=color)
            draw.text((round(x)+132,1804),text,fill='#222222',font=font)
            x+=width+100
        b=np.array(ink)
    # White padding holds a 4-pixel neutral frame outside all existing content.
    pad=24
    framed=Image.new('RGB',(im.width+pad*2,im.height+pad*2),'white')
    framed.paste(Image.fromarray(b),(pad,pad))
    ImageDraw.Draw(framed).rectangle((3,3,framed.width-4,framed.height-4),outline='#333333',width=4)
    target=outdir/f'Figure_{n}.png';framed.save(target,dpi=im.info.get('dpi',(400,400)))
    report.append({'figure':n,'original_grayscale_unchanged_outside_legend':True,'legend_relabelled':n==4,'source_dimensions':im.size,'dimensions':framed.size,'outer_frame_px':4,'padding_px':24,'palette':newcols,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(outdir/'palette_verification.json').write_text(json.dumps(report,indent=2),'utf-8')
print(json.dumps(report))
