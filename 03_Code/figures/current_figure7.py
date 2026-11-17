from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
P=Path(__file__).resolve().parents[2]
F=P/'outputs/figures';F.mkdir(parents=True,exist_ok=True)
W=F
con=json.loads((P/'05_Results/process_associations.json').read_text('utf-8'))
blue='#315E77';gray='#999999'
mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
def save(fig,stem):
    fig.savefig(F/f'{stem}.png',dpi=600,facecolor='white')
    fig.savefig(F/f'{stem}.pdf',facecolor='white')
    fig.savefig(F/f'{stem}.svg',facecolor='white')
    fig.savefig(F/f'{stem}.tiff',dpi=1000,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
fig,ax=plt.subplots(figsize=(6.1,3.55));fig.subplots_adjust(left=.40,right=.96,top=.83,bottom=.23)
spec=[('G1_then_G3','y2',0,'G1 → G3 (≥3 queries)'),('G2','y2',1,'Any need adjustment (G2)'),('G4','y2',2,'Any verification (G4)'),('G1_then_G3','y1',4,'G1 → G3 (≥3 queries)'),('constraints','y1',5,'Explicit initial constraints')]
for key,out,y,lab in spec:
    r=next(t for t in con if t['exposure']==key and t['outcome']==out and t['sample']=='all')
    for method,dy,marker,col in [('AIPW',-.12,'o',blue),('ATO',.12,'s',gray)]:
        x=r[method]*100;lo,hi=np.array(r[method+'_CI95'])*100
        ax.errorbar(x,y+dy,xerr=[[x-lo],[hi-x]],fmt=marker,color=col,capsize=2,ms=3.7,lw=1)
ax.axvline(0,color='#777777',lw=.7);ax.set_axisbelow(True);ax.grid(axis='x',color='#ECEFF1',lw=.5)
ax.set(xlim=(-37,13),ylim=(5.6,-1),yticks=[r[2] for r in spec],yticklabels=[r[3] for r in spec],xlabel='Adjusted risk difference (percentage points)')
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
ax.text(-39,-.67,'Achievement if determinable (y2)',ha='right',weight='bold',fontsize=8)
ax.text(-39,3.23,'Outcome determinability (y1)',ha='right',weight='bold',fontsize=8)
fig.text(.035,.96,'Adjusted process associations',weight='bold',fontsize=10)
fig.legend(handles=[Line2D([],[],color=blue,marker='o',label='AIPW'),Line2D([],[],color=gray,marker='s',label='ATO (overlap weights)')],loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False,fontsize=8)
fig.text(.035,.018,'Points and 95% intervals. Each contrast retains its original eligible task sample.',fontsize=7)
save(fig,'Figure_7')
