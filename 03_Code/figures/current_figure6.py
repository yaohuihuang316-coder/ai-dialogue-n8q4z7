from pathlib import Path
import json,csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
P=Path(__file__).resolve().parents[2]
F=P/'outputs/figures';F.mkdir(parents=True,exist_ok=True)
W=F
reg=json.loads((P/'05_Results/regression_results.json').read_text('utf-8'))
mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.7})
names=['need_S','need_A','need_M','need_P','need_X','need_D','need_Y','need_F','clear','background','constraints','G1','G2','G3','G4','G5','queries','Search_Arena']
labels=['Experience seeking','Generation / processing','About the AI','Procedures / troubleshooting','Open exploration','Comparison / evaluation','Prediction / hypotheticals','Fact location','Clear goal','Background provided','Explicit constraints','G1 Understanding (+1)','G2 Need adjustment (+1)','G3 Solving / producing (+1)','G4 Verification (+1)','G5 Progress / feedback (+1)','Queries in task (+1)','Search Arena']
y=np.arange(18,dtype=float)
y[8:]+=.8;y[11:]+=.8;y[16:]+=.8
fig,axes=plt.subplots(1,2,figsize=(6.1,5.7),sharex=True,sharey=True)
fig.subplots_adjust(left=.365,right=.985,top=.90,bottom=.17,wspace=.13)
blue='#3B7090';grey='#C4C8CB';dark='#70777B';sparse='#C57D31'
rows=[]
for ax,key,title in zip(axes,['y1','y2'],['a  Outcome determinability\n(n = 5,100)','b  Achievement if determinable\n(n = 2,496)']):
    coef={r['name']:r for r in reg[key]['coefficients']}
    for i,name in enumerate(names):
        r=coef[name];x=float(r['AME'])*100;lo,hi=np.asarray(r['AME_CI95'])*100
        point_only=key=='y2' and name in ['need_S','need_X'];sig=float(r['holm_p'])<.05
        rows.append({'outcome':key,'variable':name,'AME_pp':x,'low95_pp':float(lo),'high95_pp':float(hi),'Holm_p':r['holm_p'],'point_only':point_only})
        if point_only:
            ax.plot(x,y[i],marker='D',markersize=3.5,mfc='white',mec=sparse,mew=.9,zorder=4)
        else:
            col=blue if sig else grey
            ax.barh(y[i],x,height=.46,color=col,edgecolor='none',zorder=2)
            ax.errorbar(x,y[i],xerr=[[x-lo],[hi-x]],fmt='none',ecolor=blue if sig else dark,elinewidth=.85,capsize=1.6,capthick=.75,zorder=3)
    ax.axvline(0,color='#747A7D',lw=.65,zorder=1)
    ax.set_axisbelow(True);ax.grid(axis='x',color='#EBECEE',linewidth=.5)
    ax.set_xlim(-42,36);ax.set_xticks([-40,-20,0,20]);ax.set_ylim(y[-1]+.7,-.8)
    ax.set_title(title,fontsize=8,fontweight='bold',loc='left',pad=10,linespacing=1.5)
    ax.set_yticks(y,labels);ax.tick_params(axis='y',length=0,pad=7)
axes[1].tick_params(axis='y',labelleft=False)
fig.supxlabel('Average marginal effect (percentage points)',x=.67,y=.105,fontsize=8)
handles=[Patch(facecolor=blue,label='Holm p < .05'),Patch(facecolor=grey,label='Holm p ≥ .05'),Line2D([],[],marker='D',ls='none',mfc='white',mec=sparse,markersize=3.5,label='Point estimate only')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.028),ncol=3,fontsize=7,frameon=False,handlelength=1.3,columnspacing=1.5)
fig.savefig(F/'Figure_6.pdf',facecolor='white')
fig.savefig(F/'Figure_6.svg',facecolor='white')
fig.savefig(F/'Figure_6.png',dpi=600,facecolor='white')
fig.savefig(F/'Figure_6.tiff',dpi=1000,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
with (W/'Figure_6_source.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
assert len(rows)==36
print('FIGURE6: 18 variables x 2 outcomes; no statistical changes.')
