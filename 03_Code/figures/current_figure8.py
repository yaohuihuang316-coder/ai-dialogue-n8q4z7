"""Rebuild comparison Figures 8 and S6 using the frozen derived evidence.

Four-panel evidence chain: shared strategies, their wording, lexical measures,
and vocabulary overlap conditional on semantic similarity. Python/matplotlib.
Uncertainty is descriptive source-cluster bootstrap (1,000 draws), not a causal test.
"""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
P=Path(__file__).resolve().parents[2]
OUT=P/'outputs/figures';OUT.mkdir(parents=True,exist_ok=True)
E=json.loads((P/'05_Results/comparison_embeddings.json').read_text('utf-8'))
S=json.loads((P/'05_Results/shared_strategies.json').read_text('utf-8'))
U=json.loads((P/'outputs/embedding/summary.json').read_text('utf-8'))
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.titleweight':'bold','axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'savefig.facecolor':'white'})
BLUE='#3B6E8F';RED='#B03030'
def export(fig,name):
    fig.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        pass # The manuscript build records alignment; portable replication needs matplotlib only.
    else:
        require_matplotlib_panel_alignment(fig,json_out=OUT/f'{name}.alignment.json',strict=True)
    fig.add_artist(Rectangle((.002,.002),.996,.996,transform=fig.transFigure,fill=False,edgecolor='#333333',linewidth=1,clip_on=False))
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=600)
    plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(7.4,6.55))
fig.subplots_adjust(left=.225,right=.968,bottom=.12,top=.95,wspace=.48,hspace=.62)
ax=axs[0,0];keys=['SH_NARROW','SH_BROADEN','SH_SAME_INTENT_REPHRASE','SH_ASPECT_SWITCH']
for side,color,marker,offset,label in [('TREC2014',BLUE,'o',-.10,'Traditional search'),('AI_English',RED,'s',.10,'AI dialogue')]:
    vals=[100*S[side]['strategies'][k]['standardized_rate'] for k in keys];ys=np.arange(4)+offset
    ax.plot(vals,ys,linestyle='none',marker=marker,ms=5,color=color,mfc=color if marker=='o' else 'white',mew=1.3,label=label)
    for x,y in zip(vals,ys):ax.text(x+1.5,y,f'{x:.1f}',va='center',fontsize=7)
ax.set(yticks=range(4),yticklabels=['Narrowing','Broadening','Same-intent restatement','Facet shift'],xlim=(0,60),ylim=(3.6,-.5),xlabel='Ten-cell mean share (%)')
ax.set_title('a  Shared adjustment strategies',loc='left',pad=9)
ax.legend(loc='upper left',bbox_to_anchor=(-.12,-.22),fontsize=7,ncol=2,columnspacing=.7,handletextpad=.4)
ax=axs[0,1]
for y,side in enumerate(['TREC2014','AI_English']):
    n=S[side]['strategies']['SH_NARROW']['yes'];addition=S[side]['narrowing_lexical_categories']['add']/n*100
    ax.barh(y,addition,color=BLUE,height=.5,label='Pure addition' if y==0 else None)
    ax.barh(y,100-addition,left=addition,color=RED,height=.5,label='Other wording' if y==0 else None)
    for x,v in [(addition/2,addition),(addition+(100-addition)/2,100-addition)]:ax.text(x,y,f'{v:.1f}\n%' if v<15 else f'{v:.1f}%',ha='center',va='center',color='white',fontsize=6.5 if v<15 else 7)
ax.set(yticks=[0,1],yticklabels=['Traditional\n(n = 260)','AI dialogue\n(n = 668)'],xlim=(0,100),ylim=(1.6,-.6),xlabel='Share of semantic narrowing (%)')
ax.set_title('b  Wording when narrowing',loc='left',pad=9)
ax.legend(loc='upper left',bbox_to_anchor=(-.12,-.22),ncol=2,fontsize=7,columnspacing=.7,handlelength=1.4)
ax=axs[1,0]
keys=['unchanged_content','pure_add','both_add_delete','no_shared','referring','closer_to_feedback']
labels=['Unchanged content set','Pure content addition','Add and delete content','No shared content words','Referring words','Closer to prior feedback']
for side,color,off,label in [('TREC2014',BLUE,-.16,'Traditional'),('AI_English',RED,.16,'AI dialogue')]:
    vals=np.array([U['summary'][side][k]['estimate'] for k in keys])*100
    ci=np.array([U['summary'][side][k]['ci'] for k in keys])*100;ys=np.arange(6)+off
    ax.barh(ys,vals,height=.29,color=color,label=label)
    ax.errorbar(vals,ys,xerr=np.stack([vals-ci[:,0],ci[:,1]-vals]),fmt='none',ecolor='#333333',elinewidth=.8,capsize=1.7)
    for x,y,hi in zip(vals,ys,ci[:,1]):ax.text(max(x,hi)+1.0,y,f'{x:.1f}',va='center',fontsize=6.5)
ax.set(yticks=range(6),yticklabels=labels,xlim=(0,105),ylim=(5.6,-.6),xlabel='Ten-cell mean share (%)')
ax.set_title('c  Lexical change and feedback',loc='left',pad=9)
ax.legend(loc='upper left',bbox_to_anchor=(-.1,-.22),ncol=2,fontsize=7)
ax=axs[1,1]
for side,color,marker,label in [('TREC2014',BLUE,'o','Traditional'),('AI_English',RED,'s','AI dialogue')]:
    rows=sorted([r for r in U['bins'] if r['side']==side],key=lambda r:r['bin']);x=np.array([r['midpoint'] for r in rows]);v=np.array([r['mean_jaccard'] for r in rows]);ci=np.array([r['ci'] for r in rows])
    ax.errorbar(x,v,yerr=np.stack([v-ci[:,0],ci[:,1]-v]),marker=marker,color=color,ms=3.5,linewidth=1.2,capsize=1.8,label=label)
ax.set(xlim=(0,1),ylim=(0,1),xlabel='Semantic similarity (cosine bin)',ylabel='Content-word Jaccard')
ax.set_title('d  Overlap by semantic similarity',loc='left',pad=9);ax.legend(loc='upper left',fontsize=7)
export(fig,'Figure_8')

# Cluster names describe the executed features, with the original stacked-bar form retained.
fig,ax=plt.subplots(figsize=(7.4,3.7));fig.subplots_adjust(left=.19,right=.97,top=.93,bottom=.40)
order=[0,4,2,1,3];colors=[BLUE,'#88AABF',RED,'#CD8585','#D5DADD'];names=['Retain / extend','Partial substitution','Shorter rewording','Expanded rewording','No shared content']
clusters={r['cluster']:r for r in E['gmm_reconstruction']['clusters']}
left=np.zeros(2)
for k,color,name in zip(order,colors,names):
    vals=np.array([E['gmm_reconstruction']['shares'][side][str(k)]*100 for side in ['AI_English','TREC2014']]);r=clusters[k]
    ax.barh([0,1],vals,left=left,color=color,height=.55,edgecolor='white',linewidth=.8,label=f'{name} (J = {r["content_jaccard"]:.2f}, cos = {r["query_cosine"]:.2f})')
    for y,v,l in zip([0,1],vals,left):
        if v>=4:ax.text(l+v/2,y,f'{v:.0f}',ha='center',va='center',color='white' if k in [0,2] else 'black',fontsize=8)
    left+=vals
assert np.allclose(left,100)
ax.set(yticks=[0,1],yticklabels=['AI dialogue','Traditional'],xlim=(0,100),ylim=(1.6,-.6),xlabel='Ten-cell standardized composition (%)')
ax.legend(loc='upper left',bbox_to_anchor=(-.18,-.35),ncol=2,fontsize=7,columnspacing=1.2)
export(fig,'Figure_S6')
print('Rebuilt Figure 8 and Figure S6 from derived records.')
