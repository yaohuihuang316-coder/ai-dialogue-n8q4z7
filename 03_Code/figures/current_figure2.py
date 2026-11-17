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
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch
def draw_figure():
    # Redraw the author's supplied schematic with native paths and live type.
    # No data marks, counts or analytical relationships are inferred from pixels.
    plt.rcParams.update({'font.family':'Times New Roman','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(6.1,7.35),facecolor='white')
    ax=fig.add_axes([.008,.006,.984,.988]);ax.set(xlim=(0,1000),ylim=(1200,0));ax.axis('off')
    ink='#20252a';blue='#314f61';edge='#7d858b';pale='#f0f2f3'
    def box(x,y,w,h,fill='white',lw=.6,dash=False):
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=(0,(3,3)) if dash else '-')
        ax.add_patch(p)
    def t(x,y,s,size=9.2,bold=False,ha='center',color=ink):
        ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va='center',color=color,linespacing=1.23)
    def arrow(x1,y1,x2,y2,dash=False):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,linewidth=.8,color=blue,linestyle=(0,(3,3)) if dash else '-'))
    def icon(x,y,kind):
        if kind=='db':
            ax.add_patch(Rectangle((x-12,y-10),24,21,fill=False,edgecolor=blue,lw=.7))
            for yy in [y-10,y+1,y+11]:ax.add_patch(Ellipse((x,yy),24,8,facecolor='white' if yy==y-10 else 'none',edgecolor=blue,lw=.7))
        elif kind=='nodes':
            for xx,yy in [(x,y-10),(x-12,y+10),(x+12,y+10)]:ax.add_patch(Ellipse((xx,yy),8,8,fill=False,edgecolor=blue,lw=.8))
            ax.plot([x,x,x-12,x-12],[y-6,y+1,y+1,y+6],color=blue,lw=.7)
            ax.plot([x,x+12,x+12],[y+1,y+1,y+6],color=blue,lw=.7)
        else:
            ax.add_patch(Rectangle((x-11,y-14),22,28,fill=False,edgecolor=blue,lw=.7))
            if kind=='check':ax.plot([x-7,x-2,x+7],[y,y+5,y-6],color=blue,lw=1)
            else:
                for yy in [-6,0,6]:ax.plot([x-6,x+6],[y+yy,y+yy],color=blue,lw=.7)
    def stage(y,h,title):
        box(8,y+23,984,h-23,dash=True)
        box(243,y,514,48,pale,.9)
        t(500,y+24,title,12,True)
    def card(x,y,w,h,title,lines,kind='doc',body_size=9.4):
        box(x,y,w,h);box(x,y,w,44,pale)
        icon(x+25,y+22,kind);t(x+w/2+17,y+22,title,9.2,True)
        t(x+w/2,y+44+(h-44)/2,lines,body_size)
    stage(9,244,'1  Dialogue logs and task units')
    card(23,70,306,126,'Public dialogue sources','WildChat / Search Arena\nPublic dialogue /\nsearch-augmented dialogue','db')
    card(347,70,306,126,'Source selection and coding','≥4 non-empty user requests\nComplete text ≤20,000 characters\nCoding and task-completeness checks','check',8.3)
    card(671,70,306,126,'Hierarchical structure','Source conversations → tasks\n→ initial request and follow-ups','nodes')
    box(23,211,954,29,pale);t(500,225.5,'Main sample: 2,405 source conversations, 5,100 tasks, 10,850 follow-up positions',9.3)
    stage(274,235,'2  Three coding layers')
    card(23,334,306,126,'(a) Initial needs','21 need categories\nGoal, background, constraints','doc')
    card(347,334,306,126,'(b) Follow-up functions','15 action categories → 5 functions\nMultiple functions may occur\nin one request','nodes')
    card(671,334,306,126,'(c) Task outcomes','Overall and local achievement\nOutcome determinability','check')
    box(23,475,954,24,pale);t(500,487,'All three layers are coded on the same task records',9.4)
    stage(530,216,'3  Rule development and review')
    for x,w,txt in [(35,255,'Pilot coding'),(357,310,'Researcher and reviewer\nchecks'),(745,220,'Rule refinement')]:
        box(x,597,w,51,pale);t(x+w/2,622.5,txt,9.4 if '\n' in txt else 10.2,True)
    arrow(296,622,350,622);arrow(675,622,738,622)
    ax.plot([855,855,163],[656,683,683],color=blue,lw=.8);arrow(163,683,163,653)
    t(500,668,'Iterative cycle',9.8,True)
    box(23,708,954,25,pale);t(500,720.5,'Rule development and review support the shared coding representation',9.2)
    stage(767,423,'4  Analysis and complementary checks')
    box(23,827,562,286);box(23,827,562,44,pale);icon(48,849,'doc');t(316,849,'A  Main task-level analysis',10.7,True)
    items=[('RQ1  Request composition','Initial needs and function combinations'),('RQ2  Process organization','Complete paths and local structures; next-function prediction'),('RQ3  Task progress','Overall and local outcomes; adjusted process associations'),('Contextual comparison','TREC 2014 ↔ AI dialogue; semantic strategies\nand lexical changes')]
    for j,(head,body) in enumerate(items):
        yy=890+j*52;t(42,yy,head,10.3,True,ha='left')
        if '\n' in body:ax.text(42,yy+18,body,fontsize=8.8,ha='left',va='top',color=ink,linespacing=1.12)
        else:t(42,yy+23,body,8.8,ha='left')
    box(620,827,357,286);box(620,827,357,44,pale);icon(645,849,'db');t(808,849,'B  External prediction check',10.1,True)
    t(798,909,'ShareGPT\n542 source conversations',11,True)
    t(798,1007,'Local functional dependence\nPrediction check only\nSeparate from the main sample',9.8)
    # The dashed connector now exists and matches the legend; it denotes a
    # complementary check, not flow or pooling of observations between samples.
    arrow(588,957,617,957,True)
    box(23,1132,954,40,pale)
    arrow(100,1152,154,1152);t(170,1152,'Analytical workflow',9.4,ha='left')
    arrow(565,1152,619,1152,True);t(635,1152,'Complementary analysis',9.4,ha='left')
    paths=[]
    for ext in ['png','pdf','svg']:
        path=F/f'Figure_2.{ext}';fig.savefig(path,dpi=1000,facecolor='white');paths.append(str(path))
    fig.savefig(W/'Figure_2_preview.png',dpi=180,facecolor='white');plt.close(fig)
    return paths
draw_figure()
