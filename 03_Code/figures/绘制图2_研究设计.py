from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

out=Path(__file__).resolve().parent
font='Microsoft YaHei'
available={f.name for f in font_manager.fontManager.ttflist}
if font not in available:
    font='SimHei' if 'SimHei' in available else 'SimSun'
matplotlib.rcParams.update({'font.family':font,'axes.unicode_minus':False,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.0,8.6),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,100);ax.set_ylim(0,102);ax.axis('off')
def box(x,y,w,h,fill='white',edge='#777777',lw=.75,r=1.1,ls='solid'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=ls))
def txt(x,y,s,fs=8.4,weight='normal',ha='center',va='center',color='#191919',lines=1.35):
    ax.text(x,y,s,fontsize=fs,weight=weight,ha=ha,va=va,color=color,linespacing=lines)
def flow(x1,y1,x2,y2,style='-|>',lw=.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=8,linewidth=lw,color='#343434'))
def panel(x,y,w,h,title):
    box(x,y,w,h,fill='white',edge='#8b8b8b',lw=.7)
    box(x,y+h-4.3,w,4.3,fill='#eeeeee',edge='#8b8b8b',lw=.6,r=.6)
    txt(x+w/2,y+h-2.15,title,fs=9.1,weight='bold')
def stage(y,h,title):
    box(1.5,y,97,h,edge='#555555',ls=(0,(5,3)),lw=.7,r=1.2)
    box(32,y+h-1.1,36,4.4,fill='white',edge='#343434',lw=1.0,r=1)
    txt(50,y+h+1.1,title,fs=10.3,weight='bold')

stage(79.0,18,'1  日志与任务单位')
panel(3.3,85.0,30.3,9.9,'数据来源')
txt(18.45,88.0,'WildChat  /  Search Arena\n开放对话日志   /   搜索对话日志',fs=8.2)
panel(35.0,85.0,28.0,9.9,'筛选规则')
txt(49.0,88.0,'源对话至少 4 次\n非空用户提问',fs=8.4)
panel(64.4,85.0,32.3,9.9,'层级结构')
txt(80.55,88.0,'源对话 → 多个任务\n→ 起手与追问位置',fs=8.4)
box(3.3,80.1,93.4,3.5,fill='#f2f2f2',edge='#999999',lw=.6,r=.5)
txt(50,81.85,'主分析：2,405 源对话，5,100 个任务；任务不是独立用户',fs=8.3)
flow(95,78.8,95,77.3)

stage(57.5,19,'2  三层编码')
panel(3.3,62.9,29.9,11.2,'(a) 起始需求')
txt(5.5,67.4,'• 21 个细类\n• 不明确、复合需求等特殊状态',fs=8.1,ha='left')
panel(34.5,62.9,30.8,11.2,'(b) 追问功能')
txt(36.6,66.2,'• 15 类追问动作\n• 多标签：一次追问可有\n  多个功能',fs=8.1,ha='left')
panel(66.6,62.9,30.1,11.2,'(c) 过程成果')
txt(68.6,66.2,'• 任务／目标版本\n• 整体目标与子目标\n• 任务演化与最终形态',fs=8.1,ha='left')
box(3.3,58.6,93.4,3.0,fill='#f2f2f2',edge='#999999',lw=.6,r=.5)
txt(50,60.1,'动作读取当前提问前缀；成果读取完整任务过程',fs=8.3)
flow(95,57.2,95,55.7)

stage(33.2,21.5,'3  规则形成与回查')
panels=[(3.2,21.8,'研究者设计','初始规则\n与提示词'),(27.4,21.8,'模型试编','试跑材料\n进行编码'),(51.6,21.8,'复核人员逐条核对','检查编码一致性\n与边界情况'),(75.8,21.0,'修订规则／提示词','根据问题案例\n进行迭代')]
for x,w,title,body in panels:
    box(x,44.0,w,8.1,fill='white',edge='#555555',r=1)
    txt(x+w/2,49.8,title,fs=8.1,weight='bold')
    txt(x+w/2,46.4,body,fs=7.8)
for x in (25.2,49.4,73.6): flow(x,48.0,x+2.0,48.0,lw=.8)
ax.plot([86,86,38,38],[43.4,41.6,41.6,43.4],color='#333333',lw=.8)
flow(38,42.2,38,43.8,lw=.8)
txt(62,42.55,'迭代循环',fs=8.0,weight='bold')
box(3.2,34.4,93.5,5.5,fill='#f7f7f7',edge='#888888',lw=.7,r=.8,ls=(0,(4,2)))
txt(6.0,37.15,'回查与基础检查',fs=8.1,weight='bold',ha='left')
txt(37.0,37.15,'回包 ID  ·  hash 检查  ·  引文检查',fs=8.1)
txt(77.0,37.15,'核对数据记录、来源与引用',fs=8.1)
flow(95,32.9,95,31.4)

stage(1.0,29.4,'4  任务分析与对照')
panel(3.3,2.3,57.8,25.4,'A  任务分析与传统对照')
for i,(title,detail) in enumerate([
    ('① 需求与功能关系','不同起始需求下的追问功能分布'),
    ('② 完整链和局部结构','任务内追问序列的整体与局部特征'),
    ('③ 整体／局部成果','最终成果与中间子目标的分析'),
    ('④ 传统共同策略对照','与传统检索中的共同策略进行对照')]):
    yy=22.2-i*4.8
    txt(5.0,yy,title,fs=8.2,weight='bold',ha='left')
    txt(5.0,yy-1.8,detail,fs=7.8,ha='left',color='#444444')
panel(62.5,2.3,34.2,25.4,'B  ShareGPT 外部检查')
txt(79.6,21.3,'ShareGPT\n外部对话数据',fs=9.0,weight='bold')
txt(65.0,13.1,'• 采用相同编码框架\n  进行分析\n• 用于独立来源的\n  外部检查\n• 不并入主分析样本',fs=8.2,ha='left')

fig.savefig(out/'图2_研究设计与分析单位_重绘.svg',format='svg',facecolor='white')
fig.savefig(out/'图2_研究设计与分析单位_重绘.png',format='png',dpi=400,facecolor='white')
print('font',font,'svg and png saved')
