from pathlib import Path

folder=Path(__file__).resolve().parent
source=(folder/'绘制图2_研究设计.py').read_text(encoding='utf-8')
mapping={
    "font='Microsoft YaHei'":"font='Times New Roman'",
    "1  日志与任务单位":"1  Dialogue logs and task units",
    "2  三层编码":"2  Three coding layers",
    "3  规则形成与回查":"3  Rule development and review",
    "4  任务分析与对照":"4  Analysis and comparison",
    "数据来源":"Data sources",
    "筛选规则":"Inclusion criterion",
    "层级结构":"Hierarchical structure",
    "WildChat  /  Search Arena\\n开放对话日志   /   搜索对话日志":"WildChat / Search Arena\\nOpen dialogue / Search-augmented",
    "源对话至少 4 次\\n非空用户提问":"At least 4 non-empty\\nuser turns per source dialogue",
    "源对话 → 多个任务\\n→ 起手与追问位置":"Source dialogue → tasks\\n→ initial query and follow-ups",
    "主分析：2,405 源对话，5,100 个任务；任务不是独立用户":"Main sample: 2,405 source dialogues and 5,100 tasks; tasks are not individual users",
    "(a) 起始需求":"(a) Initial need",
    "(b) 追问功能":"(b) Follow-up function",
    "(c) 过程成果":"(c) Process outcomes",
    "• 21 个细类\\n• 不明确、复合需求等特殊状态":"• 21 specific categories\\n• Unclear or compound needs",
    "• 15 类追问动作\\n• 多标签：一次追问可有\\n  多个功能":"• 15 action categories\\n• Multiple functions may\\n  occur in one follow-up",
    "• 任务／目标版本\\n• 整体目标与子目标\\n• 任务演化与最终形态":"• Task and goal versions\\n• Overall goals and subgoals\\n• Task evolution and outcomes",
    "动作读取当前提问前缀；成果读取完整任务过程":"Actions use the current dialogue prefix; outcomes use the full task trajectory",
    "研究者设计":"Researcher design",
    "初始规则\\n与提示词":"Initial rules\\nand prompts",
    "模型试编":"Pilot coding",
    "试跑材料\\n进行编码":"Model coding of\\npilot materials",
    "复核人员逐条核对":"Reviewer check",
    "检查编码一致性\\n与边界情况":"Item-by-item check\\nand boundary cases",
    "修订规则／提示词":"Rule revision",
    "根据问题案例\\n进行迭代":"Rules and prompts\\nrevised from cases",
    "迭代循环":"Iterative cycle",
    "回查与基础检查":"Audit checks",
    "回包 ID  ·  hash 检查  ·  引文检查":"Response ID  ·  hash  ·  evidence quotes",
    "核对数据记录、来源与引用":"Check records, sources, citations",
    "A  任务分析与传统对照":"A  Task analysis and traditional comparison",
    "B  ShareGPT 外部检查":"B  External check: ShareGPT",
    "① 需求与功能关系":"1. Need–function relationship",
    "不同起始需求下的追问功能分布":"Functions across initial-need categories",
    "② 完整链和局部结构":"2. Full sequences and local structures",
    "任务内追问序列的整体与局部特征":"Whole-task and local sequence features",
    "③ 整体／局部成果":"3. Overall and local outcomes",
    "最终成果与中间子目标的分析":"Overall outcomes and intermediate subgoals",
    "④ 传统共同策略对照":"4. Traditional-strategy comparison",
    "与传统检索中的共同策略进行对照":"Compare shared traditional-search strategies",
    "ShareGPT\\n外部对话数据":"ShareGPT\\nExternal dialogue data",
    "• 采用相同编码框架\\n  进行分析\\n• 用于独立来源的\\n  外部检查\\n• 不并入主分析样本":"• Same coding framework\\n• Independent-source check\\n• Not pooled into main sample",
    "图2_研究设计与分析单位_重绘.svg":"Figure2_Research_Design_EN.svg",
    "图2_研究设计与分析单位_重绘.png":"Figure2_Research_Design_EN.png",
}
for before,after in mapping.items():
    n=source.count(before)
    if n != 1: raise RuntimeError(f'expected exactly one match for {before!r}; found {n}')
    source=source.replace(before,after)
source=source.replace("box(32,y+h-1.1,36,4.4", "box(25,y+h-1.1,50,4.4")
exec(compile(source,str(folder/'绘制图2_研究设计.py'),'exec'),{'__file__':str(folder/'绘制图2_英文.py')})
