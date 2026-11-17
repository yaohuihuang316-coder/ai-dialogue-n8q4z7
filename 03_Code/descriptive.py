"""Descriptive summaries using the published units and denominators."""
import collections
import numpy as np

def task_patterns(tasks):
    paths = collections.Counter(tuple(t['row']['sequence5']) for t in tasks if t['row']['steps'] > 0)
    links = collections.Counter((a,b) for t in tasks for a,b in zip(t['row']['sequence5'],t['row']['sequence5'][1:]))
    first = {}
    for a,b in [('G1','G3'),('G2','G4'),('G1','G4')]:
        counts=collections.Counter()
        for t in tasks:
            seq=[s.split('+') for s in t['row']['sequence5']]
            ia=next((i for i,s in enumerate(seq) if a in s),None);ib=next((i for i,s in enumerate(seq) if b in s),None)
            if ia is not None and ib is not None: counts['before' if ia<ib else 'after' if ia>ib else 'same_position']+=1
        first[a+' / '+b]=dict(counts)
    return dict(distinct_complete_paths=len(paths),singleton_path_types=sum(n==1 for n in paths.values()),adjacent_pairs=sum(links.values()),same_state_pairs=sum(n for (a,b),n in links.items() if a==b),first_occurrences=first,transitions=[dict(previous=a,next=b,count=n) for (a,b),n in links.most_common()])

def shared_strategies(pairs):
    labels=['SH_NARROW','SH_BROADEN','SH_SAME_INTENT_REPHRASE','SH_ASPECT_SWITCH'];cells=[(d,p) for d in range(2,6) for p in range(1,d)];out={}
    for side in ['TREC2014','AI_English']:
        frame=[r for r in pairs if r['side']==side]
        valid=[r for r in frame if r['labels']['eligibility']=='explicit_information' and r['labels']['task_relation']=='same_task']
        result=dict(frame_pairs=len(frame),sources=len({r['source_key'] for r in frame}),eligible_same_task_pairs=len(valid),excluded_pairs=len(frame)-len(valid),excluded_fraction=(len(frame)-len(valid))/len(frame) if frame else None,strategies={})
        for label in labels:
            cc=[]
            for d,p in cells:
                subset=[r for r in valid if r['depth']==d and r['position']==p];yes=sum(r['labels'][label]=='yes' for r in subset);known=sum(r['labels'][label] in ['yes','no'] for r in subset)
                cc.append(dict(depth=d,position=p,yes=yes,known=known,eligible=len(subset),rate=yes/known if known else None))
            result['strategies'][label]=dict(yes=sum(c['yes'] for c in cc),known=sum(c['known'] for c in cc),standardized_rate=float(np.mean([c['rate'] for c in cc])) if all(c['rate'] is not None for c in cc) else None,cells=cc)
        narrow=[r for r in valid if r['labels']['SH_NARROW']=='yes'];cats=collections.Counter(r['lexical']['category'] for r in narrow)
        result['narrowing_lexical_categories']=dict(cats)
        out[side]=result
    return out
