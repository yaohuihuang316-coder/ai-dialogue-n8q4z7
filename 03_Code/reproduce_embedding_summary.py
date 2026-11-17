"""Recompute lexical/embedding summaries and uncertainty from released derived pairs.

No model call is needed. The embedding inference specification is in
02_Data/embedding_method.json. Outputs never overwrite supplied results.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
P=Path(__file__).resolve().parents[1]
OUT=P/'outputs/embedding';OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(P/'02_Data/comparison_embedding_metrics.csv')
expected=json.loads((P/'05_Results/comparison_embeddings.json').read_text('utf-8'))
assert len(df)==3847
df['unchanged_content']=((df.added_fraction==0)&(df.deleted_fraction==0)).astype(int)
metrics=['query_cosine','content_jaccard','closer_to_feedback','pure_add','both_add_delete','no_shared','referring','unchanged_content']
summary={};boots={};bins=[]
for si,side in enumerate(['TREC2014','AI_English']):
    q=df[df.side==side].copy();groups=list(q.sort_values('source_bootstrap_order').source_key.drop_duplicates());gi={s:i for i,s in enumerate(groups)}
    sums=np.zeros((len(groups),10,len(metrics)));counts=sums.copy()
    for r in q.itertuples():
        for j,m in enumerate(metrics):
            v=getattr(r,m)
            if pd.notna(v):sums[gi[r.source_key],int(r.cell),j]+=float(v);counts[gi[r.source_key],int(r.cell),j]+=1
    def calc(w):
        a=np.einsum('i,ijk->jk',w,sums);b=np.einsum('i,ijk->jk',w,counts)
        return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0).mean(0)
    estimate=calc(np.ones(len(groups)));rng=np.random.default_rng(20261112+si)
    boot=np.array([calc(np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))) for _ in range(1000)])
    boots[side]=boot;summary[side]={}
    for j,m in enumerate(metrics):
        valid=boot[:,j][np.isfinite(boot[:,j])]
        summary[side][m]={'estimate':float(estimate[j]),'ci':np.quantile(valid,[.025,.975]).tolist()}
        if m in expected['summary'][side]['metrics']:
            e=expected['summary'][side]['metrics'][m]
            assert np.allclose([estimate[j],*summary[side][m]['ci']],[e['estimate'],*e['ci']],atol=1e-10),(side,m)
    # Source-cluster bootstrap within each fixed semantic bin, preserving within-source pairs.
    for binid,g in q.groupby('bin'):
        ag=g.groupby('source_key').content_jaccard.agg(['sum','count']);n=len(ag)
        rng=np.random.default_rng(20261112+100*si+int(binid));vals=[]
        for _ in range(1000):
            idx=rng.integers(0,n,n);vals.append(float(ag['sum'].to_numpy()[idx].sum()/ag['count'].to_numpy()[idx].sum()))
        bins.append({'side':side,'bin':int(binid),'midpoint':(int(binid)+.5)/10,'n':len(g),'sources':n,'mean_jaccard':float(g.content_jaccard.mean()),'ci':np.quantile(vals,[.025,.975]).tolist()})
q=df[df.query_cosine>=.5].copy()
X=sm.add_constant(pd.DataFrame({'ai':(q.side=='AI_English').astype(float),'log_previous_words':np.log1p(q.previous_words),'log_current_words':np.log1p(q.current_words)}))
fit=sm.OLS(q.content_jaccard,X).fit(cov_type='cluster',cov_kwds={'groups':q.source_key})
assert np.allclose(fit.params['ai'],expected['regression']['content_jaccard']['ai_coefficient'])
result={'summary':summary,'bins':bins,'regression':{'n':len(q),'ai_coefficient':float(fit.params['ai']),'ci':fit.conf_int().loc['ai'].tolist()},'source_summary_matches_supplied':True,'all_pairs':len(df),'missing_feedback':int(df.closer_to_feedback.isna().sum()),'excluded_by_similarity_gate':len(df)-len(q)}
(OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(df),'all_supplied_summary_estimates_and_intervals_match':True,'regression_n':len(q),'bin_estimates':len(bins)}))
