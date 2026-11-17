"""Optional local re-encoding of the fixed comparison frame.
Input: JSON list of 3,847 rows with unit_id, side, source_key, depth, position,
previous, current, feedback (empty when missing), and archived_lexical.word_Jaccard.
Use the fixed frame, not a new sample. Original text must be obtained separately.
Run smoke twice (second run verifies resume), then full, then analyze.
"""
from pathlib import Path
import sys,os,json,re,hashlib,time,unicodedata,argparse
MODEL_ID='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REV='e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
SEED=20261112
CELLS=[(d,p) for d in range(2,6) for p in range(1,d)]
SIDES=['TREC2014','AI_English']
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    t.replace(p)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def stamp():return time.strftime('%Y-%m-%dT%H:%M:%S')

def event(action,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=stamp(),action=action,**kw),ensure_ascii=False)+'\n')

def smoke_rows(rows):
    # Fixed source/position order, not chosen using similarities or outcomes.
    return [r for s in SIDES for r in [q for q in rows if q['side']==s][:20]]

def embed(mode):
    import numpy as np,torch
    from transformers import AutoTokenizer,AutoModel
    rows=load(OUT/'pairs_LOCAL_ONLY.json')
    if mode=='smoke':rows=smoke_rows(rows)
    else:
        proof=load(OUT/'smoke/readback.json');assert proof['passed'] and proof['resume_verified'] and proof['script_sha256']==sha(Path(__file__))
    texts=sorted({x[k] for x in rows for k in ['previous','current','feedback'] if x[k]})
    dst=OUT/mode;dst.mkdir(exist_ok=True)
    manifest={'hashes':[hashlib.sha256(t.encode()).hexdigest() for t in texts],'model':MODEL_ID,'revision':REV,'max_tokens':128,'script_sha256':sha(Path(__file__))}
    mp=dst/'embedding_manifest.json'
    if mp.exists():assert load(mp)==manifest
    else:put(mp,manifest)
    torch.manual_seed(SEED);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    model=AutoModel.from_pretrained(str(MODEL),local_files_only=True).to(device).eval()
    cp=dst/'embeddings.npy'
    emb=np.load(cp) if cp.exists() else np.zeros((0,384),dtype=np.float32)
    prior=len(emb);start=time.monotonic();truncated=0
    for i in range(prior,len(texts),16):
        batch=texts[i:i+16]
        tokens=tokenizer(batch,padding=True,truncation=True,max_length=128,return_tensors='pt').to(device)
        with torch.inference_mode():
            h=model(**tokens).last_hidden_state
            mask=tokens['attention_mask'].unsqueeze(-1).to(h.dtype)
            v=(h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
            v=torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy()
        emb=np.concatenate([emb,v],axis=0)
        # Every batch is a resumable checkpoint, at most seconds of work.
        with cp.with_suffix('.tmp').open('wb') as f:np.save(f,emb)
        cp.with_suffix('.tmp').replace(cp)
        if (i//16)%20==0:
            put(dst/'progress.json',{'done':len(emb),'total':len(texts),'seconds':time.monotonic()-start,'device':device})
            print(mode,len(emb),'/',len(texts),flush=True)
    check=np.load(cp);assert check.shape==(len(texts),384) and np.isfinite(check).all()
    assert np.allclose(np.linalg.norm(check,axis=1),1,atol=1e-5)
    idx={t:i for i,t in enumerate(texts)};out=[]
    for r in rows:
        a,b=check[idx[r['previous']]],check[idx[r['current']]]
        sim=float(np.clip(a@b,-1,1));fb=float(np.clip(check[idx[r['feedback']]]@b,-1,1)) if r['feedback'] else None
        out.append({'unit_id':r['unit_id'],'source_key':r['source_key'],'side':r['side'],'depth':r['depth'],'position':r['position'],'query_cosine':sim,'feedback_cosine':fb,'closer_to_feedback':fb>sim if fb is not None else None})
    put(dst/'pair_similarities.json',out)
    # Token truncation is reported from uncapped tokenizer lengths, not estimated from characters.
    lengths=tokenizer(texts,truncation=False,add_special_tokens=True)['input_ids']
    truncated=sum(len(t)>128 for t in lengths)
    proof={'passed':True,'script_sha256':sha(Path(__file__)),'pairs':len(rows),'texts':len(texts),'dimensions':384,
        'resumed_vectors':prior,'new_vectors':len(texts)-prior,'seconds':time.monotonic()-start,'device':device,
        'truncated_unique_texts':truncated,'torch':torch.__version__,'model':MODEL_ID,'revision':REV,'checkpoint_readback':True,
        'normalized_finite':True,'pair_output_readback':len(load(dst/'pair_similarities.json'))==len(rows),
        'resume_verified':prior==len(texts)}
    put(dst/'readback.json',proof);event('embed_'+mode,**proof);print(json.dumps(proof),flush=True)

def tokens(s):return re.findall(r'[^\W_]+',unicodedata.normalize('NFKC',s).casefold(),re.UNICODE)

def analyze():
    import numpy as np,pandas as pd
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    import statsmodels.api as sm
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    try:from rapidfuzz.distance import Levenshtein
    except ImportError:Levenshtein=None
    rows=load(OUT/'pairs_LOCAL_ONLY.json');ss={r['unit_id']:r for r in load(OUT/'full/pair_similarities.json')}
    rr=[]
    for r in rows:
        a,b=tokens(r['previous']),tokens(r['current']);sa,sb=set(a)-ENGLISH_STOP_WORDS,set(b)-ENGLISH_STOP_WORDS
        ja=len(sa&sb)/len(sa|sb) if sa|sb else 0.
        added=sb-sa;deleted=sa-sb
        rr.append({**ss[r['unit_id']],'content_jaccard':ja,'archived_jaccard':r['archived_lexical']['word_Jaccard'],
                   'previous_words':len(a),'current_words':len(b),'pure_add':int(bool(added) and not deleted),
                   'both_add_delete':int(bool(added) and bool(deleted)),'no_shared':int(not(sa&sb)),
                   'referring':int(bool(set(b)&{'it','its','this','that','these','those','above'})),
                   'added_fraction':len(added)/len(sb) if sb else 0.,'deleted_fraction':len(deleted)/len(sa) if sa else 0.,
                   'edit_similarity':float(Levenshtein.normalized_similarity(r['previous'].casefold(),r['current'].casefold())) if Levenshtein else None,
                   'log_length_ratio':float(np.log((len(b)+1)/(len(a)+1)))})
    df=pd.DataFrame(rr);df['cell']=[CELLS.index((d,p)) for d,p in zip(df.depth,df.position)]
    df.to_csv(OUT/'full/derived_pair_metrics.csv',index=False,encoding='utf-8-sig')
    metrics=['query_cosine','content_jaccard','archived_jaccard','closer_to_feedback','pure_add','both_add_delete','no_shared','referring']
    summary={};boots={}
    for si,side in enumerate(SIDES):
        q=df[df.side==side].copy();groups=sorted(q.source_key.unique());gi={g:i for i,g in enumerate(groups)}
        sums=np.zeros((len(groups),10,len(metrics)));counts=sums.copy()
        for _,r in q.iterrows():
            for mi,m in enumerate(metrics):
                if pd.notna(r[m]):sums[gi[r.source_key],r.cell,mi]+=float(r[m]);counts[gi[r.source_key],r.cell,mi]+=1
        def calc(w):
            a=np.einsum('i,ijk->jk',w,sums);b=np.einsum('i,ijk->jk',w,counts)
            return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0).mean(0)
        v=calc(np.ones(len(groups)));rng=np.random.default_rng(SEED+si)
        boot=np.array([calc(np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))) for _ in range(1000)])
        boots[side]=boot;summary[side]={'pairs':len(q),'sources':len(groups),'metrics':{}}
        for j,m in enumerate(metrics):
            valid=boot[:,j][np.isfinite(boot[:,j])]
            summary[side]['metrics'][m]={'estimate':float(v[j]) if np.isfinite(v[j]) else None,'ci':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,'bootstrap_valid':len(valid),'known_pairs':int(q[m].notna().sum())}
    differences={}
    for j,m in enumerate(metrics):
        a=boots['AI_English'][:,j]-boots['TREC2014'][:,j];a=a[np.isfinite(a)]
        t=summary['TREC2014']['metrics'][m]['estimate'];v=summary['AI_English']['metrics'][m]['estimate']
        differences[m]={'estimate':v-t if v is not None and t is not None else None,'ci':np.quantile(a,[.025,.975]).tolist() if len(a) else None}
    fitdf=df[df.query_cosine>=.5].copy();fitdf['ai']=(fitdf.side=='AI_English').astype(float)
    X=sm.add_constant(pd.DataFrame({'ai':fitdf.ai,'log_previous_words':np.log1p(fitdf.previous_words),'log_current_words':np.log1p(fitdf.current_words)}))
    regression={}
    for y in ['content_jaccard','archived_jaccard']:
        fit=sm.OLS(fitdf[y],X).fit(cov_type='cluster',cov_kwds={'groups':fitdf.source_key})
        regression[y]={'ai_coefficient':float(fit.params['ai']),'ci':fit.conf_int().loc['ai'].tolist(),'p':float(fit.pvalues['ai']),'n':len(fitdf),'n_by_side':fitdf.side.value_counts().to_dict()}
        weights=1/fitdf.groupby(['side','cell'])['unit_id'].transform('size')
        wf=sm.WLS(fitdf[y],X,weights=weights).fit(cov_type='cluster',cov_kwds={'groups':fitdf.source_key})
        regression[y]['cell_weighted_sensitivity']={'ai_coefficient':float(wf.params['ai']),'ci':wf.conf_int().loc['ai'].tolist()}
    # Figure 8d reconstruction: fixed 0.1-width bins; report source-cluster uncertainty separately from point means.
    df['bin']=np.minimum(np.floor(df.query_cosine.clip(0,1)*10).astype(int),9)
    binned=[]
    for (side,binid),q in df.groupby(['side','bin']):
        binned.append({'side':side,'bin':int(binid),'midpoint':(binid+.5)/10,'n':len(q),'mean_cosine':float(q.query_cosine.mean()),'mean_jaccard':float(q.content_jaccard.mean()),'sources':int(q.source_key.nunique())})
    # Topic comparison is new reconstruction because historical fitting choices are unavailable.
    texts=sorted({x[k] for x in rows for k in ['previous','current','feedback'] if x[k]});ti={t:i for i,t in enumerate(texts)}
    vec=np.load(OUT/'full/embeddings.npy');previous_vec=np.array([vec[ti[r['previous']]] for r in rows])
    km=KMeans(n_clusters=10,n_init=10,random_state=SEED).fit(previous_vec)
    df['topic_cluster']=km.labels_;topic={};shared=[c for c,q in df.groupby('topic_cluster') if q.side.nunique()==2]
    weights=df.topic_cluster.value_counts(normalize=True);mass=float(weights[shared].sum())
    for side in SIDES:
        topic[side]={m:float(sum(weights[c]*df[(df.side==side)&(df.topic_cluster==c)][m].mean() for c in shared)/mass) for m in ['content_jaccard','query_cosine','pure_add']}
    topic['shared_cluster_count']=len(shared);topic['covered_pair_fraction']=mass
    topic['cell_counts']=[{'cluster':int(c),'side':s,'n':int(n)} for (c,s),n in df.groupby(['topic_cluster','side']).size().items()]
    gmm=None
    if Levenshtein:
        features=['content_jaccard','added_fraction','deleted_fraction','edit_similarity','query_cosine','log_length_ratio']
        mat=StandardScaler().fit_transform(df[features]);g=GaussianMixture(n_components=5,n_init=5,random_state=SEED,covariance_type='full').fit(mat)
        df['gmm_cluster']=g.predict(mat)
        gmm={'features':features,'converged':bool(g.converged_),'bic':float(g.bic(mat)),'clusters':[],'shares':{}}
        for c,q in df.groupby('gmm_cluster'):gmm['clusters'].append({'cluster':int(c),'n':len(q),**{m:float(q[m].mean()) for m in features}})
        for side in SIDES:
            q=df[df.side==side];gmm['shares'][side]={str(c):float(np.mean([np.mean(q[q.cell==i].gmm_cluster==c) for i in range(10)])) for c in range(5)}
    df.to_csv(OUT/'full/derived_pair_metrics.csv',index=False,encoding='utf-8-sig')
    historical={'TREC2014':{'query_cosine':.789,'content_jaccard':.484,'closer_to_feedback':.018},'AI_English':{'query_cosine':.455,'content_jaccard':.156,'closer_to_feedback':.360}}
    comparison=[]
    for side,ms in historical.items():
        for m,old in ms.items():
            new=summary[side]['metrics'][m]['estimate'];tol=.0005
            comparison.append({'side':side,'metric':m,'reported':old,'reconstructed':new,'difference':new-old if new is not None else None,'same_at_reported_precision':abs(new-old)<tol if new is not None else False})
    old=-.193;new=regression['content_jaccard']['ai_coefficient']
    comparison.append({'side':'AI_minus_TREC_adjusted','metric':'content_jaccard_cosine_ge_0.5','reported':old,'reconstructed':new,'difference':new-old,'same_at_reported_precision':abs(new-old)<.0005})
    result={'model':MODEL_ID,'revision':REV,'status':'candidate reconstruction, not verified historical run','summary':summary,'differences':differences,'regression':regression,'figure8d_bins':binned,'topic_reconstruction':topic,'gmm_reconstruction':gmm,'comparison_to_reported':comparison,'all_selected_metrics_match':all(x['same_at_reported_precision'] for x in comparison),'excluded_by_similarity_gate':len(df)-len(fitdf),'seed':SEED}
    put(OUT/'full/results.json',result)
    # Edge cases validate the metric, without changing the fixed corpus.
    def jac(a,b):
        a=set(tokens(a))-ENGLISH_STOP_WORDS;b=set(tokens(b))-ENGLISH_STOP_WORDS
        return len(a&b)/len(a|b) if a|b else 0.
    edge={'empty':jac('',''),'singleton_same':jac('research','research'),'singleton_disjoint':jac('research','library')}
    assert edge=={'empty':0.,'singleton_same':1.,'singleton_disjoint':0.}
    put(OUT/'full/analysis_readback.json',{'rows':len(pd.read_csv(OUT/'full/derived_pair_metrics.csv')),'result_readback':bool(load(OUT/'full/results.json')),'edge_cases':edge,'stopword_count':len(ENGLISH_STOP_WORDS),'stopword_sha256':hashlib.sha256('\n'.join(sorted(ENGLISH_STOP_WORDS)).encode()).hexdigest()})
    print(json.dumps(comparison,ensure_ascii=False,indent=2),flush=True);event('analysis_complete',pairs=len(df),all_selected_metrics_match=result['all_selected_metrics_match'])
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['smoke','full','analyze'])
    ap.add_argument('--input',type=Path,required=True,help='Fixed source-text JSON, stored locally')
    ap.add_argument('--model-dir',type=Path,required=True,help='Local files for the named locked model revision')
    ap.add_argument('--output',type=Path,required=True,help='Dedicated new local run directory')
    args=ap.parse_args();OUT=args.output.resolve();MODEL=args.model_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
    rows=load(args.input);assert len(rows)==3847
    assert len({(r['side'],r['source_key'],r['position']) for r in rows})==3847
    target=OUT/'pairs_LOCAL_ONLY.json'
    if target.exists():assert load(target)==rows
    else:put(target,rows)
    if args.command=='analyze':analyze()
    else:embed(args.command)
