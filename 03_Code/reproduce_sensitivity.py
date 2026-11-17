"""Source-cluster sensitivity checks for the prespecified G1 then G3 association.
Run after reproduce_outcomes.py; all calculations use coded tasks, not reported estimates.
"""
from pathlib import Path
import sys,json,hashlib,argparse,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.special import logit
from scipy.stats import binom
from threadpoolctl import threadpool_limits
import outcome_models as m
from reproduce_outcomes import put,read
ROOT=Path(__file__).resolve().parents[1]

def weighted(rows):
    allrows=[r for r in rows if r['queries']>=3]
    X,_=m.propensity_design(allrows);known=np.array([r['outcome'] in ['confirmed','unmet'] for r in allrows]);T=np.array([m.process_exposure(r,'G1_then_G3') for r in allrows]);y=np.array([r['outcome']=='confirmed' for r in allrows],int)
    obs=m.fit_nuisance(np.c_[X,T],known.astype(int));s=np.clip(m.predict_nuisance(obs,np.c_[X,T]),.01,.99)
    xx=X[known];tt=T[known];yy=y[known];sw=1/s[known]
    # The propensity for the structure is estimated in the full eligible population.
    ps=m.fit_nuisance(X,T);e=np.clip(m.predict_nuisance(ps,xx),.01,.99)
    out=[]
    for w1,w0 in [(sw*tt/e,sw*(1-tt)/(1-e)),(sw*tt*(1-e),sw*(1-tt)*e)]:out.append(float(np.dot(w1,yy)/w1.sum()-np.dot(w0,yy)/w0.sum()))
    return out,dict(eligible_tasks=len(allrows),tasks=int(known.sum()),exposed=int(tt.sum()),sources=len({r['source_key'] for r in allrows}),observation_model=obs[3],propensity_model=ps[3])

def matching(rows):
    results=[]
    for exposure,outcome in [('G1_then_G3','y2'),('G1_then_G3','y1'),('G2','y2'),('G4','y2'),('constraints','y1'),('G1_then_G3_WildChat','y2')]:
        rr=[r for r in rows if r['dataset']=='WildChat'] if exposure.endswith('_WildChat') else rows
        result,state=m.contrast(rr,exposure.replace('_WildChat',''),outcome);selected,X,y,T,g=state
        model=m.fit_nuisance(X,T);e=np.clip(m.predict_nuisance(model,X),.01,.99);z=logit(e);caliper=.2*float(np.std(z,ddof=1));available=set(np.flatnonzero(T==0));pairs=[]
        for i in sorted(np.flatnonzero(T==1),key=lambda i:(-z[i],i)):
            if not available:break
            j=min(available,key=lambda j:(abs(z[j]-z[i]),j))
            if abs(z[j]-z[i])<=caliper:pairs.append((int(i),int(j)));available.remove(j)
        dif=np.array([y[i]-y[j] for i,j in pairs]);positive=int((dif>0).sum());negative=int((dif<0).sum());discord=positive+negative
        sign=1 if dif.mean()>0 else -1;favored=max(positive,negative);gamma=None
        if discord:
            for v in np.arange(1,10.001,.01):
                if binom.sf(favored-1,discord,v/(1+v))>=.05:gamma=float(v);break
        results.append(dict(exposure=exposure,outcome=outcome,pairs=len(pairs),risk_difference=float(dif.mean()),caliper_logit=caliper,discordant_positive=positive,discordant_negative=negative,Rosenbaum_gamma_one_sided=gamma,direction=sign))
    return results

def execute(smoke,B,output):
    signature=hashlib.sha256(Path(__file__).read_bytes()+Path(m.__file__).read_bytes()+(ROOT/'02_Data/main_tasks.json').read_bytes()).hexdigest();folder=output/signature[:10]/('smoke' if smoke else 'full');folder.mkdir(parents=True,exist_ok=True)
    final=folder/'sensitivity_results.json'
    if final.exists():assert read(final)['passed'];print('resume_verified',final,flush=True);return
    start=time.time();rows=m.encode(read(ROOT/'02_Data/main_tasks.json'))[0]
    if smoke:rows=[rows[i] for i in np.random.default_rng(1501).permutation(len(rows))[:2400]]
    if not smoke:assert read(output/signature[:10]/'smoke/sensitivity_results.json')['passed']
    estimate,detail=weighted(rows);keys=sorted({r['source_key'] for r in rows});group={k:[] for k in keys}
    for r in rows:group[r['source_key']].append(r)
    boots=[];fail=[]
    for b in range(B):
        cp=folder/'bootstrap'/f'{b:04}.json'
        if cp.exists():v=read(cp)
        else:
            rng=np.random.default_rng(20261105+b);draw=rng.choice(keys,len(keys));rr=[r for k in draw for r in group[k]]
            try:v=dict(success=True,estimates=weighted(rr)[0])
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as ex:v=dict(success=False,error=str(ex))
            put(cp,v)
        if v['success']:boots.append(v['estimates'])
        else:fail.append(b)
        if (b+1)%200==0:print('selection weighting',b+1,flush=True)
    assert len(boots)>=.8*B;ci=np.quantile(boots,[.025,.975],axis=0)
    matches=matching(rows)
    out=dict(signature=signature,passed=True,smoke=smoke,input_tasks=len(rows),reweighted=dict(**detail,IPW=estimate[0],ATO=estimate[1],IPW_CI95=ci[:,0].tolist(),ATO_CI95=ci[:,1].tolist(),bootstrap_requested=B,bootstrap_success=len(boots),failed_replicates=fail),matching=matches,seconds=time.time()-start)
    put(final,out);assert read(final)==out;print(json.dumps({k:v for k,v in out.items() if k not in ['matching','reweighted']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--output',type=Path,default=ROOT/'05_Results/sensitivity_runs');a=p.parse_args()
    with threadpool_limits(limits=1):
        execute(True,5,a.output);execute(True,5,a.output)
        if not a.smoke:execute(False,a.bootstrap,a.output)
