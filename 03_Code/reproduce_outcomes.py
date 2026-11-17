"""Rebuild outcome regressions. Run --smoke before full execution.

python 03_Code/reproduce_outcomes.py --stage all --bootstrap 1000
Stages: regression, contrasts, hierarchical, all. Each stage has checkpoints.
"""
from pathlib import Path
import sys,argparse,json,hashlib,time,warnings
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
import outcome_models as m

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf-8');tmp.replace(p)
def fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()+Path(m.__file__).read_bytes()+(ROOT/'02_Data/main_tasks.json').read_bytes()).hexdigest()
def check_numerics():
    from scipy.optimize import check_grad
    import statsmodels.api as sm
    rng=np.random.default_rng(8501);X=np.c_[np.ones(100),rng.normal(size=(100,2))];y=(rng.random(100)<expit(X@np.array([.3,-.6,.5]))).astype(int);b=np.array([.2,.1,-.3])
    error=check_grad(lambda b:-m.likelihood(b,X,y,np.ones(100),True)[0],lambda b:-m.likelihood(b,X,y,np.ones(100),True)[1],b)
    assert error<2e-5
    ours=m.fit_logit(X,y);ind=sm.GLM(y,X,family=sm.families.Binomial()).fit();assert np.max(np.abs(ours['beta']-ind.params))<1e-6
    separated=(X[:,1]>0).astype(int);f=m.fit_logit(X,separated,firth=True);cal=m.flic(f['beta'],X,separated);assert np.isfinite(f['beta']).all() and abs(expit(X@cal).mean()-separated.mean())<1e-10
    w=rng.integers(0,4,len(y));idx=np.repeat(np.arange(len(y)),w)
    fw=m.fit_logit(X,y,w=w,firth=True);fe=m.fit_logit(X[idx],y[idx],firth=True)
    assert np.max(np.abs(fw['beta']-fe['beta']))<1e-6
    for badX,bady in [(X[:0],y[:0]),(X[:1],y[:1]),(X,np.zeros(len(y)))]:
        try:m.fit_logit(badX,bady);raise AssertionError('empty/single class fit accepted')
        except ValueError:pass
    return dict(firth_gradient_error=error,MLE_statsmodels_max_error=float(np.max(np.abs(ours['beta']-ind.params))),separation_finite=True,FLIC_calibrated=True,frequency_weight_equivalence=True,empty_single_class_rejected=True)

def regression(rows,X,y1,y2,groups,folder,B,smoke):
    output={};selection={'y1':np.ones(len(rows),bool),'y2':y1==1}
    for key,mask in selection.items():
        path=folder/(key+'_regression.json')
        if path.exists():output[key]=read(path);continue
        xx=X[mask];yy=(y1 if key=='y1' else y2)[mask];gg=groups[mask];firth=key=='y2';fit=m.fit_logit(xx,yy,firth=firth)
        b=fit['beta'];predbeta=m.flic(b,xx,yy) if firth else b;point=m.ame(predbeta,xx)
        if firth:
            profile=m.profiles(fit,xx,yy,indices=[2,18] if smoke else None)
            coefs=[dict(name=m.NAMES[j],**profile[m.NAMES[j]]) for j in ([2,18] if smoke else range(1,len(b)))]
        else:
            cov=m.clustered_cov(b,xx,yy,gg);se=np.sqrt(np.diag(cov));pp=2*norm.sf(np.abs(b/se))
            coefs=[dict(name=m.NAMES[j],beta=float(b[j]),OR=float(np.exp(b[j])),CI95_OR=np.exp([b[j]-1.96*se[j],b[j]+1.96*se[j]]).tolist(),p=float(pp[j])) for j in range(1,len(b))]
        for c,h,q in zip(coefs,m.p_adjust([c['p'] for c in coefs]),m.p_adjust([c['p'] for c in coefs],'bh')):c['holm_p']=float(h);c['bh_q']=float(q)
        keys,codes=np.unique(gg,return_inverse=True);boots=[];failures=[]
        for rep in range(B):
            cp=folder/'bootstrap'/f'{key}_{rep:04}.json'
            if cp.exists():v=read(cp)
            else:
                rng=np.random.default_rng(20261105+rep);counts=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys));w=counts[codes]
                try:
                    fb=m.fit_logit(xx,yy,w=w,firth=firth,start=b);pb=m.flic(fb['beta'],xx,yy,w) if firth else fb['beta'];v=dict(rep=rep,success=True,ame=m.ame(pb,xx,w).tolist())
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:v=dict(rep=rep,success=False,error=str(e))
                put(cp,v)
            if v['success']:boots.append(v['ame'])
            else:failures.append(v)
            if (rep+1)%100==0:print(key,'bootstrap',rep+1,'/',B,flush=True)
        assert len(boots)>=max(2,.8*B)
        interval=np.quantile(boots,[.025,.975],axis=0)
        for c in coefs:
            j=m.NAMES.index(c['name'])-1;c['AME']=float(point[j]);c['AME_CI95']=interval[:,j].tolist()
        output[key]=dict(tasks=len(yy),sources=len(keys),positives=int(yy.sum()),predictors=m.NAMES[1:],coefficients=coefs,beta=b.tolist(),prediction_beta=predbeta.tolist(),mean_prediction=float(expit(xx@predbeta).mean()),observed_rate=float(yy.mean()),bootstrap_requested=B,bootstrap_success=len(boots),bootstrap_failed=len(failures),failures=failures,fit_diagnostics={k:v for k,v in fit.items() if k not in ['beta','information']})
        put(path,output[key]);assert read(path)==output[key]
    # Prespecified production x opening-constraint interaction, source-cluster interval.
    inter=X[:,m.NAMES.index('need_A')]*X[:,m.NAMES.index('constraints')];xx=np.c_[X,inter]
    fit=m.fit_logit(xx,y1);cov=m.clustered_cov(fit['beta'],xx,y1,groups);b=fit['beta'][-1];se=np.sqrt(cov[-1,-1])
    output['interaction']=dict(OR=float(np.exp(b)),CI95_OR=np.exp([b-1.96*se,b+1.96*se]).tolist(),p=float(2*norm.sf(abs(b/se))))
    put(folder/'regression_results.json',output)
    return dict(y1_tasks=output['y1']['tasks'],y2_tasks=output['y2']['tasks'],bootstrap_y1=output['y1']['bootstrap_success'],bootstrap_y2=output['y2']['bootstrap_success'])

def contrasts(rows,folder,B,smoke):
    contrasts=[('G1_then_G3','y2'),('G1_then_G3','y1'),('G2','y2'),('G4','y2'),('constraints','y1'),('G1_then_G3_WildChat','y2')]
    output=[]
    for exposure,outcome in contrasts:
        name=exposure+'_'+outcome;path=folder/(name+'.json')
        if path.exists():output.append(read(path));continue
        current=[r for r in rows if r['dataset']=='WildChat'] if exposure.endswith('_WildChat') else rows
        result,state=m.contrast(current,exposure.replace('_WildChat',''),outcome);result['sample']='WildChat' if exposure.endswith('_WildChat') else 'all';selected,X,y,T,groups=state
        keys,codes=np.unique(groups,return_inverse=True);indices=[np.flatnonzero(codes==j) for j in range(len(keys))];boot=[];failed=[]
        for rep in range(B):
            cp=folder/'bootstrap'/f'{name}_{rep:04}.json'
            if cp.exists():value=read(cp)
            else:
                rng=np.random.default_rng(20261105+rep);draw=rng.integers(0,len(keys),len(keys));idx=np.concatenate([indices[j] for j in draw]);xx=X[idx];tt=T[idx];yy=y[idx]
                try:
                    nuisance=m.fit_nuisance(xx,tt);e=np.clip(m.predict_nuisance(nuisance,xx),.01,.99)
                    w1=tt/e;w0=(1-tt)/(1-e);o1=tt*(1-e);o0=(1-tt)*e
                    value=dict(success=True,IPW=float(np.dot(w1,yy)/w1.sum()-np.dot(w0,yy)/w0.sum()),ATO=float(np.dot(o1,yy)/o1.sum()-np.dot(o0,yy)/o0.sum()))
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:value=dict(success=False,error=str(e))
                put(cp,value)
            if value['success']:boot.append([value['IPW'],value['ATO']])
            else:failed.append(dict(rep=rep,**value))
            if (rep+1)%200==0:print(name,'bootstrap',rep+1,'/',B,flush=True)
        assert len(boot)>=max(2,.8*B)
        intervals=np.quantile(boot,[.025,.975],axis=0)
        result.update(IPW_CI95=intervals[:,0].tolist(),ATO_CI95=intervals[:,1].tolist(),bootstrap_requested=B,bootstrap_success=len(boot),bootstrap_failures=failed)
        put(path,result);assert read(path)==result;output.append(result)
    put(folder/'process_associations.json',output)
    if not smoke:assert output[0]['tasks']==1090 and output[0]['exposed']==416 and output[0]['exposed_positive']==397 and output[0]['control_positive']==599
    return dict(contrasts=len(output),G1_G3=output[0]['AIPW'])

def hierarchical(rows,X,y1,y2,groups,folder,smoke):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    result={}
    for name,mask in [('y1',np.ones(len(y1),bool)),('y2',y1==1)]:
        path=folder/(name+'_hierarchical.json')
        if path.exists():result[name]=read(path);continue
        xx=X[mask];yy=(y1 if name=='y1' else y2)[mask];gg=groups[mask];base=m.fit_logit(xx,yy,firth=name=='y2')['beta']
        cp=folder/(name+'_glmm.json')
        if cp.exists():glmm=read(cp)
        else:glmm=m.random_intercept(xx,yy,gg,start=base,nodes=15 if smoke else 35,penalty_sd=2.5 if name=='y2' else None);put(cp,glmm)
        check=None if smoke else m.random_intercept(xx,yy,gg,start=np.array(glmm['beta']),nodes=55,penalty_sd=2.5 if name=='y2' else None)
        if check is not None:
            glmm['quadrature_check']=dict(nodes=55,sigma_difference=check['sigma']-glmm['sigma'],max_beta_difference=float(np.max(np.abs(np.array(check['beta'])-glmm['beta']))))
            assert abs(check['sigma']-glmm['sigma'])<.03
        keys,codes=np.unique(gg,return_inverse=True);Z=csr_matrix((np.ones(len(gg)),(np.arange(len(gg)),codes)),shape=(len(gg),len(keys)))
        scale=np.maximum(xx.std(axis=0),.2);scale[0]=1;xs=xx/scale
        model=BinomialBayesMixedGLM(yy,xs,Z,np.zeros(len(keys),int),vcp_p=.5,fe_p=2)
        mean=np.r_[base*scale,0.,np.zeros(len(keys))];sd=np.full(len(mean),.3)
        with warnings.catch_warnings(record=True) as ws:
            fitted=model.fit_vb(mean=mean,sd=sd,fit_method='L-BFGS-B',minim_opts={'maxiter':2000,'ftol':1e-11,'gtol':1e-5})
        assert fitted.optim_retvals.success, str(fitted.optim_retvals.message)
        beta=fitted.fe_mean/scale;se=fitted.fe_sd/scale;sigma=float(np.exp(fitted.vcp_mean[0]))
        bayes=dict(method='mean-field variational Bayes',fixed_effect_prior='Normal(0, 2^2) on scaled columns',log_sigma_prior='Normal(0, 0.5^2)',column_scales=scale.tolist(),beta=beta.tolist(),OR=np.exp(beta).tolist(),CI95_OR=np.exp(np.c_[beta-1.96*se,beta+1.96*se]).tolist(),sigma_median=sigma,ICC_at_sigma_median=sigma*sigma/(sigma*sigma+np.pi*np.pi/3),converged=bool(fitted.optim_retvals.success),warnings=[str(w.message) for w in ws])
        result[name]=dict(tasks=len(yy),sources=len(keys),names=m.NAMES,GLMM=glmm,Bayesian=bayes);put(path,result[name])
    put(folder/'hierarchical_results.json',result)
    return dict(models=4,y1_sigma=result['y1']['GLMM']['sigma'],y2_sigma=result['y2']['GLMM']['sigma'])

def run(stage,smoke,B,output):
    sig=fingerprint();mode='smoke' if smoke else 'full';folder=output/sig[:10]/mode/stage;folder.mkdir(parents=True,exist_ok=True);start=time.time()
    if not smoke:
        gate=read(output/sig[:10]/'smoke'/stage/'readback.json');assert gate['passed'] and gate['signature']==sig
    tasks=read(ROOT/'02_Data/main_tasks.json')
    if smoke:
        # Engineering subset: fixed random order, extended only until both designs have full rank.
        rng=np.random.default_rng(1501);order=rng.permutation(len(tasks))
        for n in [2400]:
            subset=[tasks[i] for i in order[:n]];rr,xx,yy1,yy2,gg=m.encode(subset)
            if np.linalg.matrix_rank(xx)==21 and np.linalg.matrix_rank(xx[yy1==1])==21:tasks=subset;break
        else:raise RuntimeError('no full rank engineering subset within 2500 rows')
    rows,X,y1,y2,groups=m.encode(tasks)
    put(folder/'input_summary.json',dict(tasks=len(rows),sources=len(set(groups)),determinable=int(y1.sum()),attained=int(y2.sum()),unknown_openings=sum(r['opening'] in ['UNCLEAR','MIXED'] for r in rows),predictors=m.NAMES[1:],signature=sig))
    with threadpool_limits(limits=1):
        tests=check_numerics() if smoke else read(output/sig[:10]/'smoke'/stage/'readback.json')['numerical_checks']
        details={'regression':regression,'contrasts':lambda r,x,a,b,g,f,B,s:contrasts(r,f,B,s),'hierarchical':lambda r,x,a,b,g,f,B,s:hierarchical(r,x,a,b,g,f,s)}[stage](rows,X,y1,y2,groups,folder,5 if smoke else B,smoke)
    record=dict(signature=sig,passed=True,stage=stage,smoke=smoke,numerical_checks=tests,details=details,seconds=time.time()-start,API_calls=0)
    put(folder/'readback.json',record);assert read(folder/'readback.json')==record
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--stage',choices=['all','regression','contrasts','hierarchical'],default='all');p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--output',type=Path,default=ROOT/'05_Results/outcome_rebuild');a=p.parse_args()
    for stage in ['regression','contrasts','hierarchical'] if a.stage=='all' else [a.stage]:
        if not a.smoke:
            run(stage,True,a.bootstrap,a.output);run(stage,True,a.bootstrap,a.output)
        run(stage,a.smoke,a.bootstrap,a.output)
