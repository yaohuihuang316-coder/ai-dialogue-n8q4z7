"""Outcome regression rebuilt from the published specification, November 2026.

Unpenalized logistic regression, Firth likelihood and FLIC, profile intervals,
source-cluster sandwich and bootstrap, random-intercept likelihood, and AIPW.
No manuscript estimates are used during fitting.
"""
from pathlib import Path
import json, collections
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import minimize, brentq
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm, chi2

NAMES=['intercept','queries']+['need_'+g for g in 'ADFMPSXY']+['clear','background','constraints','deliverable_yes','deliverable_partial']+['G'+str(i) for i in range(1,6)]+['Search_Arena']
CONTINUOUS=['queries','G1','G2','G3','G4','G5']

def encode(tasks):
    rows=[t['row'] if 'row' in t else t for t in tasks];X=[]
    for r in rows:
        f=r['opening_fields'];opening=r['opening']
        # Unknown and mixed openings have no named contrast in the original 20-column specification.
        needs=[int(opening not in ['UNCLEAR','MIXED'] and opening.startswith(g)) for g in 'ADFMPSXY']
        X.append([1,r['queries']]+needs+[int(f['goal_clarity']=='clear'),int(f['background_present']=='yes'),int(f['constraints_present']=='yes'),int(f['checkable_deliverable']=='yes'),int(f['checkable_deliverable']=='partial')]+[sum(g in s.split('+') for s in r['sequence5']) for g in ['G1','G2','G3','G4','G5']]+[int(r['dataset']=='Search Arena')])
    y1=np.array([r['outcome'] in ['confirmed','unmet'] for r in rows],int)
    y2=np.array([r['outcome']=='confirmed' for r in rows],int)
    return rows,np.array(X,float),y1,y2,np.array([r['source_key'] for r in rows])

def likelihood(beta,X,y,w,firth=False):
    z=X@beta;p=expit(z);v=p*(1-p)
    info=X.T@((w*v)[:,None]*X)
    ll=np.dot(w,y*z-np.logaddexp(0,z));score=X.T@(w*(y-p))
    if firth:
        try:
            cf=cho_factor(info,lower=True,check_finite=False)
            invx=cho_solve(cf,X.T,check_finite=False).T
        except np.linalg.LinAlgError:return -np.inf,np.zeros(X.shape[1]),info
        ll+=np.log(np.diag(cf[0])).sum()
        leverage=w*v*np.sum(X*invx,axis=1)
        score+=X.T@(leverage*(.5-p))
    return float(ll),score,info

def fit_logit(X,y,w=None,firth=False,start=None,fixed=None,ridge=0.):
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    use=w>0;X=X[use];y=y[use];w=w[use]
    if np.unique(y).size<2:raise ValueError('single outcome class')
    if np.linalg.matrix_rank(X)<X.shape[1]:raise ValueError('rank deficient design')
    scale=np.maximum(np.std(X,axis=0),.2);scale[0]=1;Z=X/scale
    theta=np.zeros(X.shape[1]) if start is None else np.asarray(start)*scale
    if start is None:theta[0]=np.log((np.dot(w,y)+.5)/(np.dot(w,1-y)+.5))
    free=np.arange(X.shape[1])
    if fixed is not None:
        j,value=fixed;theta[j]=value*scale[j];free=free[free!=j]
    def obj(a):
        b=theta.copy();b[free]=a;ll,g,_=likelihood(b,Z,y,w,firth)
        penalty=ridge*np.dot(b[1:],b[1:])/2
        g[1:]-=ridge*b[1:]
        return -ll+penalty,-g[free]
    result=minimize(obj,theta[free],jac=True,method='BFGS',options={'maxiter':500,'gtol':1e-7})
    theta[free]=result.x;beta=theta/scale
    ll,grad,info=likelihood(beta,X,y,w,firth)
    maxscore=float(np.max(np.abs(obj(result.x)[1])))
    ok=bool(np.isfinite(beta).all() and np.isfinite(ll) and maxscore<2e-4)
    if not ok:raise RuntimeError(f'fit not converged: score={maxscore:.3g}; {result.message}')
    return dict(beta=beta,ll=ll,information=info,converged=ok,iterations=int(result.nit),max_scaled_score=maxscore)

def flic(beta,X,y,w=None):
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    offset=X@beta-beta[0]
    intercept=brentq(lambda a:np.dot(w,expit(a+offset)-y),-100,100,xtol=1e-12)
    out=beta.copy();out[0]=intercept
    return out

def ame(beta,X,w=None):
    w=np.ones(len(X)) if w is None else np.asarray(w,float);w=w/w.sum();z=X@beta;out=[]
    for j,name in enumerate(NAMES[1:],1):
        if name in CONTINUOUS:d=expit(z+beta[j])-expit(z)
        elif name.startswith('need_'):
            base=z-X[:,2:10]@beta[2:10]
            d=expit(base+beta[j])-expit(base)
        elif name.startswith('deliverable_'):
            base=z-X[:,13:15]@beta[13:15]
            d=expit(base+beta[j])-expit(base)
        else:d=expit(z+(1-X[:,j])*beta[j])-expit(z-X[:,j]*beta[j])
        out.append(float(np.dot(w,d)))
    return np.array(out)

def clustered_cov(beta,X,y,groups):
    _,_,I=likelihood(beta,X,y,np.ones(len(y)));bread=np.linalg.inv(I)
    keys,codes=np.unique(groups,return_inverse=True);scores=np.zeros((len(keys),X.shape[1]))
    np.add.at(scores,codes,X*(y-expit(X@beta))[:,None])
    correction=len(keys)/(len(keys)-1)*(len(y)-1)/(len(y)-X.shape[1])
    return correction*bread@scores.T@scores@bread

def p_adjust(p,method='holm'):
    p=np.asarray(p);order=np.argsort(p);m=len(p);out=np.zeros(m)
    if method=='holm':out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(m-np.arange(m))))
    else:out[order]=np.minimum(1,np.minimum.accumulate((p[order]*m/(np.arange(m)+1))[::-1])[::-1])
    return out

def profiles(fit,X,y,indices=None):
    beta=fit['beta'];se=np.sqrt(np.diag(np.linalg.inv(fit['information'])));results={};cut=chi2.ppf(.95,1)
    for j in (range(1,len(beta)) if indices is None else indices):
        cache={}
        def prof(value):
            if value not in cache:cache[value]=fit_logit(X,y,firth=True,start=beta,fixed=(j,float(value)))['ll']
            return cache[value]
        null=prof(0.);p=float(chi2.sf(max(0,2*(fit['ll']-null)),1));bounds=[]
        for direction in [-1,1]:
            width=max(.2,2*se[j]);far=beta[j]+direction*width
            for _ in range(16):
                if 2*(fit['ll']-prof(far))>cut:break
                width*=1.7;far=beta[j]+direction*width
            else:raise RuntimeError('profile interval unbounded')
            root=brentq(lambda b:2*(fit['ll']-prof(float(b)))-cut,min(far,beta[j]),max(far,beta[j]),xtol=2e-6)
            bounds.append(float(root))
        results[NAMES[j]]=dict(beta=float(beta[j]),OR=float(np.exp(beta[j])),CI95_OR=np.exp(bounds).tolist(),p=p,profile_fits=len(cache))
    return results

def random_intercept(X,y,groups,start=None,nodes=25,penalty_sd=None):
    # Direct integration over one normally distributed source intercept.
    _,codes=np.unique(groups,return_inverse=True);G=int(codes.max()+1)
    order=np.argsort(codes,kind='stable');codes=codes[order];X=X[order];y=y[order];starts=np.r_[0,np.flatnonzero(np.diff(codes))+1]
    scale=np.maximum(np.std(X,axis=0),.2);scale[0]=1;Z=X/scale
    points,weights=np.polynomial.hermite.hermgauss(nodes);logw=np.log(weights)-.5*np.log(np.pi)
    def obj(theta):
        sigma=np.exp(theta[-1]);u=np.sqrt(2)*sigma*points
        eta=(Z@theta[:-1])[:,None]+u
        residual=y[:,None]-expit(eta)
        group_ll=np.add.reduceat(y[:,None]*eta-np.logaddexp(0,eta),starts,axis=0)+logw
        totals=logsumexp(group_ll,axis=1);posterior=np.exp(group_ll-totals[:,None]);perrow=posterior[codes]
        gb=Z.T@np.sum(residual*perrow,axis=1)
        gs=np.sum(residual*perrow*u)
        value=-float(totals.sum());gradient=-np.r_[gb,gs]
        if penalty_sd is not None:
            raw=theta[:-1]/scale
            value+=.5*np.sum((raw/penalty_sd)**2)
            gradient[:-1]+=raw/(penalty_sd**2*scale)
        return value,gradient
    initial=np.r_[np.zeros(X.shape[1]) if start is None else start*scale,0.]
    opt=minimize(obj,initial,jac=True,method='L-BFGS-B',bounds=[(None,None)]*X.shape[1]+[(-7,3)],options={'maxiter':1200,'ftol':1e-12,'gtol':1e-6,'maxls':40})
    theta=opt.x;grad=obj(theta)[1];ok=bool(opt.success and np.max(np.abs(grad))<.01)
    if not ok:raise RuntimeError(f'GLMM did not converge: {opt.message}; score={np.max(np.abs(grad))}')
    h=1e-4;H=np.column_stack([(obj(theta+np.eye(len(theta))[j]*h)[1]-obj(theta-np.eye(len(theta))[j]*h)[1])/(2*h) for j in range(len(theta))]);H=(H+H.T)/2
    cov=np.linalg.pinv(H,rcond=1e-9);beta=theta[:-1]/scale;se=np.sqrt(np.maximum(0,np.diag(cov)[:-1]))/scale;sigma=float(np.exp(theta[-1]));icc=sigma*sigma/(sigma*sigma+np.pi*np.pi/3)
    separated=[j for j in range(1,X.shape[1]) if set(np.unique(X[:,j]))<={0.,1.} and np.any(X[:,j]==1) and np.unique(y[X[:,j]==1]).size==1]
    intervals=[];odds=[]
    for j in range(len(beta)):
        unstable=(j in separated and penalty_sd is None) or abs(beta[j])>15 or se[j]>30
        odds.append(None if unstable else float(np.exp(beta[j])))
        intervals.append([None,None] if unstable else np.exp([beta[j]-1.96*se[j],beta[j]+1.96*se[j]]).tolist())
    return dict(beta=beta.tolist(),OR=odds,CI95_OR=intervals,separated_columns=separated,sigma=sigma,ICC=icc,MOR=float(np.exp(np.sqrt(2)*norm.ppf(.75)*sigma)),nodes=nodes,penalty_sd=penalty_sd,converged=ok,max_score=float(np.max(np.abs(grad))),objective=-float(opt.fun))

def propensity_design(rows):
    # Same initial attributes and source as the outcome model; length 1/2/3/4/5/6/7+.
    _,X,_,_,_=encode(rows)
    columns=[j for j,n in enumerate(NAMES) if n not in CONTINUOUS]
    base=X[:,columns];length=np.array([r['queries'] for r in rows]);levels=np.minimum(length,7)
    return np.c_[base,*[(levels==k).astype(float) for k in range(2,8)]], [NAMES[j] for j in columns]+['length_'+str(k) for k in range(2,8)]

def independent_columns(X):
    # Retain the intercept and a deterministic left-to-right linearly independent basis.
    kept=[];rank=0
    for j in range(X.shape[1]):
        new=np.linalg.matrix_rank(X[:,kept+[j]])
        if new>rank:kept.append(j);rank=new
    return kept

def fit_nuisance(X,y):
    if np.unique(y).size<2:return None,float((y.sum()+.5)/(len(y)+1)),[],'single_class'
    cols=independent_columns(X);Z=X[:,cols]
    try:
        fit=fit_logit(Z,y)
        if np.max(np.abs(fit['beta']))>20:raise RuntimeError('separation')
        return fit['beta'],None,cols,'MLE'
    except (RuntimeError,np.linalg.LinAlgError,ValueError):
        fit=fit_logit(Z,y,firth=True);b=flic(fit['beta'],Z,y)
        return b,None,cols,'Firth_FLIC'

def predict_nuisance(model,X):
    beta,p,cols,_=model
    return np.full(len(X),p) if beta is None else expit(X[:,cols]@beta)

def process_exposure(row,name):
    seq=[s.split('+') for s in row['sequence5']]
    if name=='G1_then_G3':return int(any('G1' in a and any('G3' in b for b in seq[i+1:]) for i,a in enumerate(seq)))
    if name in ['G2','G4']:return int(any(name in s for s in seq))
    return int(row['opening_fields']['constraints_present']=='yes')

def contrast(rows,exposure,outcome,seed=20261105):
    from sklearn.model_selection import StratifiedGroupKFold
    selected=[r for r in rows if (r['queries']>=3 if exposure=='G1_then_G3' else r['queries']>=2 if exposure in ['G2','G4'] else True) and (r['outcome'] in ['confirmed','unmet'] if outcome=='y2' else True)]
    X,names=propensity_design(selected)
    if exposure=='constraints':j=names.index('constraints');X=np.delete(X,j,axis=1);names.pop(j)
    y=np.array([r['outcome']=='confirmed' if outcome=='y2' else r['outcome'] in ['confirmed','unmet'] for r in selected],int)
    T=np.array([process_exposure(r,exposure) for r in selected]);groups=np.array([r['source_key'] for r in selected])
    nuisance=fit_nuisance(X,T);e=np.clip(predict_nuisance(nuisance,X),.01,.99)
    def weighted(ee,weights=None):
        w=np.ones(len(y)) if weights is None else weights
        a=w*T/ee;b=w*(1-T)/(1-ee);over1=w*T*(1-ee);over0=w*(1-T)*ee
        return [np.dot(a,y)/a.sum()-np.dot(b,y)/b.sum(),np.dot(over1,y)/over1.sum()-np.dot(over0,y)/over0.sum()]
    ipw,ato=weighted(e);pred=np.zeros((len(y),3));folds=[]
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    for k,(tr,te) in enumerate(cv.split(X,2*T+y,groups)):
        assert not set(groups[tr])&set(groups[te])
        ps=fit_nuisance(X[tr],T[tr]);m1=fit_nuisance(X[tr][T[tr]==1],y[tr][T[tr]==1]);m0=fit_nuisance(X[tr][T[tr]==0],y[tr][T[tr]==0])
        pred[te]=np.c_[predict_nuisance(ps,X[te]),predict_nuisance(m1,X[te]),predict_nuisance(m0,X[te])]
        folds.append(dict(fold=k,test_sources=len(set(groups[te])),train_sources=len(set(groups[tr])),ps_fit=ps[3],outcome1_fit=m1[3],outcome0_fit=m0[3],source_disjoint=True))
    ec=np.clip(pred[:,0],.01,.99);mu1=pred[:,1];mu0=pred[:,2]
    influence=mu1-mu0+T*(y-mu1)/ec-(1-T)*(y-mu0)/(1-ec);estimate=float(influence.mean())
    keys,codes=np.unique(groups,return_inverse=True);totals=np.bincount(codes,weights=influence-estimate)
    se=float(np.sqrt(len(keys)/(len(keys)-1)*np.sum(totals**2))/len(y))
    def smd(w):
        d=[]
        for j in range(1,X.shape[1]):
            xx=X[:,j];var=(np.var(xx[T==1],ddof=1)+np.var(xx[T==0],ddof=1))/2
            d.append(abs(np.average(xx[T==1],weights=w[T==1])-np.average(xx[T==0],weights=w[T==0]))/np.sqrt(var) if var>0 else 0.)
        return float(max(d,default=0))
    result=dict(exposure=exposure,outcome=outcome,tasks=len(y),sources=len(keys),exposed=int(T.sum()),exposed_positive=int(np.dot(T,y)),control_positive=int(np.dot(1-T,y)),IPW=float(ipw),ATO=float(ato),AIPW=estimate,AIPW_CI95=[estimate-1.96*se,estimate+1.96*se],max_SMD=dict(unweighted=smd(np.ones(len(y))),IPW=smd(np.where(T==1,1/e,1/(1-e))),ATO=smd(np.where(T==1,1-e,e))),propensity_clipped=int(np.sum((predict_nuisance(nuisance,X)<.01)|(predict_nuisance(nuisance,X)>.99))),folds=folds,design_names=names,excluded_tasks=len(rows)-len(selected))
    return result,(selected,X,y,T,groups)
