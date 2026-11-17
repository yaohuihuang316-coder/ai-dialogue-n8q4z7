"""Reproduce descriptive counts and frozen external next-function prediction.

Run: python 03_Code/reproduce.py [--smoke] [--refit] [--output PATH]
No network calls. Results and resumable checkpoints are written to --output.
"""
from pathlib import Path
import argparse, collections, hashlib, json, time, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
import model_core as core
import descriptive

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '02_Data'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def put(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
def build(tasks, lag):
    out = []
    for t in tasks:
        r = t['row']
        v = [np.array([int(g in s.split('+')) for g in ['G1','G2','G3','G4','G5']], float) for s in r['sequence5']]
        for i in range(lag, len(v)):
            out.append(dict(group_id=str(r.get('source_order', r['source_key'])).zfill(8), x1=v[i-1], x2=v[i-2] if lag == 2 else None, target=v[i]))
    return out
def describe(tasks):
    rows = [t['row'] for t in tasks]
    return dict(sources=len({r['source_key'] for r in rows}), tasks=len(rows),
                followup_tasks=sum(r['steps'] > 0 for r in rows), followups=sum(r['steps'] for r in rows),
                outcome_counts=dict(collections.Counter(r.get('outcome','not_in_prediction_export') for r in rows)),
                function_sets=dict(collections.Counter(s for r in rows for s in r['sequence5'])),
                multi_function_positions=sum('+' in s for r in rows for s in r['sequence5']))
def run(smoke, refit, output):
    start = time.time()
    files = [Path(__file__), Path(core.__file__), Path(descriptive.__file__)] + sorted(DATA.glob('*.json'))
    sig = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    folder = output / ('smoke' if smoke else 'full')
    if not smoke:
        gate = read(output / 'smoke' / 'readback.json')
        assert gate['passed'] and gate['signature'] == sig
    main = read(DATA / 'main_tasks.json'); external = read(DATA / 'external_tasks.json')
    old = [t for t in external if t['row']['batch'] == 'original_342']
    new = [t for t in external if t['row']['batch'] == 'additional_200_v13']
    assert not {t['row']['source_key'] for t in main} & {t['row']['source_key'] for t in external}
    assert not {t['row']['source_key'] for t in old} & {t['row']['source_key'] for t in new}
    if smoke: main, old, new = main[:40], old[:40], new[:40]
    scopes = dict(old342=old, new200=new, combined542=old+new)
    descriptions = dict(main=describe(main), **{k: describe(v) for k,v in scopes.items()})
    put(folder / 'sample_counts.json', descriptions)
    assert read(folder / 'sample_counts.json') == descriptions
    patterns=descriptive.task_patterns(main)
    pairs=read(DATA/'comparison_pairs.json')
    shared=descriptive.shared_strategies(pairs[:40] if smoke else pairs)
    put(folder/'functional_patterns.json',patterns);put(folder/'shared_strategies.json',shared)
    if not smoke:
        assert (patterns['distinct_complete_paths'],patterns['singleton_path_types'],patterns['adjacent_pairs'],patterns['same_state_pairs']) == (1468,1227,7776,3399)
        assert shared['TREC2014']['excluded_pairs']==174 and shared['AI_English']['excluded_pairs']==316
    frozen = read(DATA / 'frozen_function_models.json')
    expected = read(DATA / 'expected_external_scores.json')
    results = {}; resumed = 0; refit_errors = []
    with threadpool_limits(limits=1):
        for lag in [1,2]:
            train = build(main, lag); models = ['M0','M1'] if lag == 1 else ['M0','M1','M2','U2']
            parameters = []
            for j in range(5):
                y = np.array([r['target'][j] for r in train]); groups = [r['group_id'] for r in train]
                base = core.marginal(y, core.weights(groups), len(set(groups)))
                pars = {'M0': base}
                for model in models[1:]:
                    d = frozen[str(lag)][j]['diagnostics'][model]
                    if refit or smoke:
                        theta, p0, actual = core.fit(core.design(train, model), y, groups)
                        if smoke: d = actual
                        elif theta is not None:
                            error = float(np.max(np.abs(theta - np.array(d['coefficients']))))
                            assert error < 1e-8, (lag, j, model, error)
                            refit_errors.append(error)
                    pars[model] = d
                parameters.append(pars)
            for scope, tasks in scopes.items():
                key = f'{scope}_lag{lag}'; path = folder / (key + '.json')
                if path.exists():
                    saved = read(path); assert saved['signature'] == sig
                    results[key] = saved['result']; resumed += 1; continue
                test = build(tasks, lag); losses = []
                for j, pars in enumerate(parameters):
                    probs = {'M0': np.full(len(test), pars['M0'])}
                    for model in models[1:]:
                        d = pars[model]
                        if d.get('success'):
                            theta = np.array(d['coefficients'])
                            probs[model] = expit(theta[0] + core.design(test, model)[:,d['kept_columns']] @ theta[1:])
                        else: probs[model] = probs['M0']
                    rec = [dict(group_id=r['group_id'], y=int(r['target'][j]), probabilities={m:float(probs[m][i]) for m in models}) for i,r in enumerate(test)]
                    losses.append({m:core.group_losses(rec,m) for m in models})
                gs = sorted(losses[0]['M0'])
                metrics = {m:dict(zip(['Brier','log_loss'],np.mean([np.mean(list(ll[m].values()),axis=0) for ll in losses],axis=0).tolist())) for m in models}
                diffs = []
                for a,b in [('M1','M0')] + ([('M2','M1'),('M2','U2')] if lag == 2 else []):
                    values = np.array([np.mean([ll[a][s]-ll[b][s] for ll in losses],axis=0) for s in gs])
                    rng = np.random.default_rng(20261105); B = 50 if smoke else 2000
                    boot = np.array([values[rng.integers(0,len(gs),len(gs))].mean(axis=0) for _ in range(B)])
                    diffs.append(dict(comparison=a+'-'+b,delta_Brier=float(values[:,0].mean()),CI95_Brier=np.quantile(boot[:,0],[.025,.975]).tolist(),delta_log_loss=float(values[:,1].mean()),CI95_log_loss=np.quantile(boot[:,1],[.025,.975]).tolist(),bootstrap=B,seed=20261105))
                result = dict(sources=len(gs),positions=len(test),metrics=metrics,differences=diffs)
                if not smoke:
                    want = expected[key]
                    assert result['sources'] == want['sources'] and result['positions'] == want['positions']
                    for model in models:
                        for metric,value in metrics[model].items(): assert abs(value-want['metrics'][model][metric]) < 1e-12
                    for actual,wanted in zip(diffs,want['differences']):
                        for field in ['delta_Brier','delta_log_loss','CI95_Brier','CI95_log_loss']: assert np.allclose(actual[field],wanted[field],atol=1e-12,rtol=0), (key,field)
                put(path,dict(signature=sig,result=result)); assert read(path)['result'] == result
                results[key] = result
    put(folder/'external_scores.json', results)
    put(folder/'readback.json',dict(signature=sig,passed=True,smoke=smoke,resumed_checkpoints=resumed,refit=refit,maximum_refit_coefficient_error=max(refit_errors,default=0),seconds=time.time()-start))
    print(json.dumps(read(folder/'readback.json')))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--refit',action='store_true'); parser.add_argument('--output',type=Path,default=ROOT/'05_Results')
    args=parser.parse_args()
    # Full execution always checks a small write/read/resume run first.
    if not args.smoke:
        run(True,args.refit,args.output); run(True,args.refit,args.output)
    run(args.smoke,args.refit,args.output)
