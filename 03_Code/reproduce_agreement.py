"""Reproduce the reported V13 five-function agreement using archived statistic functions."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
NBOOT=2000
SEED=20261106
def kalpha_nom(M):
    """M: units×coders 数组，nan=缺失；nominal Krippendorff α"""
    vals = []
    Do = 0.0
    for row in M:
        r = row[~np.isnan(row)]
        if len(r) < 2:
            continue
        m = len(r)
        c = Counter(r.tolist())
        Do += (m * m - sum((v * v for v in c.values()))) / (m - 1)
        vals.extend(r.tolist())
    n = len(vals)
    if n < 2:
        return np.nan
    cnt = Counter(vals)
    De = (n * n - sum((v * v for v in cnt.values()))) / (n - 1)
    return np.nan if De == 0 else 1 - Do / De

def cohen(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = (x[ok], y[ok])
    if len(x) == 0:
        return np.nan
    po = np.mean(x == y)
    cats = np.unique(np.concatenate([x, y]))
    pe = sum((np.mean(x == c) * np.mean(y == c) for c in cats))
    return np.nan if pe == 1 else (po - pe) / (1 - pe)

def pair_counts(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.array([np.sum(ok & (a == 0) & (b == 0), axis=0), np.sum(ok & (a == 0) & (b == 1), axis=0), np.sum(ok & (a == 1) & (b == 0), axis=0), np.sum(ok & (a == 1) & (b == 1), axis=0)], float)

def count_stats(c):
    nn, ny, yn, yy = c
    n = c.sum(axis=0)
    disagree = ny + yn
    pos = ny + yn + 2 * yy
    neg = 2 * n - pos
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = 1 - disagree * (2 * n - 1) / (pos * neg)
        po = (nn + yy) / n
        pe = ((yn + yy) * (ny + yy) + (nn + ny) * (nn + yn)) / n ** 2
        kap = (po - pe) / (1 - pe)
        f1 = 2 * yy / (2 * yy + ny + yn)
    return {'alpha': alpha, 'kappa': kap, 'agreement': po, 'positive_agreement_F1': f1}

def layer(a, b, ids, names):
    counts = pair_counts(a, b)
    st = count_stats(counts)
    for j in range(len(names)):
        assert np.isclose(st['alpha'][j], kalpha_nom(np.c_[a[:, j], b[:, j]]), equal_nan=True)
        assert np.isclose(st['kappa'][j], cohen(a[:, j], b[:, j]), equal_nan=True)
    clusters = sorted(set(ids))
    cc = np.array([pair_counts(a[np.array(ids) == k], b[np.array(ids) == k]) for k in clusters])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(clusters), (NBOOT, len(clusters)))
    bst = count_stats(cc[draws].sum(axis=1).transpose(1, 0, 2))
    macro = np.nanmean(bst['alpha'], axis=1)
    per = []
    for j, l in enumerate(names):
        vals = bst['alpha'][:, j]
        good = vals[np.isfinite(vals)]
        per.append({'label': l, 'paired_n': int(counts[:, j].sum()), 'GPT_yes': int(counts[2:, j].sum()), 'DeepSeek_yes': int(counts[[1, 3], j].sum()), **{k: float(v[j]) for k, v in st.items()}, 'alpha_ci95': np.percentile(good, [2.5, 97.5]).tolist() if len(good) else None, 'valid_bootstrap': len(good), 'confusion_00_01_10_11': counts[:, j].astype(int).tolist()})
    complete = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    x, y = (a[complete], b[complete])
    u = np.sum((x == 1) | (y == 1), axis=1)
    inter = np.sum((x == 1) & (y == 1), axis=1)
    jacc = np.divide(inter, u, out=np.ones_like(inter, dtype=float), where=u > 0)
    return {'positions': len(a), 'sources': len(clusters), 'paired_cells': int(counts.sum()), 'excluded_cells': int(a.size - counts.sum()), 'macro_alpha': np.nanmean(st['alpha']), 'macro_alpha_ci95': np.nanpercentile(macro, [2.5, 97.5]).tolist(), 'macro_alpha_finite_labels': int(np.isfinite(st['alpha']).sum()), 'micro_agreement': float(np.trace(np.array([[counts[0].sum(), counts[1].sum()], [counts[2].sum(), counts[3].sum()]])) / counts.sum()), 'micro_positive_agreement_F1': float(count_stats(counts.sum(axis=1))['positive_agreement_F1']), 'complete_set_positions': int(complete.sum()), 'exact_set_agreement': float(np.mean(np.all(x == y, axis=1))) if len(x) else None, 'mean_set_jaccard': float(np.mean(jacc)) if len(x) else None, 'per_label': per}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'02_Data/agreement_function_pairs.json').read_text('utf-8'))
    rows=data['rows'];a=np.array([r['GPT'] for r in rows],float);b=np.array([r['DeepSeek'] for r in rows],float)
    actual=layer(a,b,[r['case_id'] for r in rows],data['labels'])
    expected=json.loads((root/'05_Results/agreement_functions_expected.json').read_text('utf-8'))
    for key in ['positions','sources','paired_cells','excluded_cells','macro_alpha','micro_agreement']:
        assert np.isclose(actual[key],expected[key]),key
    assert np.allclose(actual['macro_alpha_ci95'],expected['macro_alpha_ci95'])
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.generic):return v.item()
        return v
    out=root/'outputs/agreement';out.mkdir(parents=True,exist_ok=True)
    path=out/'agreement_functions.json';path.write_text(json.dumps(clean(actual),indent=2,allow_nan=False),'utf-8')
    saved=json.loads(path.read_text('utf-8'));assert saved['paired_cells']==1591
    print(json.dumps({'passed':True,'positions':321,'sources':50,'agreement':actual['micro_agreement'],'macro_alpha':actual['macro_alpha'],'readback':str(path)}))
