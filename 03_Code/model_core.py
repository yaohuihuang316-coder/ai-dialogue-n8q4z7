"""Frozen mathematical functions reused from the study analysis."""
import collections
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def weights(groups):
    counts = collections.Counter(groups)
    return np.array([1.0 / (len(counts) * counts[g]) for g in groups])

def marginal(y, w, G):
    return float((np.dot(w, y) * G + 0.5) / (G + 1.0))

def objective(theta, X, y, w):
    z = theta[0] + X @ theta[1:]
    delta = w * (expit(z) - y)
    loss = np.dot(w, np.logaddexp(0.0, z) - y * z) + 0.005 * np.dot(theta[1:], theta[1:])
    grad = np.r_[delta.sum(), X.T @ delta + 0.01 * theta[1:]]
    return (float(loss), grad)

def fit(X, y, groups, force_failure=False):
    w = weights(groups)
    p0 = marginal(y, w, len(set(groups)))
    constant = np.ptp(X, axis=0) == 0
    cols = np.flatnonzero(~constant)
    diag = {'kept_columns': cols.tolist(), 'constant_columns': np.flatnonzero(constant).tolist(), 'train_groups': len(set(groups)), 'train_rows': len(y), 'train_positives': int(y.sum())}
    if len(set(y)) < 2:
        return (None, p0, {**diag, 'reason': 'single_class', 'actual_model': 'M0'})
    if not len(cols):
        return (None, p0, {**diag, 'reason': 'constant_features', 'actual_model': 'M0'})
    xx = X[:, cols]
    initial = np.r_[np.log(p0 / (1 - p0)), np.zeros(len(cols))]
    opt = minimize(objective, initial, args=(xx, y, w), method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-06})
    ok = bool(opt.success and np.isfinite(opt.fun) and np.isfinite(opt.x).all() and (not force_failure))
    diag.update(success=ok, loss=float(opt.fun), message=str(opt.message), iterations=int(opt.nit), coefficients=opt.x.tolist(), reason='fitted' if ok else 'solver_failed')
    return (opt.x if ok else None, p0, diag)

def design(ss, model):
    x1 = np.array([s['x1'] for s in ss])
    x2 = np.array([s['x2'] for s in ss]) if model in ['M2', 'U2'] else None
    return np.concatenate([x1, x2], axis=1) if model == 'M2' else (x1 + x2) / 2 if model == 'U2' else x1

def group_losses(records, model):
    gs = collections.defaultdict(list)
    for r in records:
        p = r['probabilities'][model]
        assert np.isfinite(p) and 0 <= p <= 1, 'invalid probability'
        y = r['y']
        clip = np.clip(p, 1e-06, 1 - 1e-06)
        gs[r['group_id']].append(((p - y) ** 2, -y * np.log(clip) - (1 - y) * np.log(1 - clip)))
    return {g: np.mean(v, axis=0) for g, v in gs.items()}
