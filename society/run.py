"""
Study 8: evolve the society under three conditions and test the pre-registered predictions.
Usage: python run.py   (~3 min on 8 cores) -> results.json
"""
import os, json
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(8)
CONDITIONS = {
    'main':        M.World(),
    'flat_warmth': M.World(warm_matters=False),     # P4 control: warmth changes nothing you get
    'safe_bold':   M.World(bold_costs=False),       # P2 control: boldness is free
}
GRID = np.linspace(-4, 4, 161)
NAMES = 'WKBF'


def job(args):
    name, seed = args
    out = M.Society(replace(CONDITIONS[name], seed=seed)).run()
    return name, seed, out


def curves(genes, traits, rng):
    """Population-mean preference along each dimension, others held at the population mean."""
    g = np.array(genes); base = np.append(np.array(traits).mean(0), 0.0)
    res = {}
    for c in range(4):
        x = np.tile(base, (len(g), 1)); ys = []
        for v in GRID:
            x[:, c] = v; ys.append(M.score(g, x).mean())
        res[NAMES[c]] = np.array(ys)
    return res


def shape_fit(genes, traits, rng, n_agents=300, n_prof=400):
    """Held-out R^2 of a linear vs an ideal-point (additive quadratic) model of each agent's scores."""
    g = np.array(genes); t = np.array(traits)
    pick = rng.choice(len(g), min(n_agents, len(g)), replace=False)
    r2 = {'linear': [], 'ideal': []}
    for i in pick:
        prof = np.column_stack([t[rng.integers(len(t), size=n_prof)], rng.normal(0, 1, n_prof)])
        y = M.score(np.repeat(g[i:i + 1], n_prof, 0), prof)
        tr, te = slice(0, n_prof // 2), slice(n_prof // 2, None)
        for k, X in (('linear', prof), ('ideal', np.column_stack([prof, prof ** 2]))):
            X1 = np.column_stack([np.ones(n_prof), X])
            beta = np.linalg.lstsq(X1[tr], y[tr], rcond=None)[0]
            res = y[te] - X1[te] @ beta
            r2[k].append(1 - res.var() / max(y[te].var(), 1e-12))
    return {k: float(np.mean(v)) for k, v in r2.items()}


def pair_arrays(pairs):
    d = np.array([p['end'] - p['start'] for p in pairs])
    r = np.array([p['reason'] for p in pairs])
    top = np.array([p['top'] for p in pairs])
    wa = np.array([[p['ph_a'][0], p['ph_b'][0]] for p in pairs]).mean(1)
    fit = np.array([p['fit'] for p in pairs])
    return d, r, top, wa, fit


def hazard(d, r, bins):
    """Exit hazard per year within each bin of durations; deaths and ongoing pairs are censored."""
    out = []
    for lo, hi in bins:
        risk = sum(((d >= t)).sum() for t in range(lo, hi + 1))
        ex = ((d >= lo) & (d <= hi) & (r == 'exit')).sum()
        out.append(float(ex / max(risk, 1)))
    return out


def landscape(pairs):
    """Ground truth: children raised to 16 per pair, regressed on one partner's traits (both views)."""
    rows, y = [], []
    for p in pairs:
        if p['reason'] == 'ongoing': continue
        for own, oth in ((p['ph_a'], p['ph_b']), (p['ph_b'], p['ph_a'])):
            W, K, B = oth
            rows.append([1, W, K, B, W * W, K * K, B * B, W * K, p['fit'], *own])
            y.append(p['raised'])
    X, y = np.array(rows), np.array(y, float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    names = ['const', 'W', 'K', 'B', 'W2', 'K2', 'B2', 'WxK', 'F', 'own_W', 'own_K', 'own_B']
    return dict(zip(names, beta.round(4).tolist()))


def deciles(pairs):
    """Nonparametric ground truth: children raised by decile of one partner's trait."""
    t = np.array([p['ph_b'] for p in pairs if p['reason'] != 'ongoing'] +
                 [p['ph_a'] for p in pairs if p['reason'] != 'ongoing'])
    y = np.array([p['raised'] for p in pairs if p['reason'] != 'ongoing'] * 2, float)
    out = {}
    for i, c in enumerate('WKB'):
        q = np.percentile(t[:, i], np.linspace(0, 100, 11))
        b = np.clip(np.searchsorted(q, t[:, i], side='right') - 1, 0, 9)
        out[c] = [float(y[b == k].mean()) for k in range(10)]
    return out


def decline(curve, lo, hi):
    """How far the evolved preference falls from its peak to the population's 90th percentile,
    as a share of its range over the 10th-90th percentiles. 0 = monotone up to p90."""
    m = (GRID >= lo) & (GRID <= hi)
    seg = curve[m]
    rng_ = seg.max() - seg.min()
    return float((seg.max() - seg[-1]) / rng_) if rng_ > 1e-9 else 0.0


def ci(v):
    v = np.array(v, float)
    return [float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))]


if __name__ == '__main__':
    with Pool() as pool:
        runs = pool.map(job, [(n, s) for n in CONDITIONS for s in SEEDS])
    BINS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 14), (15, 19), (20, 29), (30, 49)]
    res = {'bins': BINS, 'grid': GRID.tolist(), 'conditions': {}}
    for name in CONDITIONS:
        rs = [o for n, s, o in runs if n == name]
        rng = np.random.default_rng(0)
        per = []
        for o in rs:
            t = np.array(o['final_traits']); g = np.array(o['final_genes'])
            pct = {c: np.percentile(t[:, i], [10, 90]).tolist() for i, c in enumerate('WKB')}
            pct['F'] = [-1.2816, 1.2816]
            cv = curves(g, t, rng)
            peak = {c: float(GRID[np.argmax(cv[c])]) for c in NAMES}
            d, r, top, wa, fit = pair_arrays(o['pairs'])
            com = top >= 2
            surv = {}
            for T in (0, 5, 10, 20):
                m = d >= T
                surv[T] = {'W': float(wa[m].mean()), 'F': float(fit[m].mean()),
                           'W_committed': float(wa[m & com].mean()), 'F_committed': float(fit[m & com].mean())}
            per.append({
                'peak': peak, 'pct': pct,
                'interior': {c: bool(pct[c][0] < peak[c] < pct[c][1]) for c in NAMES},
                'curve': {c: cv[c].round(4).tolist() for c in NAMES},
                'genes': dict(zip([s_[0] for s_ in M.SPEC], g.mean(0).round(3).tolist())),
                'traits': t.mean(0).round(3).tolist(),
                'shape_r2': shape_fit(g, t, rng),
                'hazard_all': hazard(d, r, BINS), 'hazard_committed': hazard(d[com], r[com], BINS),
                'exit_share': float((r == 'exit').mean()),
                'exit_share_committed': float((r[com] == 'exit').mean()),
                'survivors': surv, 'landscape': landscape(o['pairs']), 'deciles': deciles(o['pairs']),
                'decline': {c: decline(cv[c], *pct[c]) for c in NAMES},
                'series': o['series'],
            })
        summ = {
            'zeta': ci([p['genes']['zeta'] for p in per]),
            'peak': {c: ci([p['peak'][c] for p in per]) for c in NAMES},
            'p90': {c: ci([p['pct'][c][1] for p in per]) for c in NAMES},
            'p10': {c: ci([p['pct'][c][0] for p in per]) for c in NAMES},
            'interior_share': {c: float(np.mean([p['interior'][c] for p in per])) for c in NAMES},
            'weight': {c: ci([p['genes'][f'weight_{c}'] for p in per]) for c in NAMES},
            'traits': {c: ci([p['traits'][i] for p in per]) for i, c in enumerate('WKB')},
            'shape_r2': {k: ci([p['shape_r2'][k] for p in per]) for k in ('linear', 'ideal')},
            'hazard_all': np.mean([p['hazard_all'] for p in per], 0).round(4).tolist(),
            'hazard_committed': np.mean([p['hazard_committed'] for p in per], 0).round(4).tolist(),
            'exit_share': ci([p['exit_share'] for p in per]),
            'exit_share_committed': ci([p['exit_share_committed'] for p in per]),
            'survivors': {T: {k: ci([p['survivors'][T][k] for p in per]) for k in per[0]['survivors'][T]}
                          for T in (0, 5, 10, 20)},
            'landscape': {k: ci([p['landscape'][k] for p in per]) for k in per[0]['landscape']},
            'deciles': {c: np.mean([p['deciles'][c] for p in per], 0).round(3).tolist() for c in 'WKB'},
            'decline': {c: ci([p['decline'][c] for p in per]) for c in NAMES},
            'selection_dW_0_20': ci([p['survivors'][20]['W'] - p['survivors'][0]['W'] for p in per]),
            'selection_dF_0_20': ci([p['survivors'][20]['F'] - p['survivors'][0]['F'] for p in per]),
            'zeta_series': np.mean([[x['genes']['zeta'] for x in p['series']] for p in per], 0)[::10].round(3).tolist(),
            'mean_curve': {c: np.mean([p['curve'][c] for p in per], 0).round(4).tolist() for c in NAMES},
            'climb': ci([p['genes']['climb'] for p in per]), 'drop': ci([p['genes']['drop'] for p in per]),
        }
        res['conditions'][name] = {'summary': summ, 'runs': per}
        print(f"\n=== {name}")
        for k, v in summ.items():
            if k != 'mean_curve': print(f"  {k}: {v}")
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'))
