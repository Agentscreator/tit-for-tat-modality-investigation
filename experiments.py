"""
Runs the simplified replication, the falsification tests, and the human-like v2 world.
Usage: python experiments.py   (~5 min on 8 cores) -> results.json
"""
import os, json
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 2, 3, 4, 5]
CONDITIONS = {
    'v1':           M.V1,                                   # original world (Phases 1-4)
    'v1_costly':    replace(M.V1, cooldown=30),             # F2: exit actually costs something
    'v1_high':      M.V1_HIGH_STAKES,                       # Phase 6 world
    'v2_no_human':  replace(M.V2, human=False),             # F3: Phase 6 + inconsistent partners
    'v2':           M.V2,                                   # + negativity bias, doubt floor, options
}
LOGGED = {'v2_no_human', 'v2'}


def job(args):
    name, seed = args
    hist, genomes, logs = M.evolve(CONDITIONS[name], seed, log_last=10 if name in LOGGED else 0)
    return name, seed, hist, genomes, logs


def ci(v):
    v = np.array(v, float)
    return [float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))]


def fmt(m, e, d=2):
    return f"{m:.{d}f} ± {e:.{d}f}"


if __name__ == '__main__':
    with Pool() as pool:
        runs = pool.map(job, [(n, s) for n in CONDITIONS for s in SEEDS])
    by = {n: [r for r in runs if r[0] == n] for n in CONDITIONS}
    out = {'evolved': {}, 'payoff_per_round': {}}

    # ---- evolved genomes and payoffs ----
    print("EVOLVED GENOMES (mean of last 10 generations, ± 95% CI across 5 seeds)")
    print(f"{'condition':>12} " + " ".join(f"{k:>13}" for k in M.GENES))
    for n, rs in by.items():
        tails = [r[2][-10:] for r in rs]
        out['evolved'][n] = {k: ci([np.mean([h['genome'][k] for h in t]) for t in tails]) for k in M.GENES}
        kinds = tails[0][0]['payoff'].keys()
        out['payoff_per_round'][n] = {k: ci([np.mean([h['payoff'][k] for h in t]) for t in tails]) for k in kinds}
        print(f"{n:>12} " + " ".join(f"{fmt(*out['evolved'][n][k]):>13}" for k in M.GENES))
    print("\nPAYOFF PER ROUND BY KIND")
    for n, d in out['payoff_per_round'].items():
        print(f"{n:>12} " + "  ".join(f"{k} {fmt(*v)}" for k, v in d.items()))

    rng = np.random.default_rng(0)
    pops = {n: [g for r in rs for g in r[3]] for n, rs in by.items()}
    mean_g = {n: {k: float(np.mean([g[k] for g in gs])) for k in M.GENES} for n, gs in pops.items()}

    # ---- F1: does tolerated betrayal keep growing with history? ----
    print("\nF1  ROUNDS OF PURE BETRAYAL BEFORE LEAVING, BY GOOD ROUNDS FIRST (median over evolved genomes)")
    Gs = [0, 10, 20, 40, 80, 160, 320]
    out['lag'] = {}
    for n in ('v1', 'v2'):
        sample = pops[n][::5]
        row = []
        for G in Gs:
            res = [M.trial(g, G, 1.0, rng, horizon=3000) for g in sample]
            fin = [x for x in res if x is not None]
            row.append({'G': G, 'median': float(np.median(fin)) if fin else None,
                        'never_pct': 100 * (1 - len(fin) / len(res))})
        mg = mean_g[n]
        lim = M.lag_limit(mg['tolerance'], mg['neg_bias'], mg['doubt_floor'])
        out['lag'][n] = {'rows': row, 'limit_mean_genome': None if np.isinf(lim) else float(lim)}
        print(f"  {n:>4}: " + "  ".join(f"G={r['G']}: {r['median']:.0f}" if r['median'] else f"G={r['G']}: -" for r in row)
              + (f"   (closed-form limit for mean genome: {lim:.1f})" if not np.isinf(lim) else "   (no limit: grows forever)"))

    # ---- intermittent betrayal, 20 good rounds first ----
    print("\nINTERMITTENT BETRAYAL: % WHO LEAVE WITHIN 1000 ROUNDS (20 good rounds first)")
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    out['intermittent'] = {}
    print(f"{'condition':>12} {'cutoff':>7} " + " ".join(f"q={q:<4}" for q in qs))
    for n in ('v1_high', 'v2_no_human', 'v2'):
        sample = pops[n][::2]
        row = []
        for q in qs:
            res = [M.trial(g, 20, q, rng) for g in sample]
            row.append(100 * np.mean([x is not None for x in res]))
        cut = M.cutoff(mean_g[n]['tolerance'], mean_g[n]['neg_bias'])
        out['intermittent'][n] = {'q': qs, 'leave_pct': row, 'cutoff_mean_genome': cut}
        print(f"{n:>12} {cut:>7.2f} " + " ".join(f"{v:>5.0f}%" for v in row))

    # ---- v2 social findings: who stays with bad partners, and does a bad start scar? ----
    print("\nWHO STAYS WITH A BAD PARTNER (spells with defectors, inconsistent partners q>=0.5)")
    out['spells'] = {}
    for n in ('v2_no_human', 'v2'):
        spells = [s for r in by[n] for lg in r[4] for s in lg['spells'] if s[0][4] != 'defector']
        if not spells:
            continue
        vals = np.median([s[0][2] for s in spells])
        groups = {'young (paired before round 40)': [s for s in spells if s[0][1] < 40],
                  'older (paired at round 40+)':    [s for s in spells if s[0][1] >= 40],
                  'own value above median':          [s for s in spells if s[0][2] >= vals],
                  'own value below median':          [s for s in spells if s[0][2] < vals]}
        out['spells'][n] = {}
        print(f"  {n}")
        for label, ss in groups.items():
            left = 100 * np.mean([s[2] for s in ss])
            dur = float(np.median([s[1] for s in ss]))
            out['spells'][n][label] = {'n': len(ss), 'left_pct': float(left), 'median_rounds': dur}
            print(f"    {label:32s} n={len(ss):5d}  left {left:5.1f}%  median spell {dur:5.1f} rounds")

    print("\nDOES A BAD FIRST PARTNER LEAVE A MARK? (evolving agents, v2)")
    out['first_partner'] = {}
    for n in ('v2_no_human', 'v2'):
        ag = [a for r in by[n] for lg in r[4] for a in lg['agents'] if a[0] is not None]
        res = {}
        for flag, label in ((False, 'good first partner'), (True, 'bad first partner')):
            xs = [a for a in ag if a[0] == flag]
            res[label] = {'n': len(xs), 'value_change': float(np.mean([a[2] - a[1] for a in xs])),
                          'payoff_per_round': float(np.mean([a[3] for a in xs]))}
        out['first_partner'][n] = res
        print(f"  {n}: " + "   ".join(f"{k}: n={v['n']}, value {v['value_change']:+.1f}, payoff/round {v['payoff_per_round']:.2f}"
                                    for k, v in res.items()))

    out['mean_genome'] = mean_g
    json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    json.dump({n: g for n, g in pops.items()}, open(os.path.join(HERE, 'genomes.json'), 'w'))
    print("\nsaved results.json, genomes.json")
