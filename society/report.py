"""
Study 8 in human units: the evolved society's lives, marriages and children, side by side with
human figures, plus the essay's "thinking late" experiment.
Usage: python report.py   (~1 min on 8 cores) -> report.json, printed report

Each seed evolves the main world to year 3000, then is copied: one copy carries on as before
("continuous"), the other only takes in evidence about a partner when the relationship is already
near its exit line ("late"). Both run to year 4000. Human-scale numbers come from "continuous".
Vocabulary: a "pairing" is any pair; a "marriage" is a pair in which both reached commitment
(the rung at which children become possible); a "divorce" is a marriage ended by exit.
"""
import os, json, copy
from dataclasses import replace
from math import erf, sqrt
from multiprocessing import Pool
import numpy as np
import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(8)
SPLIT, END = 3000, 4000


def pct(F):
    """Match quality as a percentile of all possible pairings (a random stranger is 50)."""
    return 100 * 0.5 * (1 + np.vectorize(erf)(np.asarray(F) / sqrt(2)))


def finish(s):
    for r in s.pairs:
        if r['end'] is None: r['end'], r['reason'] = s.year, 'ongoing'
    return {'pairs': s.pairs, 'lives': s.lives, 'kids': s.kids_out}


def job(seed):
    s = M.Society(M.World(seed=seed, record_from=SPLIT))
    for _ in range(SPLIT): s.step()
    late = copy.deepcopy(s); late.w = replace(late.w, late_attention=True)
    for _ in range(END - SPLIT): s.step(); late.step()
    return finish(s), finish(late)


def demography(lives):
    age = np.array([l['age'] for l in lives])
    ad = [l for l in lives if l['age'] >= 16]
    done = [l for l in lives if l['age'] >= 46]
    comm = np.array([l['committed'] for l in done])
    return {
        'life_expectancy_at_birth': age.mean(),
        'died_before_16_pct': 100 * (age < 16).mean(),
        'life_expectancy_at_16': 16 + (age[age >= 16] - 16).mean(),
        'children_per_completed_life': np.mean([l['births'] for l in done]),
        'median_age_first_partner': np.median([l['first_pair'] for l in ad if l['first_pair'] >= 0]),
        'median_age_first_marriage': np.median([l['first_commit'] for l in ad if l['first_commit'] >= 0]),
        'ever_married_pct': 100 * (comm >= 1).mean(),
        'married_once_pct': 100 * (comm == 1).mean(),
        'married_2plus_pct': 100 * (comm >= 2).mean(),
        'marriages_per_completed_life': comm.mean(),
    }


def relationships(pairs):
    d = np.array([p['end'] - p['start'] for p in pairs]); why = np.array([p['reason'] for p in pairs])
    married = np.array([p['t_commit'] is not None for p in pairs])
    F = np.array([p['fit'] for p in pairs])
    m = married & (why != 'ongoing')
    out = {
        'pairings_that_never_marry_pct': 100 * (~married).mean(),
        'unmarried_pairings_ending_by_breakup_pct': 100 * (why[~married] == 'exit').mean(),
        'median_years_to_marriage': np.median([p['t_commit'] for p in pairs if p['t_commit'] is not None]),
        'median_years_to_devotion': np.median([p['t_devote'] for p in pairs if p['t_devote'] is not None]),
        'reached_devotion_pct_of_marriages': 100 * np.mean([p['t_devote'] is not None for p in pairs if p['t_commit'] is not None]),
        'marriages_ending_in_divorce_pct': 100 * (why[m] == 'exit').mean(),
        'marriages_ending_in_widowhood_pct': 100 * (why[m] == 'death').mean(),
        'median_marriage_length_divorced': np.median(d[m & (why == 'exit')]),
        'median_marriage_length_widowed': np.median(d[m & (why == 'death')]),
        'years_first_doubt_to_divorce': np.median([p['end'] - p['start'] - p['t_first_drop'] for p in pairs
                                                    if p['t_commit'] is not None and p['reason'] == 'exit'
                                                    and p['t_first_drop'] is not None]),
    }
    # divorce hazard per year of marriage-pairing, among marriages (deaths and ongoing censored)
    haz = []
    for t in range(1, 31):
        risk = (married & (d >= t)).sum(); ex = (married & (d == t) & (why == 'exit')).sum()
        haz.append(1000 * ex / max(risk, 1))
    out['divorces_per_1000_marriages_by_year'] = haz
    # selection: match quality on a 0-100 percentile scale
    out['match_quality'] = {
        'random_stranger': 50.0,
        'all_pairings': pct(F).mean(),
        'marriages': pct(F[married]).mean(),
        'marriages_intact_at_10y': pct(F[married & (d >= 10)]).mean(),
        'marriages_intact_at_25y': pct(F[married & (d >= 25)]).mean(),
        'divorced_marriages': pct(F[married & (why == 'exit')]).mean(),
    }
    # learning, not decay: what partners believed over time, against the truth
    def side(p, s, key, i):
        v = p[key + s]
        return v[i] if len(v) > abs(i) else None
    groups = {'divorced': [p for p in pairs if p['t_commit'] is not None and p['reason'] == 'exit'],
              'intact_25y': [p for p in pairs if p['t_commit'] is not None and p['end'] - p['start'] >= 25]}
    learn = {}
    for gname, ps in groups.items():
        row = {'true_match': pct([p['fit'] for p in ps]).mean(),
               'true_reliability': 100 * np.mean([M.sig(p['ph_b' if s == 'a' else 'ph_a'][0]) for p in ps for s in 'ab'])}
        for label, i in (('year1', 0), ('year3', 2), ('final', -1)):
            if gname == 'intact_25y' and label == 'final': i = 24
            fv = [side(p, s, 'f', i) for p in ps for s in 'ab']
            vv = [side(p, s, 'v', i) for p in ps for s in 'ab']
            wv = [side(p, s, 'w', i) for p in ps for s in 'ab']
            row[f'felt_value_{label}'] = 100 * np.mean([x for x in vv if x is not None])
            row[f'believed_match_{label}'] = pct([x for x in fv if x is not None]).mean()
            row[f'believed_reliability_{label}'] = 100 * np.mean([M.sig(x) for x in wv if x is not None])
        learn[gname] = row
    out['learning'] = learn
    return out


def children(kids):
    out = {}
    for status in ('together', 'exit', 'death'):
        k = [x for x in kids if x['parents'] == status]
        out[status] = {'n': len(k), 'survived_to_16_pct': 100 * np.mean([x['survived'] for x in k]) if k else None}
    return out


def attention(pairs, lo, hi):
    """Years spent in poor and good marriages, and how they end."""
    ms = [p for p in pairs if p['t_commit'] is not None and p['reason'] != 'ongoing']
    poor = [p for p in ms if p['fit'] <= lo]; good = [p for p in ms if p['fit'] >= hi]
    f = lambda ps: {'years': np.mean([p['end'] - p['start'] for p in ps]),
                    'divorced_pct': 100 * np.mean([p['reason'] == 'exit' for p in ps]),
                    'children_raised': np.mean([p['raised'] for p in ps])}
    return {'poor': f(poor), 'good': f(good),
            'children_raised_per_marriage': np.mean([p['raised'] for p in ms])}


def ci(v):
    v = np.array(v, float)
    return [round(float(v.mean()), 2), round(float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), 2)]


def agg(dicts):
    """Mean +/- 95% CI across seeds for every number in a nested dict (lists element-wise)."""
    d0 = dicts[0]
    if isinstance(d0, dict): return {k: agg([d[k] for d in dicts]) for k in d0}
    if isinstance(d0, list): return [round(float(np.mean(x)), 2) for x in zip(*dicts)]
    if d0 is None: return None
    return ci(dicts)


HUMAN = {   # comparison figures; ranges, not point targets
    'life_expectancy_at_birth': ('21-37', 'hunter-gatherers (Gurven & Kaplan 2007)'),
    'died_before_16_pct': ('~49', 'hunter-gatherers, death before puberty (Volk & Atkinson 2013)'),
    'life_expectancy_at_16': ('~54', 'hunter-gatherers at 15 (Gurven & Kaplan 2007)'),
    'children_per_completed_life': ('4-7', 'forager total fertility (Gurven & Kaplan 2007)'),
    'marriages_ending_in_divorce_pct': ('40-50', 'US first marriages; forager rates vary widely'),
    'divorce_peak': ('year ~4', 'divorce mode across 58 societies (Fisher 1989)'),
}

if __name__ == '__main__':
    with Pool() as pool:
        runs = pool.map(job, SEEDS)
    cont = [r[0] for r in runs]; late = [r[1] for r in runs]
    # poor / good = bottom / top quartile of true match among continuous-arm marriages, per seed
    att_c, att_l = [], []
    for c, l in zip(cont, late):
        F = np.array([p['fit'] for p in c['pairs'] if p['t_commit'] is not None])
        lo, hi = np.percentile(F, [25, 75])
        att_c.append(attention(c['pairs'], lo, hi)); att_l.append(attention(l['pairs'], lo, hi))
    rep = {
        'demography': agg([demography(c['lives']) for c in cont]),
        'relationships': agg([relationships(c['pairs']) for c in cont]),
        'children': agg([children(c['kids']) for c in cont]),
        'thinking_late': {'continuous': agg(att_c), 'late': agg(att_l)},
        'human': HUMAN,
    }
    json.dump(rep, open(os.path.join(HERE, 'report.json'), 'w'), indent=1)

    def show(title, d, ind='  '):
        print(title)
        for k, v in d.items():
            if isinstance(v, dict): show(ind + k, v, ind + '  ')
            else:
                h = f"   [human: {HUMAN[k][0]}, {HUMAN[k][1]}]" if k in HUMAN else ''
                print(f"{ind}{k}: {v}{h}")
    for k in ('demography', 'relationships', 'children', 'thinking_late'):
        show(k.upper(), rep[k])
