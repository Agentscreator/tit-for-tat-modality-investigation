"""
Phase 5: specialists vs generalists.
Each evolving agent splits a fixed budget of 100 points across three channels (trust,
attraction, provision) at birth. A partner is worth what its channels add up to under an
aggregation rule: conjunctive (near-min power mean) or additive (arithmetic mean). That
worth sets the agent's place in matching, whether partners reject it on first contact, and
how much a cooperative round with it pays. Evolution picks the split.

Points turn into channel score with curvature gamma: score = (points/100)**gamma.
gamma < 1 diminishing returns, 1 linear, > 1 compounding (specialising pays more).
Prediction: maximising a power mean (exponent rho) of x**gamma under a fixed budget favours
balance when gamma*rho < 1 and a single channel when gamma*rho > 1.
"""
import os, json
import numpy as np
from core import Agent, decide, payoffs, match

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 2, 3, 4, 5]
RULES = {'conjunctive': -5.0, 'additive': 1.0}     # power-mean exponent rho
GAMMAS = [0.7, 1.0, 1.5]
CHANNELS = ['trust', 'attraction', 'provision']
# behaviour fixed at the Phase 2 consensus genome, so only the allocation evolves
BEHAVIOUR = {'threshold': 18.7, 'prior_variance': 2.06,
             'belief_exit_threshold': -0.41, 'retaliate_prob': 0.70}


def allocation(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def worth(alloc, rho, gamma):
    s = np.maximum(alloc, 1e-9) ** gamma
    return float(np.mean(s ** rho) ** (1 / rho))


def run_generation(logit_pop, n_tft, n_defect, n_rounds, rho, gamma, seed, perception_noise=0.10):
    rng = np.random.default_rng(seed)
    balanced = np.full(3, 1 / 3)
    raw = [worth(allocation(l), rho, gamma) for l in logit_pop] + \
          [worth(balanced, rho, gamma)] * (n_tft + n_defect)
    scale = 50 / np.mean(raw)                       # population mean worth = 50, as in Phases 1-4
    pop = []
    for i, w in enumerate(raw):
        kind = 'evo' if i < len(logit_pop) else ('tft' if i < len(logit_pop) + n_tft else 'defector')
        pop.append(Agent(i, float(np.clip(w * scale, 1, 100)),
                         dict(BEHAVIOUR) if kind == 'evo' else None, kind))
    idx = {a.id: a for a in pop}

    for _ in range(n_rounds):
        for a in pop:
            a.age += 1
            if a.cooldown > 0: a.cooldown -= 1
        match(pop, rng)
        done = set()
        for a in pop:
            if a.partner is None or a.id in done: continue
            b = idx[a.partner]
            m1, m2 = decide(a, b, rng), decide(b, a, rng)
            o1, o2 = m2, m1
            if o1 in ('cooperate', 'defect') and rng.random() < perception_noise:
                o1 = 'defect' if o1 == 'cooperate' else 'cooperate'
            if o2 in ('cooperate', 'defect') and rng.random() < perception_noise:
                o2 = 'defect' if o2 == 'cooperate' else 'cooperate'
            p1, p2, ended = payoffs(m1, m2)
            if m1 == m2 == 'cooperate':             # cooperation pays in proportion to partner's worth
                p1, p2 = 3.0 * b.value / 50, 3.0 * a.value / 50
            a.payoff += p1; b.payoff += p2
            a.last_seen[b.id] = o1; b.last_seen[a.id] = o2
            if ended:
                a.partner = b.partner = None
                a.cooldown = b.cooldown = 1         # free exit
            done.add(a.id); done.add(b.id)
    return [a.payoff for a in pop if a.kind == 'evo']


def evolve(rho, gamma, seed, n_gen=100, n_evo=200, n_tft=40, n_defect=80, n_rounds=80,
           elite_frac=0.15, immigrant_frac=0.05, sigma=0.3):
    mrng = np.random.default_rng(seed)
    pop = [mrng.normal(0, 1, 3) for _ in range(n_evo)]
    hist = []
    for g_i in range(n_gen):
        pay = run_generation(pop, n_tft, n_defect, n_rounds, rho, gamma, seed * 100000 + g_i)
        order = np.argsort(-np.array(pay) + mrng.normal(0, 1e-9, len(pay)))
        shares = np.sort([allocation(l) for l in pop], axis=1)    # smallest, middle, largest
        hist.append({'largest': float(shares[:, 2].mean()), 'smallest': float(shares[:, 0].mean()),
                     'specialist_pct': float(100 * np.mean(shares[:, 2] > 0.6)),
                     'payoff': float(np.mean(pay))})
        n_elite, n_imm = max(2, int(elite_frac * n_evo)), max(1, int(immigrant_frac * n_evo))
        elites = [pop[i] for i in order[:n_elite]]
        new = [e.copy() for e in elites]
        while len(new) < n_evo - n_imm:
            new.append(elites[mrng.integers(0, n_elite)] + mrng.normal(0, sigma, 3))
        while len(new) < n_evo:
            new.append(mrng.normal(0, 1, 3))
        pop = new
    return hist


def ci(v):
    v = np.array(v, float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


if __name__ == '__main__':
    out = {}
    print(f"{'rule':>12} {'gamma':>6} {'gamma*rho':>10} {'predicted':>11} {'largest share':>15} "
          f"{'smallest share':>15} {'% specialists':>14}")
    for rule, rho in RULES.items():
        for gamma in GAMMAS:
            finals = []
            for s in SEEDS:
                tail = evolve(rho, gamma, s)[-10:]
                finals.append({k: float(np.mean([h[k] for h in tail])) for k in tail[0]})
            res = {k: ci([f[k] for f in finals]) for k in finals[0]}
            gr = gamma * rho
            pred = 'balance' if gr < 1 else ('specialise' if gr > 1 else 'neutral')
            out[f"{rule}|{gamma}"] = {'rho': rho, 'gamma': gamma, 'predicted': pred, **res}
            print(f"{rule:>12} {gamma:>6.1f} {gr:>10.1f} {pred:>11} "
                  f"{res['largest'][0]:>8.2f} ±{res['largest'][1]:.2f} "
                  f"{res['smallest'][0]:>8.2f} ±{res['smallest'][1]:.2f} "
                  f"{res['specialist_pct'][0]:>8.1f} ±{res['specialist_pct'][1]:.1f}", flush=True)
    json.dump(out, open(os.path.join(HERE, 'phase5.json'), 'w'), indent=2)
    print("\nsaved phase5.json")
