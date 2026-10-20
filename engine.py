"""Generation runner + evolutionary loop, with configurable exit friction."""
import numpy as np
from core import (Agent, random_genome, mutate, decide, payoffs, match, GENE_BOUNDS)

def build_population(genomes, n_tft, n_defect, rng):
    pop = []
    n_total = len(genomes) + n_tft + n_defect
    loc = rng.normal(50, 15, size=n_total)          # neighbourhood value baseline
    i = 0
    for g in genomes:
        pop.append(Agent(i, float(np.clip(rng.normal(loc[i], 8), 1, 100)), g, 'evo')); i += 1
    for _ in range(n_tft):
        pop.append(Agent(i, float(np.clip(rng.normal(loc[i], 8), 1, 100)), None, 'tft')); i += 1
    for _ in range(n_defect):
        pop.append(Agent(i, float(np.clip(rng.normal(loc[i], 8), 1, 100)), None, 'defector')); i += 1
    return pop


def run_generation(genomes, n_tft, n_defect, n_rounds, seed, perception_noise=0.0,
                   exit_cost_mode='entangled', flat_cost=2, cost_rate=0.5, max_cost=10):
    """
    exit_cost_mode: 'free' (1 round), 'flat' (fixed), 'entangled' (scales with time together)
    Returns (pop, diagnostics dict)
    """
    rng = np.random.default_rng(seed)
    pop = build_population(genomes, n_tft, n_defect, rng)
    idx = {a.id: a for a in pop}

    exit_costs, exit_tenures = [], []

    for _ in range(n_rounds):
        for a in pop:
            a.age += 1
            if a.cooldown > 0: a.cooldown -= 1
        match(pop, rng)

        done = set()
        for a in pop:
            if a.partner is None or a.id in done: continue
            b = idx[a.partner]
            if b.id in done: continue

            m1, m2 = decide(a, b, rng), decide(b, a, rng)
            # perception noise: what each side *observes* can differ from what was played
            o1, o2 = m2, m1
            if perception_noise > 0:
                if o1 in ('cooperate','defect') and rng.random() < perception_noise:
                    o1 = 'defect' if o1 == 'cooperate' else 'cooperate'
                if o2 in ('cooperate','defect') and rng.random() < perception_noise:
                    o2 = 'defect' if o2 == 'cooperate' else 'cooperate'
            p1, p2, ended = payoffs(m1, m2)
            a.payoff += p1; b.payoff += p2
            a.last_seen[b.id] = o1
            b.last_seen[a.id] = o2

            # track tit-for-tat trapped in mutual defection with a pure defector
            if not ended:
                if a.kind == 'tft' and b.kind == 'defector': a.stuck_rounds += 1
                if b.kind == 'tft' and a.kind == 'defector': b.stuck_rounds += 1

            if ended:
                tenure = a.rounds_with
                if   exit_cost_mode == 'free':      cd = 1
                elif exit_cost_mode == 'flat':      cd = flat_cost
                else:                               cd = int(round(min(max_cost, tenure * cost_rate)))
                exit_costs.append(cd); exit_tenures.append(tenure)
                a.partner = b.partner = None
                a.cooldown = b.cooldown = cd
            else:
                a.rounds_with += 1; b.rounds_with += 1
            done.add(a.id); done.add(b.id)

        mean_val = np.mean([x.value for x in pop])
        for a in pop:
            peer = idx[a.partner].value if a.partner is not None else mean_val
            a.value = float(np.clip(a.value + 0.01 * (peer - a.value), 1, 100))

    diag = {
        'exit_costs': exit_costs,
        'exit_tenures': exit_tenures,
        'tft_payoffs': [a.payoff for a in pop if a.kind == 'tft'],
        'tft_stuck':   [a.stuck_rounds for a in pop if a.kind == 'tft'],
        'evo_payoffs': [a.payoff for a in pop if a.kind == 'evo'],
        'def_payoffs': [a.payoff for a in pop if a.kind == 'defector'],
    }
    return [a for a in pop if a.kind == 'evo'], diag


def a_keys(evo, jitter):
    return [a.payoff + j for a, j in zip(evo, jitter)]


def evolve(n_gen=120, n_evo=200, n_tft=40, n_defect=80, n_rounds=80, seed=0,
           exit_cost_mode='entangled', elite_frac=0.15, immigrant_frac=0.05, sigma=0.10,
           perception_noise=0.10):
    """Tighter elite fraction than the first pass -> stronger selection pressure."""
    mrng = np.random.default_rng(seed)
    genomes = [random_genome(mrng) for _ in range(n_evo)]
    hist = []

    for g_i in range(n_gen):
        evo, diag = run_generation(genomes, n_tft, n_defect, n_rounds,
                                   seed=seed * 100000 + g_i, exit_cost_mode=exit_cost_mode,
                                   perception_noise=perception_noise)
        jitter = mrng.normal(0, 1e-9, size=len(evo))
        ranked = [a for _, a in sorted(zip(a_keys(evo, jitter), evo),
                                       key=lambda t: t[0], reverse=True)]
        hist.append({
            'gen': g_i,
            'evo_mean':  float(np.mean([a.payoff for a in evo])),
            'tft_mean':  float(np.mean(diag['tft_payoffs'])) if diag['tft_payoffs'] else np.nan,
            'def_mean':  float(np.mean(diag['def_payoffs'])) if diag['def_payoffs'] else np.nan,
            'genome':    {k: float(np.mean([a.genome[k] for a in evo])) for k in GENE_BOUNDS},
            'genome_sd': {k: float(np.std ([a.genome[k] for a in evo])) for k in GENE_BOUNDS},
        })

        n = len(ranked)
        n_elite = max(2, int(elite_frac * n))
        n_imm   = max(1, int(immigrant_frac * n))
        elites  = [dict(a.genome) for a in ranked[:n_elite]]
        new = list(elites)
        while len(new) < n - n_imm:
            new.append(mutate(elites[mrng.integers(0, len(elites))], mrng, sigma))
        while len(new) < n:
            new.append(random_genome(mrng))
        genomes = new

    return hist, genomes
