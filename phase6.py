"""
Phase 6: high-stakes world where partners shape each other.
Same agents and belief mechanism, two changes:
  1. Steeper payoffs. Mutual cooperation pays 4 x (partner value / 50), the sucker loses 4,
     temptation pays 6, mutual defection costs 1. Good partners pay a lot; bad ones cost
     every round you stay, even when you retaliate.
  2. Value coupling. Each round together your value moves 5% toward your partner's, +0.5
     for each cooperation they play and -1 for each defection. Value feeds back into what
     cooperation with you pays and who you get matched with, so partners change each
     other's trajectory.
Evolve under free exit, compare with the baseline world, then rerun the Phase 4
intermittent betrayal test under the new stakes.
"""
import os, json
import numpy as np
from core import Agent, random_genome, mutate, decide, match, GENE_BOUNDS, OBS_NOISE_VAR
from engine import build_population

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 2, 3, 4, 5]
R, T, S, P = 4.0, 6.0, -4.0, -1.0
PULL, LIFT, SCAR = 0.05, 0.5, 1.0


def payoff(m1, m2, v1, v2):
    """Payoffs to (agent 1, agent 2); v1, v2 are their values."""
    if m1 == 'exit' or m2 == 'exit':              return 0.0, 0.0, True
    if m1 == 'cooperate' and m2 == 'cooperate':   return R * v2 / 50, R * v1 / 50, False
    if m1 == 'cooperate' and m2 == 'defect':      return S, T, False
    if m1 == 'defect'    and m2 == 'cooperate':   return T, S, False
    return P, P, False


def run_generation(genomes, n_tft, n_defect, n_rounds, seed, perception_noise=0.10):
    rng = np.random.default_rng(seed)
    pop = build_population(genomes, n_tft, n_defect, rng)
    idx = {a.id: a for a in pop}
    start = {a.id: a.value for a in pop}
    for _ in range(n_rounds):
        for a in pop:
            a.age += 1
            if a.cooldown > 0: a.cooldown -= 1
        match(pop, rng)
        done, moves = set(), {}
        for a in pop:
            if a.partner is None or a.id in done: continue
            b = idx[a.partner]
            m1, m2 = decide(a, b, rng), decide(b, a, rng)
            o1, o2 = m2, m1
            if o1 in ('cooperate', 'defect') and rng.random() < perception_noise:
                o1 = 'defect' if o1 == 'cooperate' else 'cooperate'
            if o2 in ('cooperate', 'defect') and rng.random() < perception_noise:
                o2 = 'defect' if o2 == 'cooperate' else 'cooperate'
            p1, p2, ended = payoff(m1, m2, a.value, b.value)
            a.payoff += p1; b.payoff += p2
            a.last_seen[b.id] = o1; b.last_seen[a.id] = o2
            if not ended:
                moves[a.id], moves[b.id] = m2, m1          # what the partner actually played
                if a.kind == 'tft' and b.kind == 'defector': a.stuck_rounds += 1
                if b.kind == 'tft' and a.kind == 'defector': b.stuck_rounds += 1
            else:
                a.partner = b.partner = None
                a.cooldown = b.cooldown = 1                # free exit
            done.add(a.id); done.add(b.id)
        mean_val = np.mean([x.value for x in pop])
        new = {}
        for a in pop:
            if a.id in moves:
                peer = idx[a.partner].value
                kick = LIFT if moves[a.id] == 'cooperate' else -SCAR
                new[a.id] = a.value + PULL * (peer - a.value) + kick
            else:
                new[a.id] = a.value + 0.01 * (mean_val - a.value)
        for a in pop:
            a.value = float(np.clip(new[a.id], 1, 100))

    def kind(k): return [a for a in pop if a.kind == k]
    diag = {
        'evo_payoffs': [a.payoff for a in kind('evo')],
        'tft_payoffs': [a.payoff for a in kind('tft')],
        'def_payoffs': [a.payoff for a in kind('defector')],
        'tft_stuck':   [a.stuck_rounds for a in kind('tft')],
        'evo_dvalue':  [a.value - start[a.id] for a in kind('evo')],
        'value_sd':    (float(np.std(list(start.values()))), float(np.std([a.value for a in pop]))),
    }
    return kind('evo'), diag


def evolve(seed, n_gen=100, n_evo=200, n_tft=40, n_defect=80, n_rounds=80,
           elite_frac=0.15, immigrant_frac=0.05, sigma=0.10):
    mrng = np.random.default_rng(seed)
    genomes = [random_genome(mrng) for _ in range(n_evo)]
    hist = []
    for g_i in range(n_gen):
        evo, d = run_generation(genomes, n_tft, n_defect, n_rounds, seed * 100000 + g_i)
        order = np.argsort(-np.array([a.payoff for a in evo]) + mrng.normal(0, 1e-9, len(evo)))
        st = np.array(d['tft_stuck'])
        dv = np.array(d['evo_dvalue'])
        hist.append({'genome': {k: float(np.mean([a.genome[k] for a in evo])) for k in GENE_BOUNDS},
                     'evo_mean': float(np.mean(d['evo_payoffs'])),
                     'tft_mean': float(np.mean(d['tft_payoffs'])),
                     'def_mean': float(np.mean(d['def_payoffs'])),
                     'tft_stuck_pct': float(100 * np.mean(st > 0)),
                     'value_sd_start': d['value_sd'][0], 'value_sd_end': d['value_sd'][1],
                     'evo_rose20_pct': float(100 * np.mean(dv > 20)),
                     'evo_fell20_pct': float(100 * np.mean(dv < -20))})
        n_elite, n_imm = max(2, int(elite_frac * n_evo)), max(1, int(immigrant_frac * n_evo))
        elites = [dict(evo[i].genome) for i in order[:n_elite]]
        new = list(elites)
        while len(new) < n_evo - n_imm:
            new.append(mutate(elites[mrng.integers(0, n_elite)], mrng, sigma))
        while len(new) < n_evo:
            new.append(random_genome(mrng))
        genomes = new
    return hist, genomes


def trial(genome, good, q, rng, horizon=1000, partner_value=50.0):
    """Phase 4 test under Phase 6 stakes. Returns (rounds to leave or None,
    payoff per round stayed after the turn, agent's value after 100 post-turn rounds or at exit)."""
    mean, var = 0.0, genome['prior_variance']
    thr, r = genome['belief_exit_threshold'], genome['retaliate_prob']
    last, pay, n_after, v, v100 = None, 0.0, 0, 50.0, None
    for t in range(good + horizon):
        p_move = 'cooperate' if t < good or rng.random() >= q else 'defect'
        a_move = 'defect' if last == 'defect' and rng.random() < r else 'cooperate'
        p1, _, _ = payoff(a_move, p_move, v, partner_value)
        v = float(np.clip(v + PULL * (partner_value - v) + (LIFT if p_move == 'cooperate' else -SCAR), 1, 100))
        if t >= good:
            pay += p1; n_after += 1
            if n_after == 100: v100 = v
        obs = 1.0 if p_move == 'cooperate' else -1.0
        gain = var / (var + OBS_NOISE_VAR)
        mean += gain * (obs - mean); var *= (1 - gain)
        last = p_move
        if mean < thr and t >= good:
            return n_after, pay / n_after, v
    return None, pay / n_after, v100


def ci(v):
    v = np.array(v, float)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


if __name__ == '__main__':
    out = {'evolution': {}, 'intermittent': {}}
    print("evolving in the high-stakes world (free exit)...")
    finals, pops = [], []
    for s in SEEDS:
        hist, genomes = evolve(s)
        tail = hist[-10:]
        row = {k: float(np.mean([h['genome'][k] for h in tail])) for k in GENE_BOUNDS}
        row.update({k: float(np.mean([h[k] for h in tail])) for k in tail[0] if k != 'genome'})
        finals.append(row); pops.append(genomes)
        print("  seed", s, "done", flush=True)
    out['evolution'] = {k: ci([f[k] for f in finals]) for k in finals[0]}
    json.dump(pops, open(os.path.join(HERE, 'genomes_highstakes.json'), 'w'))
    print("\nevolved outcome (last 10 generations, mean ± 95% CI across seeds)")
    for k, (m, e) in out['evolution'].items():
        print(f"  {k:24s} {m:9.3f} ± {e:.3f}")

    baseline = json.load(open(os.path.join(HERE, 'genomes_free.json')))
    rng = np.random.default_rng(0)
    for label, src in [('high-stakes genomes', pops), ('baseline genomes', baseline)]:
        th = np.mean([-g['belief_exit_threshold'] for p in src for g in p])
        print(f"\nintermittent betrayal under high stakes, {label} "
              f"(mean theta {th:.3f}, predicted cutoff q = {(1 + th) / 2:.2f}), 20 good rounds first")
        print(f"{'q':>5} {'% who leave':>12} {'median rounds':>14} {'payoff/round staying':>21} {'value at exit/100 rds':>22}")
        for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            res = [trial(g, 20, q, rng) for p in src for g in p for _ in range(3)]
            fin = [x[0] for x in res if x[0] is not None]
            vals = [x[2] for x in res if x[2] is not None]
            row = {'leave_pct': 100 * len(fin) / len(res),
                   'median_lag': float(np.median(fin)) if fin else None,
                   'payoff_per_round': float(np.mean([x[1] for x in res])),
                   'value': float(np.mean(vals))}
            out['intermittent'][f"{label}|{q}"] = row
            ms = f"{row['median_lag']:.0f}" if fin else "-"
            print(f"{q:>5.1f} {row['leave_pct']:>11.1f}% {ms:>14} {row['payoff_per_round']:>21.2f} {row['value']:>22.1f}")
    json.dump(out, open(os.path.join(HERE, 'phase6.json'), 'w'), indent=2)
    print("\nsaved phase6.json")
