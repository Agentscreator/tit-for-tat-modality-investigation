"""
Phase 4: intermittent betrayal.
Partner cooperates `good` rounds, then defects with probability q each round (q=1 is the
Phase 3 clean switch). Does a partner who is bad only some of the time escape detection,
and if so, is the agent actually wrong to stay?
"""
import os, json
import numpy as np
from engine import evolve
from core import OBS_NOISE_VAR

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 2, 3, 4, 5]
HORIZON = 1000          # rounds AFTER the turn (Phase 3 capped total rounds, which censored large G)
REPS = 5                # random betrayal schedules per genome
GOOD = [0, 20, 80]
QS = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PAY = {('C','C'): 3.0, ('C','D'): -1.0, ('D','C'): 5.0, ('D','D'): 0.0}


def load_genomes():
    """Evolve under free exit (same settings as Phase 3); cache so later phases reuse them."""
    path = os.path.join(HERE, 'genomes_free.json')
    if os.path.exists(path):
        return json.load(open(path))
    pops = []
    for s in SEEDS:
        _, genomes = evolve(n_gen=100, n_evo=200, n_tft=40, n_defect=80, n_rounds=80,
                            seed=s, exit_cost_mode='free', perception_noise=0.10)
        pops.append(genomes)
        print("  seed", s, "evolved")
    json.dump(pops, open(path, 'w'))
    return pops


def trial(genome, good, q, rng):
    """Returns (rounds after the turn until exit or None, agent's mean payoff per post-turn round stayed)."""
    mean, var = 0.0, genome['prior_variance']
    thr, r = genome['belief_exit_threshold'], genome['retaliate_prob']
    last, pay, n_after = None, 0.0, 0
    for t in range(good + HORIZON):
        p_move = 'C' if t < good or rng.random() >= q else 'D'
        a_move = 'D' if last == 'D' and rng.random() < r else 'C'
        if t >= good:
            pay += PAY[(a_move, p_move)]; n_after += 1
        obs = 1.0 if p_move == 'C' else -1.0
        gain = var / (var + OBS_NOISE_VAR)
        mean += gain * (obs - mean)
        var *= (1 - gain)
        last = p_move
        if mean < thr and t >= good:
            return n_after, pay / n_after
    return None, pay / n_after


if __name__ == '__main__':
    print("loading evolved genomes (free exit)...")
    pops = load_genomes()
    rng = np.random.default_rng(0)
    thetas = [-g['belief_exit_threshold'] for p in pops for g in p]
    print(f"  {sum(map(len, pops))} genomes, mean exit threshold -{np.mean(thetas):.3f}")

    table = {}
    for good in GOOD:
        print(f"\n{good} good rounds, then partner defects with probability q")
        print(f"{'q':>5} {'% who leave':>12} {'median rounds to leave':>23} {'payoff/round while staying':>27}")
        for q in QS:
            left, lags, pays = [], [], []
            for genomes in pops:
                res = [trial(g, good, q, rng) for g in genomes for _ in range(REPS)]
                fin = [lag for lag, _ in res if lag is not None]
                left.append(100 * len(fin) / len(res))
                lags.append(np.median(fin) if fin else np.nan)
                pays.append(np.mean([p for _, p in res]))
            med = float(np.nanmean(lags)) if not np.all(np.isnan(lags)) else None
            row = {'leave_pct': float(np.mean(left)), 'median_lag': med,
                   'payoff_per_round': float(np.mean(pays))}
            table[f"{good}|{q}"] = row
            ms = f"{med:.1f}" if med is not None else "-"
            print(f"{q:>5.1f} {row['leave_pct']:>11.1f}% {ms:>23} {row['payoff_per_round']:>27.2f}")

    json.dump({'horizon': HORIZON, 'reps': REPS, 'mean_theta': float(np.mean(thetas)), 'table': table},
              open(os.path.join(HERE, 'phase4.json'), 'w'), indent=2)
    print("\nsaved phase4.json")
