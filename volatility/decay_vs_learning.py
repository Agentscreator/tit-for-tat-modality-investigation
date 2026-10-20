"""
Part A: can dissolution data tell decay from learning? (pre-registered as A1-A3)
Four models of perceived relationship value over 40 years, 20,000 couples each; a couple
splits the first year perceived value falls below 45.
  decay_same    every couple's value decays at the same rate (the essay's decay model)
  decay_mixed   decay rates differ between couples (lognormal)
  learn_calib   value never changes; partners learn it from noisy yearly signals, calibrated prior
  learn_optim   same, but starting beliefs are optimistic (prior mean 65 vs true mean 55)
-> decay_vs_learning.json
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N, YEARS, EXIT = 20000, 40, 45.0


def decay(rng, mixed):
    v0 = rng.normal(65, 8, N)
    r = rng.lognormal(np.log(0.02), 1.0, N) if mixed else np.full(N, 0.02)
    t = np.arange(YEARS + 1)[None, :]
    v = v0[:, None] * np.exp(-r[:, None] * t) + rng.normal(0, 3, (N, YEARS + 1))
    return v, v0


def learn(rng, prior_mean):
    mu = rng.normal(55, 12, N)                       # true match quality, fixed for life
    m, s2, noise2 = np.full(N, prior_mean), 12.0 ** 2, 20.0 ** 2
    v = [m.copy()]
    for _ in range(YEARS):
        y = mu + rng.normal(0, np.sqrt(noise2), N)
        k = s2 / (s2 + noise2)
        m = m + k * (y - m); s2 = s2 * (1 - k)
        v.append(m.copy())
    return np.array(v).T, mu


def analyse(v, quality):
    below = v[:, 1:] < EXIT
    exit_year = np.where(below.any(1), below.argmax(1) + 1, YEARS + 1)
    hazard, drift = [], []
    for t in range(1, YEARS + 1):
        at_risk = exit_year >= t
        hazard.append(float((exit_year[at_risk] == t).mean()) if at_risk.sum() else 0.0)
        drift.append((float((v[at_risk, t] - v[at_risk, t - 1]).mean()) if at_risk.sum() else 0.0,
                      int(at_risk.sum())))
    h = np.array(hazard)
    peak = int(h.argmax()) + 1
    late = float(h[29:].mean())
    surv = exit_year > 25
    return {'hazard': hazard, 'peak_year': peak, 'peak_hazard': float(h.max()), 'late_hazard_30_40': late,
            'hump': bool(1 < peak < 30 and late < 0.5 * h.max()),
            'split_pct': float(100 * (exit_year <= YEARS).mean()),
            'quality_all': float(quality.mean()),
            'quality_survivors_25y': float(quality[surv].mean()) if surv.any() else None,
            'drift': drift}


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    runs = {'decay_same': decay(rng, False), 'decay_mixed': decay(rng, True),
            'learn_calib': learn(rng, 55.0), 'learn_optim': learn(rng, 65.0)}
    out = {k: analyse(*x) for k, x in runs.items()}
    print(f"{'model':>12} {'split %':>8} {'peak yr':>8} {'peak h':>7} {'h yrs 30-40':>12} {'hump':>5} "
          f"{'quality all':>12} {'survivors 25y':>14}   mean yearly change in perceived value (yr 1, 2, 3, 5, 10, 20)")
    for k, o in out.items():
        d = o['drift']
        print(f"{k:>12} {o['split_pct']:>8.1f} {o['peak_year']:>8} {o['peak_hazard']:>7.3f} {o['late_hazard_30_40']:>12.4f} "
              f"{str(o['hump']):>5} {o['quality_all']:>12.1f} {o['quality_survivors_25y']:>14.1f}   "
              + " ".join(f"{d[t - 1][0]:+.2f}" for t in (1, 2, 3, 5, 10, 20)))
    calib_ok = all(abs(m) <= 0.5 for m, n in out['learn_calib']['drift'] if n >= 200)
    decay_neg = all(m < 0 for m, n in out['decay_mixed']['drift'] if n >= 200)
    verdict = {
        'A1': {'falsified': not out['decay_mixed']['hump']},
        'A2': {'falsified': not out['decay_mixed']['quality_survivors_25y'] > out['decay_mixed']['quality_all']},
        'A3': {'falsified': not (calib_ok and decay_neg)},
    }
    print("\nPRE-REGISTERED VERDICTS")
    for k, x in verdict.items():
        print(f"  {k}: {'FALSIFIED' if x['falsified'] else 'survived'}")
    out['verdicts'] = verdict
    json.dump(out, open(os.path.join(HERE, 'decay_vs_learning.json'), 'w'), indent=1)
