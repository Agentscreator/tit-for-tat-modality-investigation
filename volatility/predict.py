"""Bayes-optimal predictions, computed before any learned agent is trained -> predictions.json"""
import json, math, os
import torch
import env as E

HERE = os.path.dirname(os.path.abspath(__file__))
LAMS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
GS = [0, 5, 10, 20, 40, 80, 160]
THRS = torch.linspace(0.05, 0.95, 37)


def best_threshold(lam, n=4000, seed=1):
    noise = E.make_noise(n * len(THRS), seed)
    thr = THRS.repeat_interleave(n)
    r = E.run(E.Bayes(lam, thr), lam, noise).view(len(THRS), n).mean(1)
    i = int(r.argmax())
    return float(THRS[i]), float(r[i])


def grace(policy, G, q, n=400, L=600, seed=2):
    """Rounds after G good rounds until leaving, partner defecting with prob q afterwards."""
    g = torch.Generator().manual_seed(seed)
    seqs = torch.zeros(n, G + L, dtype=torch.bool)
    seqs[:, G:] = torch.rand(n, L, generator=g) < q
    left = E.scripted(policy, seqs) - G + 1
    left = torch.where(left < 1, torch.tensor(float('inf')), left)   # left before the turn
    return float(left.median()), float(100 * torch.isinf(left).float().mean())


def saturation(lam):
    """History stops counting once the posterior reaches its floor: odds* = r*lam/(1-r),
    r = exp(LLR_C). Rounds of good history needed to get there from the prior."""
    if lam == 0:
        return math.inf
    r = math.exp(E.LLR_C)
    floor = math.log(r * lam / (1 - r))
    return (floor - math.log(E.PI0 / (1 - E.PI0))) / E.LLR_C


if __name__ == '__main__':
    out = {}
    print(f"{'lam':>6} {'thr':>5} {'reward':>7} {'G*':>6}  grace after G good rounds, partner turns to q=0.5 (median rounds)")
    for lam in LAMS:
        thr, rew = best_threshold(lam)
        pol = E.Bayes(lam, thr)
        curve_half = {G: grace(pol, G, E.Q_BAD) for G in GS}
        curve_all = {G: grace(pol, G, 1.0) for G in GS}
        out[lam] = {'thr': thr, 'reward': rew, 'saturation_G': saturation(lam),
                    'grace_q05': curve_half, 'grace_q1': curve_all}
        print(f"{lam:>6} {thr:>5.2f} {rew:>7.3f} {saturation(lam):>6.1f}  " +
              "  ".join(f"G={G}:{v[0]:.0f}" for G, v in curve_half.items()))
    print("\npure betrayal (q=1): " )
    for lam in LAMS:
        print(f"{lam:>6}  " + "  ".join(f"G={G}:{v[0]:.0f}" for G, v in out[lam]['grace_q1'].items()))
    json.dump(out, open(os.path.join(HERE, 'predictions.json'), 'w'), indent=1)
