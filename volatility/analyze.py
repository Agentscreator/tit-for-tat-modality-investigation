"""
Study 7: trains the GRU per volatility level, tunes the baselines, measures grace curves,
probes the GRU's hidden state, and scores each pre-registered hypothesis -> results.json
Usage: python analyze.py   (~5 min on 8 cores)
"""
import json, os, itertools
from multiprocessing import Pool
import torch
import env as E
import gru as N
from predict import LAMS, GS, best_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [1, 2, 3]
EVAL = E.make_noise(4000, 12345)


def train_job(args):
    torch.set_num_threads(1)
    lam, seed = args
    theta, curve = N.train(lam, seed)
    return lam, seed, theta, curve


def grid_eval(make, combos, lam, lives=600, seed=7):
    """Evaluate every parameter combo on the same lives; return the best combo."""
    noise = E.make_noise(lives, seed)
    noise = {k: v.repeat(len(combos), 1) for k, v in noise.items()}
    cols = [torch.tensor([c[i] for c in combos], dtype=torch.float).repeat_interleave(lives)
            for i in range(len(combos[0]))]
    r = E.run(make(*cols), lam, noise).view(len(combos), lives).mean(1)
    return combos[int(r.argmax())]


def tune_job(lam):
    torch.set_num_threads(1)
    thetas = [round(-0.95 + 0.05 * i, 2) for i in range(40)]
    tally = grid_eval(E.Tally, list(itertools.product(thetas, [0.5, 1, 2, 4])), lam)
    human = grid_eval(E.Human, list(itertools.product(thetas[::2], [1, 1.5, 2, 2.75, 3.5, 4.5],
                                                      [0, 0.02, 0.05, 0.1, 0.2, 0.3], [1, 3])), lam, lives=250)
    thr, _ = best_threshold(lam)
    return lam, {'tally': tally, 'human': human, 'bayes_thr': thr}


def agents(lam, tuned, thetas, n):
    t, h = tuned['tally'], tuned['human']
    out = {'bayes': E.Bayes(lam, tuned['bayes_thr']), 'tally': E.Tally(*t), 'human': E.Human(*h)}
    for s, th in thetas.items():
        out[f'gru{s}'] = N.GRU(th.unsqueeze(0), n)
    return out


def grace(policy, G, q, n=400, L=600, seed=2):
    g = torch.Generator().manual_seed(seed)
    seqs = torch.zeros(n, G + L, dtype=torch.bool)
    seqs[:, G:] = torch.rand(n, L, generator=g) < q
    left = E.scripted(policy, seqs) - G + 1
    early = float(100 * (left < 1).float().mean())
    left = left.clamp(min=0)                           # leaving during the good run counts as 0
    return float(left.median()), early


class Probe:
    """Runs a GRU in the environment while shadowing Bayes, tally and human beliefs."""
    def __init__(self, g, bayes, tally, human):
        self.g, self.others = g, (bayes, tally, human)
        self.rows = []
    def reset(self, m):
        self.g.reset(m)
        for o in self.others: o.reset(m)
    def act(self, d):
        leave = self.g.act(d)
        b, t, h = self.others
        for o in self.others: o.act(d)
        hidden, logit = self.g.trace[-1]
        tally_belief = t.s / (1 / t.prior + t.n)
        self.rows.append((self.active.clone(), hidden, logit, b.lo.clone(), tally_belief, h.mu.clone()))
        self.g.trace.clear()
        return leave


def r2(X, y, train):
    X = torch.cat([X, torch.ones(len(X), 1)], 1)
    w = torch.linalg.lstsq(X[train], y[train].unsqueeze(1)).solution
    pred = (X[~train] @ w).squeeze(1)
    yt = y[~train]
    return float(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())


def probe(lam, tuned, theta, n=800):
    noise = E.make_noise(n, 777)
    p = Probe(N.GRU(theta.unsqueeze(0), n, record=True), E.Bayes(lam, tuned['bayes_thr']),
              E.Tally(*tuned['tally']), E.Human(*tuned['human']))
    E.run(p, lam, noise)
    life = torch.arange(n)
    act = torch.cat([r[0] for r in p.rows])
    H = torch.cat([r[1] for r in p.rows])[act]
    lifeid = life.repeat(len(p.rows))[act]
    train = lifeid < n // 2                            # held out by life, not by round
    targets = {'bayes_logodds': torch.cat([r[3] for r in p.rows])[act],
               'tally_belief': torch.cat([r[4] for r in p.rows])[act],
               'human_belief': torch.cat([r[5] for r in p.rows])[act]}
    out = {k: r2(H, y.clamp(-30, 30), train) for k, y in targets.items()}
    logit = torch.cat([r[2] for r in p.rows])[act]
    out['corr_leave_logit_bayes'] = float(torch.corrcoef(torch.stack([logit, targets['bayes_logodds'].clamp(-30, 30)]))[0, 1])
    return out


if __name__ == '__main__':
    with Pool(8) as pool:
        trained = pool.map(train_job, [(l, s) for l in LAMS for s in SEEDS])
        tuned = dict(pool.map(tune_job, LAMS))
    torch.set_num_threads(8)
    res = {}
    for lam in LAMS:
        thetas = {s: th for l, s, th, _ in trained if l == lam}
        ag = agents(lam, tuned[lam], thetas, 4000)
        reward = {k: float(E.run(a, lam, EVAL).mean()) for k, a in ag.items()}
        gr = {}
        for k in ag:
            one = agents(lam, tuned[lam], thetas, 400)[k]
            gr[k] = {q: {G: grace(one, G, q) for G in GS} for q in (0.5, 1.0)}
        pr = {s: probe(lam, tuned[lam], th) for s, th in thetas.items()}
        res[lam] = {'tuned': tuned[lam], 'reward': reward, 'grace': gr, 'probe': pr,
                    'curves': {s: c for l, s, _, c in trained if l == lam}}
        gm = lambda G, q=0.5: sum(gr[f'gru{s}'][q][G][0] for s in SEEDS) / len(SEEDS)
        print(f"\nlam={lam}  tuned tally {tuned[lam]['tally']}  human {tuned[lam]['human']}  bayes thr {tuned[lam]['bayes_thr']:.2f}")
        print("  reward/round: " + "  ".join(f"{k} {v:.3f}" for k, v in reward.items()))
        for k in ('bayes', 'tally', 'human', 'gru1'):
            print(f"  grace q=0.5 {k:>6}: " + " ".join(f"G{G}:{gr[k][0.5][G][0]:.0f}" for G in GS))
        print(f"  GRU mean grace q=0.5: " + " ".join(f"G{G}:{gm(G):.1f}" for G in GS))
        print("  probe R2: " + "  ".join(f"seed{s}: bayes {v['bayes_logodds']:.2f} tally {v['tally_belief']:.2f} human {v['human_belief']:.2f}" for s, v in pr.items()))

    # ---------------- pre-registered verdicts ----------------
    def gmean(lam, G, q=0.5):
        return sum(res[lam]['grace'][f'gru{s}'][q][G][0] for s in SEEDS) / len(SEEDS)
    def gru_eff(lam):
        return sum(res[lam]['reward'][f'gru{s}'] for s in SEEDS) / len(SEEDS) / res[lam]['reward']['bayes']
    v = {}
    vol = [l for l in LAMS if l >= 0.003]
    v['H1'] = {'detail': {l: (gmean(l, 160), gmean(l, 20)) for l in vol},
               'falsified': any(gmean(l, 160) > 1.5 * gmean(l, 20) for l in vol)}
    seq = [gmean(l, 160) for l in vol]
    rises = sum(b > a for a, b in zip(seq, seq[1:]))
    v['H2'] = {'detail': dict(zip(vol, seq)), 'falsified': seq[0] < 2 * seq[-1] or rises > 1}
    h3 = {l: [(res[l]['probe'][s]['bayes_logodds'], res[l]['probe'][s]['tally_belief']) for s in SEEDS]
          for l in LAMS if l >= 0.01}
    v['H3'] = {'detail': h3, 'falsified': any(b < 0.8 or b < t + 0.1 for l in h3 for b, t in h3[l])}
    eff = {l: res[l]['reward']['tally'] / res[l]['reward']['bayes'] for l in LAMS}
    v['H4'] = {'detail': eff, 'falsified': any(eff[l] >= 0.9 for l in LAMS if l >= 0.03) or eff[0.0] < 0.95}
    ge = {l: gru_eff(l) for l in LAMS}
    v['H5'] = {'detail': ge, 'falsified': any(e < 0.9 for e in ge.values())}
    v['H6'] = {'detail': (gmean(0.0, 160), gmean(0.0, 20)), 'falsified': gmean(0.0, 160) >= 2 * gmean(0.0, 20)}
    print("\nPRE-REGISTERED VERDICTS")
    for k, x in v.items():
        print(f"  {k}: {'FALSIFIED' if x['falsified'] else 'survived'}   {x['detail']}")
    res['verdicts'] = v
    json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=str)
    torch.save({f"{l}|{s}": th for l, s, th, _ in trained}, os.path.join(HERE, 'gru_params.pt'))
    print("\nsaved results.json, gru_params.pt")
