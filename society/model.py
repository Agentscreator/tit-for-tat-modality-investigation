"""
A small evolving society. Agents pair up, climb an investment ladder, raise children and die.
What they want in a partner is heritable and free to take any shape; evolution decides.

Traits (heritable, plus developmental noise):
  W warmth      you deliver sig(W) of what your rung says you'll invest
  K competence  production x exp(0.3 K)
  B boldness    production x exp(0.15 B), but +0.004 exp(0.8 B) yearly mortality
Each pair also has a fit F ~ N(0, 1): their pot grows x exp(FIT F) and they conceive at 0.5 sig(1 + 2F).
Preference for partner trait x (W, K, B, and the pair's F): weight * exp(-(x - ideal)^2 / (2 width^2)), summed and
normalised to [0, 1]. Gate zeta: competence counts (1 - zeta) + zeta * sig(2 W_hat), so zeta = 0
judges traits independently and zeta = 1 counts a cold partner's competence for nothing.

Ladder: goodwill 0, willingness 1, commitment 2, dedication 3, devotion 4. Each partner holds their
own rung and puts INVEST[rung] x sig(W) of production into the pair pot. The pot grows by SYNERGY,
feeds the children of the household, and the rest is split evenly. Children need both partners at
commitment or above. Each year an agent drops a rung if partner value < drop, else climbs if > climb, and
leaves when it drops below goodwill. Warmth and fit are never seen: belief about each is a Kalman
estimate from one noisy reading per year. Nothing about anyone, or any pair, changes over time.
Land is shared in proportion to productivity, so the population can't outgrow it.
"""
from dataclasses import dataclass
import numpy as np

RUNGS = ['goodwill', 'willingness', 'commitment', 'dedication', 'devotion']
INVEST = np.array([0.05, 0.2, 0.4, 0.6, 0.8])
SYNERGY, NEED_A, NEED_C, FIT = 1.5, 0.6, 0.8, 0.6
CUE_SD = np.array([1.0, 0.3, 0.3, 1.0])     # first-impression noise on W, K, B, F

SPEC = ([(t, -4, 4) for t in 'WKB'] +                       # traits
        [(f'ideal_{t}', -4, 4) for t in 'WKBF'] +
        [(f'width_{t}', np.log(0.3), np.log(5)) for t in 'WKBF'] +  # log width
        [(f'weight_{t}', 0, 1) for t in 'WKBF'] +
        [('zeta', 0, 1), ('climb', 0, 1), ('drop', 0, 1)])
G = {n: i for i, (n, _, _) in enumerate(SPEC)}
LO, HI = np.array([s[1] for s in SPEC]), np.array([s[2] for s in SPEC])
NG, CAP = len(SPEC), 8000


@dataclass(frozen=True)
class World:
    warm_matters: bool = True     # False: everyone delivers 0.6 of their rung, whatever W
    bold_costs: bool = True       # False: boldness has no mortality cost
    years: int = 4000
    record_from: int = 3000
    cap: float = 500              # adults the land supports at full productivity
    meet: int = 5                 # encounters per single per year
    seed: int = 0


def sig(x):
    return 1 / (1 + np.exp(-x))


def score(g, x):
    """g: (n, NG) genomes; x: (n, 4) perceived W, K, B, F. Returns partner value in [0, 1]."""
    mu, sd, wt = g[:, 3:7], np.exp(g[:, 7:11]), g[:, 11:15]
    u = wt * np.exp(-0.5 * ((x - mu) / sd) ** 2)
    z = g[:, G['zeta']]
    u[:, 1] *= (1 - z) + z * sig(2 * x[:, 0])
    return u.sum(1) / (wt.sum(1) + 1e-9)


class Society:
    def __init__(self, w: World):
        self.w, self.rng = w, np.random.default_rng(w.seed)
        rng = self.rng
        self.alive = np.zeros(CAP, bool); self.age = np.zeros(CAP, int)
        self.g = np.zeros((CAP, NG)); self.ph = np.zeros((CAP, 3))
        self.partner = np.full(CAP, -1); self.rung = np.zeros(CAP, int)
        self.bm = np.zeros((CAP, 2)); self.bv = np.ones((CAP, 2))  # belief about partner's W, pair F
        self.fit = np.zeros(CAP)                                   # true fit of current pair
        self.custodian = np.full(CAP, -1); self.last_birth = np.full(CAP, -99)
        self.wealth = np.zeros(CAP); self.pair = np.full(CAP, -1)
        self.year = 0; self.pairs = []; self.series = []

        n = 1000
        g = np.column_stack([rng.normal(0, .5, (n, 3)), rng.uniform(-2, 2, (n, 4)),
                             np.log(rng.uniform(.5, 3, (n, 4))), rng.uniform(0, 1, (n, 4)),
                             rng.uniform(0, 1, n), rng.uniform(.3, .7, n), rng.uniform(.1, .4, n)])
        self._born(np.arange(n), g, rng.integers(0, 50, n))

    def _born(self, ids, g, age):
        self.alive[ids] = True; self.age[ids] = age; self.g[ids] = g
        self.ph[ids] = g[:, :3] + self.rng.normal(0, .5, (len(ids), 3))
        self.partner[ids] = -1; self.rung[ids] = 0; self.wealth[ids] = 0
        self.last_birth[ids] = -99; self.pair[ids] = -1

    # ---------------------------------------------------------------- pairs
    def _unpair(self, a, reason):
        b = self.partner[a]
        rec = self.pair[a]
        if rec >= 0 and self.pairs[rec]['end'] is None:
            self.pairs[rec]['end'] = self.year; self.pairs[rec]['reason'] = reason
        self.partner[a] = self.partner[b] = -1
        self.rung[a] = self.rung[b] = 0
        self.pair[a] = self.pair[b] = -1

    def _pair(self, a, b, cue_a, cue_b):
        self.partner[a], self.partner[b] = b, a
        self.rung[a] = self.rung[b] = 0
        self.bm[a], self.bm[b] = cue_a[[0, 3]], cue_b[[0, 3]]   # first impressions of W and F
        self.bv[a] = self.bv[b] = CUE_SD[[0, 3]] ** 2
        self.fit[a] = self.fit[b] = self._pending_fit
        if self.year >= self.w.record_from:
            self.pairs.append({'start': self.year, 'end': None, 'reason': None, 'a': int(a), 'b': int(b),
                               'ph_a': self.ph[a].tolist(), 'ph_b': self.ph[b].tolist(),
                               'age_a': int(self.age[a]), 'age_b': int(self.age[b]),
                               'fit': float(self.fit[a]),
                               'kids': 0, 'raised': 0, 'top': 0})
            self.pair[a] = self.pair[b] = len(self.pairs) - 1

    def _market(self, adults):
        rng = self.rng
        single = adults[(self.partner[adults] < 0) & (self.age[adults] <= 60)]
        if len(single) < 2: return
        a = np.repeat(single, self.w.meet)
        b = rng.choice(single, len(a))
        keep = a != b; a, b = a[keep], b[keep]
        F = rng.normal(0, 1, len(a))                                   # fit of each encounter
        cue_ab = np.column_stack([self.ph[b], F]) + rng.normal(0, 1, (len(a), 4)) * CUE_SD
        cue_ba = np.column_stack([self.ph[a], F]) + rng.normal(0, 1, (len(a), 4)) * CUE_SD
        sa, sb = score(self.g[a], cue_ab), score(self.g[b], cue_ba)
        ok = (sa > self.g[a, G['climb']]) & (sb > self.g[b, G['climb']])
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(-(sa[idx] + sb[idx]))]
        used = set()
        for i in idx:
            x, y = a[i], b[i]
            if x in used or y in used: continue
            used.update((x, y))
            self._pending_fit = F[i]
            self._pair(x, y, cue_ab[i], cue_ba[i])

    # ---------------------------------------------------------------- year
    def step(self):
        w, rng = self.w, self.rng
        al = np.flatnonzero(self.alive)
        adults, kids = al[self.age[al] >= 16], al[self.age[al] < 16]
        W, K, B = self.ph[adults].T
        hz = np.zeros(CAP)

        # production (land is shared: more adults, less each)
        sd = 0.2 + 0.2 * sig(B)
        e = np.exp(0.3 * K + 0.15 * B + rng.normal(0, 1, len(adults)) * sd - sd ** 2 / 2)
        prod = np.zeros(CAP)
        prod[adults] = e * w.cap / e.sum()

        # investment into the pot: rung x delivery
        deliver = np.zeros(CAP)
        deliver[adults] = sig(W) if w.warm_matters else 0.6
        paired = adults[self.partner[adults] >= 0]
        inv = np.zeros(CAP)
        inv[paired] = INVEST[self.rung[paired]] * deliver[paired] * prod[paired]
        pot = np.zeros(CAP)                      # pot indexed by each household member
        pot[paired] = SYNERGY * np.exp(FIT * self.fit[paired]) * (inv[paired] + inv[self.partner[paired]])
        # single custodians put commitment-level effort into their own kids, no synergy
        single = adults[self.partner[adults] < 0]
        kid_cust = self.custodian[kids]
        has_kids = np.zeros(CAP, bool); has_kids[kid_cust[kid_cust >= 0]] = True
        sc = single[has_kids[single]]
        inv[sc] = INVEST[2] * deliver[sc] * prod[sc]; pot[sc] = inv[sc]

        # children eat from their custodian's household first
        hid = np.arange(CAP)                     # household id: lower id of the pair, or self
        hid[paired] = np.minimum(paired, self.partner[paired])
        hh = np.where(kid_cust >= 0, hid[np.maximum(kid_cust, 0)], -1)
        nk = np.bincount(hh[hh >= 0], minlength=CAP)
        hpot = np.zeros(CAP)
        hp = hid[paired]
        hpot[hp] = pot[paired]
        hpot[sc] = pot[sc]
        fed = np.minimum(hpot, nk * NEED_C)
        food = np.zeros(len(kids))
        good = hh >= 0
        food[good] = fed[hh[good]] / np.maximum(nk[hh[good]], 1)
        food[~good] = 0.3 * NEED_C                              # orphans: kin pick up a little
        left = hpot - fed

        # adults: keep what they didn't invest, plus their share of the leftover pot
        inc = np.zeros(CAP)
        inc[adults] = prod[adults] - inv[adults]
        inc[paired] += 0.5 * left[hp]
        inc[sc] += left[sc]
        self.wealth[adults] = 0.8 * self.wealth[adults] + inc[adults] - NEED_A
        short = self.wealth[adults] < 0
        hz[adults[short]] += 0.3 * np.minimum(1, -self.wealth[adults[short]] / NEED_A)
        self.wealth[adults[short]] = 0

        # mortality
        hz[kids] += 0.01 + 0.3 * np.exp(-3 * food / NEED_C) + 0.04 * (self.age[kids] == 0)
        hz[adults] += 0.003 + 0.0006 * np.exp(0.09 * (self.age[adults] - 16))
        hz[adults] += 0.004 * (np.exp(0.8 * B) if w.bold_costs else 1)

        # learning and the ladder (all paired agents, simultaneously)
        p = paired; q = self.partner[p]
        obs = np.column_stack([self.ph[q, 0], self.fit[p]]) + rng.normal(0, 1, (len(p), 2))
        k = self.bv[p] / (self.bv[p] + 1)
        self.bm[p] += k * (obs - self.bm[p]); self.bv[p] *= 1 - k
        seen = np.column_stack([self.bm[p, 0], self.ph[q, 1:] + rng.normal(0, 1, (len(p), 2)) * CUE_SD[1:3],
                                self.bm[p, 1]])
        v = score(self.g[p], seen)
        down = v < self.g[p, G['drop']]
        up = ~down & (v > self.g[p, G['climb']])
        self.rung[p] = np.minimum(self.rung[p] + up, 4) - down
        for a in p:
            r = self.pair[a]
            if r >= 0: self.pairs[r]['top'] = max(self.pairs[r]['top'], int(max(self.rung[a], 0)))
        for a in p[self.rung[p] < 0]:
            if self.partner[a] >= 0: self._unpair(a, 'exit')
        self.rung[self.rung < 0] = 0

        # births: both partners at commitment+, both 16-45, one birth per two years
        cand = adults[(self.partner[adults] > adults)]
        if len(cand):
            b = self.partner[cand]
            ok = ((self.rung[cand] >= 2) & (self.rung[b] >= 2) & (self.age[cand] <= 45) & (self.age[b] <= 45)
                  & (self.year - self.last_birth[cand] >= 2) & (rng.random(len(cand)) < 0.5 * sig(1 + 2 * self.fit[cand])))
            m1, m2 = cand[ok], b[ok]
            free = np.flatnonzero(~self.alive)[:len(m1)]
            m1, m2 = m1[:len(free)], m2[:len(free)]
            mask = rng.random((len(free), NG)) < 0.5
            g = np.where(mask, self.g[m1], self.g[m2]) + rng.normal(0, 0.015, (len(free), NG)) * (HI - LO)
            g[:, 1] -= 0.05                                     # mutation load on competence
            g = np.clip(g, LO, HI)
            self._born(free, g, np.zeros(len(free), int))
            cust = np.where(rng.random(len(free)) < 0.5, m1, m2)
            self.custodian[free] = cust
            self.last_birth[m1] = self.last_birth[m2] = self.year
            for c, x in zip(free, m1):
                r = self.pair[x]
                self.pair[c] = r                                 # birth household, for credit
                if r >= 0: self.pairs[r]['kids'] += 1

        # deaths
        hz[al[self.age[al] > 95]] = 10
        dead = al[rng.random(len(al)) < 1 - np.exp(-hz[al])]
        for a in dead:
            if self.partner[a] >= 0: self._unpair(a, 'death')
        self.alive[dead] = False
        self.partner[dead] = -1
        self.custodian[np.isin(self.custodian, dead)] = -1

        # aging; children who reach adulthood credit their birth pair
        self.age[al] += 1
        grown = al[self.alive[al] & (self.age[al] == 16)]
        for c in grown:
            r = self.pair[c]
            if r >= 0: self.pairs[r]['raised'] += 1
            self.pair[c] = -1; self.custodian[c] = -1

        adults = np.flatnonzero(self.alive & (self.age >= 16))
        self._market(adults)
        if self.year % 50 == 0:
            self.series.append(self.snapshot())
        self.year += 1

    def snapshot(self):
        a = np.flatnonzero(self.alive & (self.age >= 16))
        return {'year': self.year, 'N': int(self.alive.sum()), 'adults': len(a),
                'paired': float((self.partner[a] >= 0).mean()),
                'genes': dict(zip([s[0] for s in SPEC], self.g[a].mean(0).round(3).tolist())),
                'traits': self.ph[a].mean(0).round(3).tolist()}

    def run(self):
        for _ in range(self.w.years):
            self.step()
            if not self.alive.any(): break
        a = np.flatnonzero(self.alive & (self.age >= 16))
        for r in self.pairs:
            if r['end'] is None: r['end'], r['reason'] = self.year, 'ongoing'
        return {'series': self.series, 'pairs': self.pairs,
                'final_genes': self.g[a].tolist(), 'final_traits': self.ph[a].tolist()}
