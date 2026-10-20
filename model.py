"""
One model for every experiment. A World config spans the original setup (v1) and the
human-like version (v2); the original is the special case neg_bias=1, doubt_floor=0, options=0.

Belief: each agent keeps, per partner, a running estimate updated toward each observation
(+1 cooperate, -neg_bias defect) with gain k = max(v / (v + 1), doubt_floor), v <- v(1 - k).
With doubt_floor=0 and neg_bias=1 this is the v1 Kalman filter, which equals the damped tally
  belief = (cooperations - defections) / (1/p + rounds seen).
Exit when belief < -tolerance + options * (outside - 1), where outside is the agent's own
market prospects (opportunity at its age x its value / 50).
"""
from dataclasses import dataclass, replace
import numpy as np

GENES = {                                  # name: (low, high)
    'threshold':   (0.0, 80.0),            # reject a partner below this value on first contact
    'prior_var':   (0.2, 5.0),             # openness: how hard early rounds move belief
    'tolerance':   (0.0, 1.0),             # theta: leave once belief < -theta
    'retaliate':   (0.0, 1.0),             # chance of defecting back after a defection
    'neg_bias':    (1.0, 4.0),             # v2: weight of a defection relative to a cooperation
    'doubt_floor': (0.0, 0.3),             # v2: belief never updates by less than this
    'options':     (0.0, 1.0),             # v2: how much better prospects raise the exit bar
}
NEUTRAL = {'neg_bias': 1.0, 'doubt_floor': 0.0, 'options': 0.0}
HUMAN = ('neg_bias', 'doubt_floor', 'options')


@dataclass(frozen=True)
class World:
    R: float = 3.0; T: float = 5.0; S: float = -1.0; P: float = 0.0
    scale_by_value: bool = False           # mutual cooperation pays R * partner value / 50
    pull: float = 0.01                     # value moves this far toward partner's each round
    lift: float = 0.0; scar: float = 0.0   # value +lift per partner cooperation, -scar per defection
    cooldown: int = 1                      # rounds out of the market after an exit
    noise: float = 0.10                    # chance each side misreads the other's move
    n_evo: int = 200; n_tft: int = 40; n_defect: int = 80
    n_mixed: int = 0                       # defect with fixed probability q ~ U(0.2, 0.8)
    n_turner: int = 0                      # cooperate, then from a random round defect with q ~ U(0.5, 1)
    rounds: int = 80
    human: bool = False                    # evolve neg_bias, doubt_floor, options


V1 = World()
V1_HIGH_STAKES = World(R=4, T=6, S=-4, P=-1, scale_by_value=True, pull=0.05, lift=0.5, scar=1.0)
V2 = replace(V1_HIGH_STAKES, n_defect=40, n_mixed=40, n_turner=40, rounds=120, human=True)


class Agent:
    def __init__(self, aid, value, kind, genome=None, q=0.0, turn=None):
        self.id, self.value, self.kind, self.genome = aid, value, kind, genome
        self.q, self.turn = q, turn
        self.age = 0; self.partner = None; self.cooldown = 0; self.payoff = 0.0
        self.last_seen = {}; self.belief = {}; self.together = 0
        self.first_bad = None                # v2 diagnostics: was the first partner a bad one?
        self.start_value = value; self.pair_start = None


def random_genome(rng, world):
    g = {'threshold': rng.uniform(0, 50), 'prior_var': rng.uniform(0.5, 4.0),
         'tolerance': rng.uniform(0.05, 0.8), 'retaliate': rng.uniform(0, 1)}
    if world.human:
        g.update({k: rng.uniform(*GENES[k]) for k in HUMAN})
    else:
        g.update(NEUTRAL)
    return g


def mutate(g, rng, world, sigma=0.10):
    out = dict(g)
    for k, (lo, hi) in GENES.items():
        if k in HUMAN and not world.human:
            continue
        out[k] = float(np.clip(g[k] + rng.normal(0, sigma * (hi - lo)), lo, hi))
    return out


def opportunity(age, peak=15, width=20, floor=0.05):
    return floor + (1 - floor) * np.exp(-0.5 * ((age - peak) / width) ** 2)


def update_belief(b, obs_coop, g):
    """b = [mean, var]; returns the new mean."""
    o = 1.0 if obs_coop else -g['neg_bias']
    k = max(b[1] / (b[1] + 1.0), g['doubt_floor'])
    b[0] += k * (o - b[0]); b[1] *= (1 - k)
    return b[0]


def exit_bar(a):
    g = a.genome
    outside = opportunity(a.age) * a.value / 50
    return -g['tolerance'] + g['options'] * (outside - 1)


def decide(a, b, rng, t):
    seen = a.last_seen.get(b.id)
    if a.kind == 'defector': return 'D'
    if a.kind == 'tft':      return 'C' if seen in (None, 'C') else 'D'
    if a.kind == 'mixed':    return 'D' if rng.random() < a.q else 'C'
    if a.kind == 'turner':   return 'D' if t >= a.turn and rng.random() < a.q else 'C'
    g = a.genome
    if seen is None and b.value < g['threshold']:
        return 'X'
    bel = a.belief.setdefault(b.id, [0.0, g['prior_var']])
    if seen is not None:
        if update_belief(bel, seen == 'C', g) < exit_bar(a):
            return 'X'
    return 'D' if seen == 'D' and rng.random() < g['retaliate'] else 'C'


def payoffs(w, m1, m2, v1, v2):
    if 'X' in (m1, m2): return 0.0, 0.0
    if m1 == m2 == 'C':
        return (w.R * v2 / 50, w.R * v1 / 50) if w.scale_by_value else (w.R, w.R)
    if m1 == 'C': return w.S, w.T
    if m2 == 'C': return w.T, w.S
    return w.P, w.P


def build(genomes, w, rng):
    kinds = (['evo'] * len(genomes) + ['tft'] * w.n_tft + ['defector'] * w.n_defect +
             ['mixed'] * w.n_mixed + ['turner'] * w.n_turner)
    loc = rng.normal(50, 15, len(kinds))
    pop = []
    for i, k in enumerate(kinds):
        v = float(np.clip(rng.normal(loc[i], 8), 1, 100))
        a = Agent(i, v, k, genomes[i] if k == 'evo' else None)
        if k == 'mixed':  a.q = rng.uniform(0.2, 0.8)
        if k == 'turner': a.q, a.turn = rng.uniform(0.5, 1.0), int(rng.integers(10, w.rounds))
        pop.append(a)
    return pop


def is_bad(b, t):
    return b.kind == 'defector' or (b.kind == 'mixed' and b.q >= 0.5) or (b.kind == 'turner' and t >= b.turn)


def run_generation(genomes, w, seed, log=False):
    rng = np.random.default_rng(seed)
    pop = build(genomes, w, rng)
    idx = {a.id: a for a in pop}
    spells = []                               # v2 diagnostics: evo agents' spells with bad partners
    for t in range(w.rounds):
        for a in pop:
            a.age += 1
            if a.cooldown: a.cooldown -= 1
        free = [a for a in pop if a.partner is None and a.cooldown == 0]
        pool = [a for a in free if rng.random() < opportunity(a.age) * 0.6]
        rng.shuffle(pool)
        pool.sort(key=lambda a: a.value + rng.normal(0, 10))
        for i in range(0, len(pool) - 1, 2):
            a, b = pool[i], pool[i + 1]
            a.partner, b.partner = b.id, a.id
            a.together = b.together = 0
            for x, y in ((a, b), (b, a)):
                if x.kind == 'evo':
                    if x.first_bad is None: x.first_bad = is_bad(y, t)
                    x.pair_start = (t, x.age, x.value, is_bad(y, t), y.kind)
        moved, done = {}, set()
        for a in pop:
            if a.partner is None or a.id in done: continue
            b = idx[a.partner]
            m1, m2 = decide(a, b, rng, t), decide(b, a, rng, t)
            o1, o2 = m2, m1
            if o1 != 'X' and rng.random() < w.noise: o1 = 'D' if o1 == 'C' else 'C'
            if o2 != 'X' and rng.random() < w.noise: o2 = 'D' if o2 == 'C' else 'C'
            p1, p2 = payoffs(w, m1, m2, a.value, b.value)
            a.payoff += p1; b.payoff += p2
            a.last_seen[b.id], b.last_seen[a.id] = o1, o2
            if 'X' in (m1, m2):
                if log:
                    for x, m in ((a, m1), (b, m2)):
                        if x.kind == 'evo' and x.pair_start and x.pair_start[3]:
                            spells.append((x.pair_start, t - x.pair_start[0], m == 'X'))
                a.partner = b.partner = None
                a.cooldown = b.cooldown = w.cooldown
            else:
                a.together += 1; b.together += 1
                moved[a.id], moved[b.id] = m2, m1
            done.update((a.id, b.id))
        mean_v = np.mean([x.value for x in pop])
        new = {}
        for a in pop:
            if a.id in moved:
                kick = w.lift if moved[a.id] == 'C' else -w.scar
                new[a.id] = a.value + w.pull * (idx[a.partner].value - a.value) + kick
            else:
                new[a.id] = a.value + 0.01 * (mean_v - a.value)
        for a in pop:
            a.value = float(np.clip(new[a.id], 1, 100))
    if log:                                   # spells still running at the end were never left
        for a in pop:
            if a.kind == 'evo' and a.partner is not None and a.pair_start and a.pair_start[3]:
                spells.append((a.pair_start, w.rounds - a.pair_start[0], False))
    return pop, spells


def evolve(w, seed, n_gen=100, elite=0.15, immigrants=0.05, log_last=0):
    rng = np.random.default_rng(seed)
    genomes = [random_genome(rng, w) for _ in range(w.n_evo)]
    hist, logs = [], []
    for gen in range(n_gen):
        log = gen >= n_gen - log_last
        pop, spells = run_generation(genomes, w, seed * 100000 + gen, log)
        evo = [a for a in pop if a.kind == 'evo']
        hist.append({'genome': {k: float(np.mean([a.genome[k] for a in evo])) for k in GENES},
                     'payoff': {k: float(np.mean([a.payoff for a in pop if a.kind == k]) / w.rounds)
                                for k in ('evo', 'tft', 'defector', 'mixed', 'turner')
                                if any(a.kind == k for a in pop)}})
        if log:
            logs.append({'spells': spells,
                         'agents': [(a.first_bad, a.start_value, a.value, a.payoff / w.rounds) for a in evo]})
        order = np.argsort([-a.payoff for a in evo])
        n_e, n_i = max(2, int(elite * w.n_evo)), max(1, int(immigrants * w.n_evo))
        elites = [evo[i].genome for i in order[:n_e]]
        genomes = list(elites)
        while len(genomes) < w.n_evo - n_i:
            genomes.append(mutate(elites[rng.integers(n_e)], rng, w))
        while len(genomes) < w.n_evo:
            genomes.append(random_genome(rng, w))
    return hist, genomes, logs


# ---------- closed forms and controlled tests ----------

def lag_v1(G, tol, p):
    """Rounds of betrayal after G good rounds before a v1 agent leaves (exact, from the tally)."""
    return int(np.floor((G * (1 + tol) + tol / p) / (1 - tol))) + 1


def lag_limit(tol, neg, floor):
    """With a doubt floor, the lag stops growing with history and tends to this constant."""
    return np.log((1 + neg) / (neg - tol)) / -np.log(1 - floor) if floor > 0 else np.inf


def cutoff(tol, neg):
    """Long-run belief against a partner who defects with probability q is 1 - q(1 + neg);
    the agent leaves only if q exceeds this."""
    return (1 + tol) / (1 + neg)


def trial(g, G, q, rng, horizon=1000):
    """Partner cooperates G rounds, then defects with probability q. Returns rounds to exit
    (None = stayed through the horizon). Perfect observation; exit bar at neutral prospects."""
    b = [0.0, g['prior_var']]
    for t in range(G + horizon):
        coop = t < G or rng.random() >= q
        if update_belief(b, coop, g) < -g['tolerance'] and t >= G:
            return t - G + 1
    return None
