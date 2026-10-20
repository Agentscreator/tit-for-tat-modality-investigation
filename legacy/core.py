"""
Core model: evolving population of agents playing repeated PD with exit option.
Agents are defined by a genome; a fixed benchmark set (tit-for-tat, always-defect)
provides stable selection pressure and a comparison baseline.
"""
import numpy as np

OBS_NOISE_VAR = 1.0

GENE_BOUNDS = {
    'threshold':             (0.0, 80.0),   # snap-judgment floor on raw partner value, pre-history
    'prior_variance':        (0.2, 5.0),    # openness about a new partner (high = easily moved early)
    'belief_exit_threshold': (-1.0, 0.0),   # how negative belief must get before exiting
    'retaliate_prob':        (0.0, 1.0),    # chance of defecting back rather than absorbing it
}

def random_genome(rng):
    return {
        'threshold':             float(rng.uniform(0, 50)),
        'prior_variance':        float(rng.uniform(0.5, 4.0)),
        'belief_exit_threshold': float(rng.uniform(-0.8, -0.05)),
        'retaliate_prob':        float(rng.uniform(0, 1)),
    }

def mutate(genome, rng, sigma=0.10):
    out = {}
    for k, v in genome.items():
        lo, hi = GENE_BOUNDS[k]
        out[k] = float(np.clip(v + rng.normal(0, sigma * (hi - lo)), lo, hi))
    return out


class Agent:
    __slots__ = ('id','value','age','genome','kind','partner','cooldown','payoff',
                 'last_seen','rounds_with','belief','defect_ct','stuck_rounds')
    def __init__(self, aid, value, genome=None, kind='evo'):
        self.id = aid; self.value = value; self.age = 0
        self.genome = genome; self.kind = kind     # 'evo' | 'tft' | 'defector'
        self.partner = None; self.cooldown = 0; self.payoff = 0.0
        self.last_seen = {}      # partner_id -> their last move
        self.rounds_with = 0
        self.belief = {}         # partner_id -> [mean, var]
        self.defect_ct = {}
        self.stuck_rounds = 0


def opportunity_rate(age, rng, peak_age=15, width=20, floor=0.05, noise_sd=0.15):
    base = np.exp(-0.5 * ((age - peak_age) / width) ** 2)
    base = floor + (1 - floor) * base
    return max(0.0, base * (1 + rng.normal(0, noise_sd)))


def decide(agent, partner, rng):
    """Returns 'cooperate' | 'defect' | 'exit'."""
    pr = agent.last_seen.get(partner.id)

    if agent.kind == 'defector':
        return 'defect'

    if agent.kind == 'tft':
        # classic: cooperate first, then mirror. Never exits - that is the point of the benchmark.
        return 'cooperate' if pr in (None, 'cooperate') else 'defect'

    g = agent.genome
    # snap judgment only at first contact, before any behavioural evidence exists
    if pr is None and partner.value < g['threshold']:
        return 'exit'

    b = agent.belief.get(partner.id)
    if b is None:
        b = [0.0, g['prior_variance']]
        agent.belief[partner.id] = b

    if pr is not None:
        o = 1.0 if pr == 'cooperate' else -1.0
        gain = b[1] / (b[1] + OBS_NOISE_VAR)      # Kalman gain: shrinks as certainty grows
        b[0] += gain * (o - b[0])
        b[1] *= (1 - gain)
        if b[0] < g['belief_exit_threshold']:
            return 'exit'

    if pr == 'defect' and rng.random() < g['retaliate_prob']:
        return 'defect'
    return 'cooperate'


def payoffs(m1, m2):
    if m1 == 'exit' or m2 == 'exit':              return 0.0, 0.0, True
    if m1 == 'cooperate' and m2 == 'cooperate':   return 3.0, 3.0, False
    if m1 == 'cooperate' and m2 == 'defect':      return -1.0, 5.0, False
    if m1 == 'defect'    and m2 == 'cooperate':   return 5.0, -1.0, False
    return 0.0, 0.0, False


def match(pop, rng, match_rate=0.6):
    free = [a for a in pop if a.partner is None and a.cooldown == 0]
    pool = [a for a in free if rng.random() < min(opportunity_rate(a.age, rng), 1.0) * match_rate]
    rng.shuffle(pool)
    pool.sort(key=lambda a: a.value + rng.normal(0, 10))   # assortative, with noise
    for i in range(0, len(pool) - 1, 2):
        a, b = pool[i], pool[i+1]
        a.partner, b.partner = b.id, a.id
        a.rounds_with = b.rounds_with = 0
