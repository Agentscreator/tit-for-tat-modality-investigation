"""
Study 7 environment: a partner who may turn.

Each new partner is bad from the start with probability PI0, else good. Every round a good
partner turns bad with probability lam (no reform). The partner defects with probability
Q_GOOD if good, Q_BAD if bad. Staying pays +1 when the partner cooperates and -2 when it
defects. After each round the agent may leave; it then spends SEARCH rounds at 0 before
meeting a fresh partner. Lives last T rounds.

Agents are vectorised policies over N parallel lives: reset(mask) clears state for lives
starting a new partner; act(d) takes a bool tensor (partner defected) and returns leave mask.
"""
import math
import torch

PI0, Q_GOOD, Q_BAD = 0.2, 0.1, 0.5
R_C, R_D = 1.0, -2.0
SEARCH, T = 5, 300
LLR_D = math.log(Q_BAD / Q_GOOD)                  # evidence for "bad" from one defection   (+1.61)
LLR_C = math.log((1 - Q_BAD) / (1 - Q_GOOD))       # evidence for "bad" from one cooperation (-0.59)


def make_noise(n, seed):
    """Common random numbers: the same draws for every policy evaluated on these lives."""
    g = torch.Generator().manual_seed(seed)
    return {'turn': torch.rand(n, T, generator=g), 'move': torch.rand(n, T, generator=g),
            'start': torch.rand(n, T, generator=g), 'leave': torch.rand(n, T, generator=g)}


def run(policy, lam, noise):
    """Returns mean reward per round for each of the N lives."""
    n = noise['turn'].shape[0]
    bad = noise['start'][:, 0] < PI0
    search = torch.zeros(n, dtype=torch.long)
    total = torch.zeros(n)
    policy.reset(torch.ones(n, dtype=torch.bool))
    for t in range(T):
        new = search == 1
        if new.any():
            bad = torch.where(new, noise['start'][:, t] < PI0, bad)
            policy.reset(new)
        search = (search - 1).clamp(min=0)
        active = search == 0
        bad = bad | (active & (noise['turn'][:, t] < lam))
        d = noise['move'][:, t] < torch.where(bad, torch.tensor(Q_BAD), torch.tensor(Q_GOOD))
        total += torch.where(active, torch.where(d, R_D, R_C), torch.tensor(0.0))
        policy.u = noise['leave'][:, t]           # stochastic policies draw from shared noise
        policy.active = active                    # for probes: which lives currently have a partner
        leave = policy.act(d) & active
        search = torch.where(leave, torch.tensor(SEARCH + 1), search)
    return total / T


def scripted(policy, seqs, cap=None):
    """Feed fixed defect sequences (bool tensor N x L) to a fresh partner; return the round
    index after the scripted good rounds at which each life leaves (inf = never)."""
    n, L = seqs.shape
    policy.reset(torch.ones(n, dtype=torch.bool))
    left = torch.full((n,), float('inf'))
    for t in range(L):
        leave = policy.act(seqs[:, t])
        left = torch.where(leave & torch.isinf(left), torch.tensor(float(t)), left)
    return left


# ---------------- policies ----------------

class Bayes:
    """Exact posterior that the partner is bad, leave when it exceeds thr. Knows lam."""
    def __init__(self, lam, thr):
        self.lam, self.thr = lam, thr
    def reset(self, m):
        if not hasattr(self, 'lo') or self.lo.shape != m.shape:
            self.lo = torch.zeros(m.shape[0])
        self.lo = torch.where(m, torch.tensor(math.log(PI0 / (1 - PI0))), self.lo)
    def act(self, d):
        if self.lam > 0:                            # the partner may have turned: odds -> (odds + lam) / (1 - lam)
            self.lo = torch.logaddexp(self.lo, torch.tensor(math.log(self.lam))) - math.log1p(-self.lam)
        self.lo = self.lo + torch.where(d, LLR_D, LLR_C)
        return torch.sigmoid(self.lo) > self.thr


class Tally:
    """Original model's belief: (C - D) / (1/p + n); leave when below -theta."""
    def __init__(self, theta, prior):
        self.theta, self.prior = torch.as_tensor(theta), torch.as_tensor(prior)
    def reset(self, m):
        if not hasattr(self, 's') or self.s.shape != m.shape:
            self.s, self.n = torch.zeros(m.shape[0]), torch.zeros(m.shape[0])
        self.s = torch.where(m, 0.0, self.s); self.n = torch.where(m, 0.0, self.n)
    def act(self, d):
        self.s += torch.where(d, -1.0, 1.0); self.n += 1
        return self.s / (1 / self.prior + self.n) < -self.theta


class Human:
    """v2 belief: negativity bias beta, doubt floor f; leave when below -theta."""
    def __init__(self, theta, beta, floor, prior):
        self.theta, self.beta = torch.as_tensor(theta), torch.as_tensor(beta)
        self.floor, self.prior = torch.as_tensor(floor), torch.as_tensor(prior)
    def reset(self, m):
        if not hasattr(self, 'mu') or self.mu.shape != m.shape:
            self.mu, self.v = torch.zeros(m.shape[0]), torch.zeros(m.shape[0])
        self.mu = torch.where(m, 0.0, self.mu)
        self.v = torch.where(m, self.prior, self.v)
    def act(self, d):
        o = torch.where(d, -self.beta, torch.tensor(1.0))
        k = torch.maximum(self.v / (self.v + 1), self.floor)
        self.mu = self.mu + k * (o - self.mu); self.v = self.v * (1 - k)
        return self.mu < -self.theta
