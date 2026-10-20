"""A tiny GRU policy, batched over a population of parameter vectors, trained by evolution strategies."""
import torch
import env as E

H = 8
SHAPES = {'w_x': (3 * H,), 'w_h': (3 * H, H), 'b': (3 * H,), 'b_hn': (H,), 'w_out': (H,), 'b_out': (1,)}
N_PARAMS = sum(torch.Size(s).numel() for s in SHAPES.values())


def unpack(theta):
    """theta: (P, N_PARAMS) -> dict of (P, *shape)."""
    out, i = {}, 0
    for k, s in SHAPES.items():
        n = torch.Size(s).numel()
        out[k] = theta[:, i:i + n].view(theta.shape[0], *s); i += n
    return out


class GRU:
    """Policy over P parameter vectors x B lives each (lives are laid out member-major)."""
    def __init__(self, theta, lives_per_member, record=False, stochastic=False):
        self.p = unpack(theta); self.P = theta.shape[0]; self.B = lives_per_member
        self.stochastic, self.u = stochastic, None
        self.h = torch.zeros(self.P, self.B, H)
        self.record = record; self.trace = []
    def reset(self, m):
        self.h = torch.where(m.view(self.P, self.B, 1), 0.0, self.h)
    def act(self, d):
        x = torch.where(d, 1.0, -1.0).view(self.P, self.B, 1)
        p = self.p
        gx = x * p['w_x'].unsqueeze(1) + p['b'].unsqueeze(1)                 # (P, B, 3H)
        gh = torch.bmm(self.h, p['w_h'].transpose(1, 2))                      # (P, B, 3H)
        r = torch.sigmoid(gx[..., :H] + gh[..., :H])
        z = torch.sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H])
        n = torch.tanh(gx[..., 2 * H:] + r * (gh[..., 2 * H:] + p['b_hn'].unsqueeze(1)))
        self.h = (1 - z) * n + z * self.h
        logit = (self.h * p['w_out'].unsqueeze(1)).sum(-1) + p['b_out']
        if self.record:
            self.trace.append((self.h.reshape(-1, H).clone(), logit.reshape(-1).clone()))
        if self.stochastic and self.u is not None:      # training: leave with probability sigmoid(logit)
            return (self.u < torch.sigmoid(logit.reshape(-1)))
        return (logit > 0).reshape(-1)


def train(lam, seed, gens=400, pairs=48, lives=48, sigma=0.1, lr=0.03, log_every=50):
    torch.manual_seed(seed)
    theta = torch.randn(N_PARAMS) * 0.3
    theta[-1] = -2.0                                   # start out mostly staying
    m, v = torch.zeros_like(theta), torch.zeros_like(theta)
    curve = []
    for g in range(gens):
        eps = torch.randn(pairs, N_PARAMS)
        pop = torch.cat([theta + sigma * eps, theta - sigma * eps])
        base = E.make_noise(lives, seed * 10_000 + g)
        noise = {k: v_.repeat(2 * pairs, 1) for k, v_ in base.items()}   # common random numbers
        fit = E.run(GRU(pop, lives, stochastic=True), lam, noise).view(2 * pairs, lives).mean(1)
        ranks = fit.argsort().argsort().float() / (2 * pairs - 1) - 0.5
        grad = ((ranks[:pairs] - ranks[pairs:]).unsqueeze(1) * eps).sum(0) / (pairs * sigma) - 0.005 * theta
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2       # Adam ascent
        theta = theta + lr * (m / (1 - 0.9 ** (g + 1))) / ((v / (1 - 0.999 ** (g + 1))).sqrt() + 1e-8)
        if g % log_every == 0 or g == gens - 1:
            curve.append((g, float(fit.mean())))
    return theta, curve
