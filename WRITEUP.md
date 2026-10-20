# Tit for Tat Can't Leave: Trust, Exit, and Accumulated Belief

*October 19, 2026*

## Summary

We asked whether leaving a relationship mirrors entering it: does trust that took a long
time to build also take a long time to lose? In this model, yes, and the cost of leaving
grows in a straight line. The typical evolved agent sits through about **2.4 rounds of
continuous betrayal for every good round that came first**.

- **Phase 1:** tit for tat does fine on average but has a bad tail. 34% of tit-for-tat
  agents get stuck in mutual defection with a pure defector for almost the whole run,
  because they can retaliate but never leave.
- **Phase 2:** the cost of exit didn't change what evolved. Free, flat and tenure-scaled
  exit costs all led to the same genome, within error bars.
- **Phase 3:** a partner cooperates for G rounds, then defects forever. The rounds of
  betrayal an agent tolerates before leaving grow linearly with G, at a slope of 2.37.
- **Phase 4:** a partner who betrays only *some* of the time is almost never caught. After
  20 good rounds, no agent leaves a partner who defects 30% of the time. Only 4% leave one
  who defects half the time. But staying is usually not the disaster it sounds like. The
  real cost sits in the middle band, where the partner is bad 50–80% of the time.

**Correction to an earlier figure.** An earlier reading put the Phase 3 slope at 1.27.
That number is a straight-line fit through all ten Phase 3 points. The top points are cut
off by the 400-round limit in the test, which pulls the fit down. The belief update has an
exact solution, and it gives a slope of (1 + θ) / (1 − θ) = 2.37 for the evolved genome.

## The model

The population is 320 agents: 200 evolving, 40 tit for tat and 80 pure defectors. They
play in discrete rounds, 80 per generation.

**The world.** Each agent has a value from 1 to 100 that stands for how good a partner it
is. Starting values are drawn around a neighbourhood baseline, mean 50 and SD 15. Each
round, an agent's value moves 1% of the way toward its partner's value, or toward the
population mean when unpaired. Good company pulls you up and bad company pulls you down.
Agents also age, and their chance of finding a partner follows a bell curve that peaks at
age 15, has width 20 and never drops below 0.05. Opportunity shrinks over a lifetime.

**Matching.** Each round, an unpaired agent enters the matching pool with probability
opportunity × 0.6. Agents in the pool are sorted by value plus Gaussian noise (SD 10) and
paired with their neighbours. So matching is assortative but noisy: a high-value agent
sometimes ends up with someone far below it.

**The game.** Each round, both partners choose at the same time: cooperate, defect or
exit.

| Outcome | Payoff to each |
| --- | --- |
| Both cooperate | 3 and 3 |
| One defects on a cooperator | 5 to the defector, −1 to the cooperator |
| Both defect | 0 and 0 |
| Either exits | 0 and 0; the pair splits up |

After an exit, both agents wait out a cooldown before they can be matched again: **free**
= 1 round, **flat** = 2 rounds, **entangled** = 0.5 × rounds together, capped at 10. Each
side also misreads the other's move 10% of the time.

## Agents and the belief mechanism

- **Pure defectors (80)** always defect. They stand for the bad-faith share of any real
  society.
- **Classic tit for tat (40)** cooperates first, then copies whatever its partner just did.
  It can never exit. It can only retaliate or forgive.
- **Evolving agents (200)** each carry a four-gene genome:

| Gene | Range | What it does |
| --- | --- | --- |
| `threshold` | 0–80 | Value floor: below it, the agent leaves a new partner on first contact |
| `prior_variance` | 0.2–5 | Openness to a new partner; higher means early rounds move belief more |
| `belief_exit_threshold` | −1 to 0 | The agent leaves once its belief falls below this |
| `retaliate_prob` | 0–1 | Chance of defecting back after a defection, rather than absorbing it |

**The belief update.** Evolving agents don't count strikes. Each holds a running belief
about its partner, starting at 0 with variance *p* = `prior_variance`. Each round it
records an observation *o*, +1 for cooperate and −1 for defect, and applies a
one-dimensional Kalman update:

```
k = v / (v + 1)        mean ← mean + k·(o − mean)        v ← v·(1 − k)
```

The gain *k* shrinks as uncertainty shrinks. A new belief swings hard on one bad round. A
belief built over 80 good rounds barely moves on one more bad round. That isn't the agent
choosing to forgive; accumulated certainty gives each new observation less weight.

With prior mean 0 and observation noise 1, the filter reduces to a closed form. After *n*
observations, belief = sum(observations) / (1/p + n). It is a damped tally of cooperations
minus defections. That's why Phases 3 and 4 behave the way they do.

## Phase 1: tit for tat's tail problem

On average tit for tat earns two-thirds of what evolving agents earn, and its bottom 10%
loses money. We ran one 80-round generation under five seeds, with entangled exit cost
and 10% misperception. The evolving agents had random genomes; nothing had evolved yet.
Values are the mean across seeds ± 95% CI.

| Measure | Evolving agents | Tit for tat | Pure defectors |
| --- | --- | --- | --- |
| Mean payoff | 195.3 ± 4.4 | 130.9 ± 8.1 | 21.7 ± 1.1 |
| 10th-percentile payoff | 150.8 ± 8.9 | −9.5 ± 1.5 | — |

**The trap.** 34% ± 5.5% of tit-for-tat agents got paired with a pure defector and stayed
in mutual defection. Those that were trapped stayed trapped for 78.0 ± 0.5 of 80 rounds.
Tit for tat can punish forever but can never leave. Any agent that can exit, even one with
a random genome, avoids this tail.

## Phase 2: exit cost doesn't change what evolves

All three exit-cost regimes converged to the same genome, within 95% CIs on every gene.
Each regime ran for 100 generations under five seeds. Each generation kept the top 15% of
evolving agents and refilled the population with mutated copies, mutation SD = 10% of each
gene's range, plus 5% random newcomers. Values are averaged over the last 10 generations,
then across seeds.

| Gene or outcome | Free exit | Flat cost | Entangled cost |
| --- | --- | --- | --- |
| `threshold` | 18.7 ± 4.3 | 16.4 ± 4.2 | 18.2 ± 4.2 |
| `prior_variance` | 2.06 ± 0.44 | 2.41 ± 0.67 | 2.50 ± 0.47 |
| `belief_exit_threshold` | −0.41 ± 0.08 | −0.46 ± 0.09 | −0.41 ± 0.06 |
| `retaliate_prob` | 0.70 ± 0.02 | 0.69 ± 0.03 | 0.70 ± 0.02 |
| Evolving mean payoff | 182.7 ± 5.1 | 180.6 ± 5.2 | 182.7 ± 3.4 |
| Tit-for-tat mean payoff | 124.1 ± 3.4 | 132.8 ± 2.6 | 125.3 ± 1.8 |

The consensus genome: skip partners with value below about 18; start moderately open;
leave once belief drops below about −0.4; retaliate against 70% of defections.

The null may come from a weak manipulation. Exit cost here is only a cooldown of 1, 2 or
at most 10 rounds out of 80, with no payoff penalty.

## Phase 3: betrayal tolerated scales with trust built

For the typical evolved agent, every good round buys a partner about 2.4 more rounds of
betrayal before the agent leaves. We took the final genomes evolved under free exit (200
per seed, 5 seeds) and ran them outside the population. Each faced a partner who
cooperates for G rounds, then defects forever.

| Good rounds first (G) | Measured median rounds to leave | Exact solution, mean genome | % that never left (test cap) |
| --- | --- | --- | --- |
| 0 | 1.0 | 1 | 2.6% |
| 10 | 24.4 | 25 | 4.9% |
| 20 | 45.5 | 48 | 7.7% |
| 40 | 83.3 | 96 | 13.8% |
| 80 | 143.8 | 191 | 30.1% |
| 160 | 195.7 | 380 | 73.9% |

An agent with exit threshold −θ and prior variance *p* leaves after *B* rounds of betrayal,
where

```
B > [ G·(1 + θ) + θ/p ] / (1 − θ)
```

The per-round slope is (1 + θ) / (1 − θ). For the free-exit mean genome (θ = 0.407,
p = 2.06), that is 2.37.

The measured medians follow the exact line up to about G = 40. After that they fall
below it, because `phase3.py` stops at 400 total rounds. At G = 160 only 240 rounds of
betrayal remain, so the agents that would leave latest are counted as "never left" and
drop out of the median. The earlier 1.27 slope came from a straight-line fit through those
cut-off points.

## Phase 4: intermittent betrayal

**Question.** Real bad behaviour is rarely all-or-nothing. In behavioural psychology,
behaviour learned under intermittent reinforcement is harder to extinguish than behaviour
learned under constant reinforcement. Is a partner who is bad *some* of the time harder to
recognise than one who is bad all the time?

**Design.** We used the same 1,000 free-exit genomes as Phase 3. The partner cooperates
for G rounds, then defects with probability q each round; q = 1 is the Phase 3 clean
switch. Each genome got 5 random betrayal schedules, and we followed it for up to 1,000
rounds after the turn. This time the cap counts only rounds after the turn, so large G
isn't cut off. The agent plays its own genome throughout, retaliating with probability
`retaliate_prob`, and we record its payoff while it stays.

**Prediction.** The belief settles at 1 − 2q. An agent leaves only if that falls below its
exit threshold −θ, which needs q > (1 + θ) / 2. The mean evolved θ is 0.431, so the
predicted cutoff is q ≈ 0.72.

### Results

After 20 good rounds:

| Partner defects (q) | % who leave within 1,000 rounds | Median rounds to leave | Agent's payoff per round while staying |
| --- | --- | --- | --- |
| 10% | 0% | — | 2.73 |
| 30% | 0% | — | 2.16 |
| 50% | 4.4% | 327 | 1.52 |
| 60% | 25.8% | 185 | 1.16 |
| 70% | 48.0% | 119 | 0.80 |
| 80% | 68.3% | 81 | 0.43 |
| 90% | 84.6% | 61 | 0.06 |
| 100% | 95.8% | 47 | −0.32 |

After 80 good rounds, the same pattern is stretched out. A half-bad partner is left by
0.1% of agents, a 70%-bad partner by 38% after a median of 381 rounds, and an always-bad
partner by 90% after a median of 172 rounds.

**1. Intermittent betrayal goes undetected.** Once there is any history, a partner who
defects 30% of the time is never left, and one who defects half the time almost never is.
The 50% crossover sits right at the predicted cutoff (48% leave at q = 0.7). Each
cooperative round pulls the tally back up. The belief stops drifting and settles, so it
never reaches the exit line. This matches the intermittent-reinforcement intuition,
although the mechanism here is averaging, not learning.

**2. But mostly it isn't worth detecting.** For comparison, an evolving agent in the full
population earns about 2.28 per round (182.7 over 80 rounds in Phase 2, counting time
spent unpartnered). A partner who defects 30% of the time still pays 2.16 per round, about
the same as the average outcome. Because the agent retaliates 70% of the time, it gets
some of the defection payoff back. Staying with a partner who is bad 30% of the time is a
reasonable deal in this world, not a trap.

**3. The real cost is in the middle band.** A partner who defects 50–80% of the time pays
only 0.4–1.5 per round, well under the 2.28 average. Yet most agents stay for hundreds of
rounds, and a long history makes it worse. This is where the recognition lag hurts. The
partner is bad enough to cost a lot and mixed enough that the belief never breaks.

**4. The first impression is the opposite.** With no history (G = 0), a single defection
in round one drops a fresh belief below the line. So 28% of agents leave a 30%-bad partner
immediately, and 54% leave a half-bad one. Without history the agent is twitchy, and with
history it is blind to mixed signals. Both come from the same filter.

## Discussion and limitations

The ability to exit is what saves evolving agents from tit for tat's tail. The same belief
mechanism makes them slow to use it. A long good history buys a partner a long run of pure
betrayal at about 2.4 to 1. A partner who mixes good and bad rounds is never caught at
all, unless more than about 70% of the rounds are bad.

- **Phase 3 is censored.** `phase3.py` stops at 400 total rounds, so the "never left"
  shares are an artefact of the cap, not agents that would truly stay forever. Phase 4
  fixes this by capping only the rounds after the turn.
- **Phases 3 and 4 have no noise.** The controlled tests feed perfect observations, but
  evolution ran with 10% misperception.
- **Phases 3 and 4 don't use the population.** The results follow from each genome's
  two belief genes; evolution only picks θ and p. Running the tests inside the live
  population would show whether selection eventually punishes a long recognition lag.
- **Phase 1 uses random genomes.** The benchmark compares tit for tat with untrained
  exiters, not evolved ones.
- **Phase 2's manipulation is weak.** Exit cost is only a cooldown of 1 to 10 rounds, with
  no payoff penalty.
- **The 2.28 benchmark is rough.** It is the average payoff per round across a whole
  generation, not the measured value of leaving and finding someone new.

## Reproducing

```sh
pip install numpy
python run_all.py   # Phases 1–2 → results.json
python phase3.py    # Phase 3   → phase3.json
python phase4.py    # Phase 4   → phase4.json (evolves and caches genomes_free.json, ~2 min)
```

`run_all.py` and `phase3.py` write their output to the hardcoded path
`/home/claude/study7/`. Change that path before running. `phase4.py` writes next to
itself.
