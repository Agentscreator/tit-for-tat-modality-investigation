# tit-for-tat-modality-investigation

An agent-based model of a repeated prisoner's dilemma with an **exit** option. It asks
whether leaving a relationship is the mirror image of entering one: does trust that took
a long time to build also take a long time to lose?

## Findings

- **Phase 1, tit for tat's tail problem.** Classic tit for tat can retaliate but never
  leave. 34% ± 5.5% of tit-for-tat agents get stuck in mutual defection with a pure
  defector for about 78 of 80 rounds. Their 10th-percentile payoff is −9.5. Agents that
  can exit score 150.8 at the 10th percentile, even with random genomes.
- **Phase 2, exit cost doesn't matter.** We evolved genomes for 100 generations under
  free, flat and tenure-scaled ("entangled") exit costs. All three converged to the same
  genome within 95% CIs: `threshold` ≈ 18, `prior_variance` ≈ 2.1–2.5,
  `belief_exit_threshold` ≈ −0.41 to −0.46, `retaliate_prob` ≈ 0.70.
- **Phase 3, betrayal tolerated scales with trust built.** An evolved agent faces a
  partner who cooperates for G rounds, then defects forever. The rounds of betrayal
  before the agent exits grow linearly with G, at a slope of (1 + θ) / (1 − θ), where −θ
  is the agent's exit threshold. For the mean evolved genome (θ = 0.407) that is
  **≈ 2.37 rounds of betrayal per prior good round**.

  > An earlier reading put this slope at 1.27. That number is a straight-line fit through
  > all ten points in `phase3.json`. But `phase3.py` stops each test at 400 total rounds
  > (`MAXR`), so at large G the agents that would leave latest are recorded as "never
  > exits" and dropped from the median (73.9% of agents at G = 160). That pulls the fit
  > down. Up to G = 40, the measured medians match the exact solution.

- **Phase 4, intermittent betrayal.** After G good rounds, the partner defects with
  probability q each round instead of every round. After 20 good rounds, 0% of agents
  leave a partner who defects 30% of the time, and 4.4% leave one who defects half the
  time. Leaving only becomes common once q is above (1 + θ) / 2 ≈ 0.72. A 30%-bad partner
  still pays 2.16 per round, close to the population average of about 2.28. The costly
  blind spot is the 50–80% band, where staying pays 0.4–1.5 per round and most agents
  stay for hundreds of rounds.
- **Phase 5, specialists vs generalists.** Agents split 100 points across trust,
  attraction and provision, and evolution picks the split. Under conjunctive (near-min)
  scoring, populations stay close to balanced (6–15% specialists) even when specialising
  pays. Under additive scoring, specialists take over (96%) only when returns compound.
  Otherwise the split drifts at random.
- **Phase 6, high stakes.** In this phase, cooperation pays 4 × the partner's value / 50,
  the sucker loses 4, and mutual defection costs 1. Partners also pull each other's value
  up or down every round. Evolution responded by making agents pickier at the door
  (threshold 19 → 33) and less retaliatory (0.70 → 0.22), but the exit rule didn't move.
  74% of agents still stay with a partner who defects 60% of the time, now losing 0.57 a
  round.

The full writeup is in [`WRITEUP.md`](WRITEUP.md).

## Model

- **Population:** 200 evolving agents, 40 tit for tat and 80 pure defectors; 80 rounds
  per generation; each side misreads the other's move 10% of the time.
- **World:** each agent has a value from 1 to 100. Each round it moves 1% of the way
  toward its partner's value, or toward the population mean when unpaired. The chance of
  joining the matching pool follows a bell curve over age, so opportunity shrinks over a
  lifetime.
- **Matching:** agents in the pool are sorted by value plus noise (SD 10) and paired with
  their neighbours, so matching is assortative but noisy.
- **Payoffs:** both cooperate 3/3 · defector 5, cooperator −1 · both defect 0/0 · exit
  0/0, and the pair splits. After an exit both agents sit out a cooldown: 1 round (free),
  2 rounds (flat) or min(10, 0.5 × rounds together) (entangled).
- **Evolving genome:** `threshold` (value floor: below it, the agent leaves a new partner
  on first contact), `prior_variance` (openness to a new partner),
  `belief_exit_threshold` (leave once belief drops below this), `retaliate_prob`
  (chance of defecting back after a defection).
- **Belief:** a one-dimensional Kalman update over observations of +1 (cooperate) and −1
  (defect). The update gain shrinks as certainty grows, so a long good history makes one
  betrayal count for little. With prior mean 0 and observation noise 1, the belief equals
  sum(observations) / (1/p + n).

## Files

| File | Contents |
| --- | --- |
| `core.py` | Agents, genome, belief update, payoffs, matching |
| `engine.py` | Generation runner and evolutionary loop with exit-cost modes |
| `run_all.py` | Phases 1 and 2 → `results.json` |
| `phase3.py` | Phase 3 controlled betrayal test → `phase3.json` |
| `phase4.py` | Phase 4 intermittent betrayal test → `phase4.json` |
| `phase5.py` | Phase 5 channel-allocation tournament → `phase5.json` |
| `phase6.py` | Phase 6 high-stakes world with value coupling → `phase6.json` |
| `results.json`, `phase3.json`–`phase6.json` | Saved results (seeds 1–5) |
| `genomes_highstakes.json` | Final genomes evolved in the Phase 6 world |
| `genomes_free.json` | Final genomes evolved under free exit, 5 seeds × 200 (cached by `phase4.py`) |
| `WRITEUP.md` | Full writeup of all six phases |

## Running

```sh
pip install numpy
python run_all.py   # Phases 1–2
python phase3.py    # Phase 3
python phase4.py    # Phase 4 (~2 min the first time, while it evolves genomes)
python phase5.py    # Phase 5 (~2 min)
python phase6.py    # Phase 6 (~2 min; run Phase 4 first)
```

`run_all.py` and `phase3.py` write their output to the hardcoded path `/home/claude/study7/`. Change that
path to a local one before running. To remove the Phase 3 cut-off, raise `MAXR` in
`phase3.py`. At G = 160 the mean genome needs about 380 rounds of betrayal on top of the
160 good rounds.
