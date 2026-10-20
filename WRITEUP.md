# Tit for Tat Can't Leave: Trust, Exit, and Accumulated Belief

*October 19, 2026*

## Summary

**The question.** Is leaving a relationship the mirror image of entering one?

**The short answer.** It depends on how the agent remembers.

An agent with perfect memory, like the agents in our original model, never forgets a good
round. Every good round buys its partner about 2 more rounds of betrayal before the agent
leaves, with no upper limit. Such an agent also never leaves a partner who defects less
than half the time.

When evolution was allowed to give agents human-like memory, it did. Agents evolved to
weigh a betrayal about twice as heavily as a cooperation, and to let old evidence fade.
With that memory, a long history buys a partner only about 10 rounds of grace, and
inconsistent partners usually get caught. The price is twitchiness: some agents leave
partners who are almost always good.

This writeup does three things:

1. **Simplifies** the original six phases down to one equation and one small model.
2. **Tries to falsify** the original claims. Three of five were overturned.
3. **Rebuilds** the model with human-like dynamics.

The original six-phase writeup is preserved in [`legacy/WRITEUP_v1.md`](legacy/WRITEUP_v1.md).

## 1. Simplification: one equation

**The world.** Agents live for 80–120 rounds. Each round, unpaired agents are matched
with others of similar value, with some noise. Paired agents play a prisoner's dilemma
with three options: cooperate, defect, or exit to end the pairing. Each side misreads the
other's move 10% of the time. Evolving agents carry a few genes, and the top 15% by
payoff each generation seed the next one.

**The belief.** The original model's "Kalman filter" turns out to be a damped tally. With
a starting belief of 0 and observation noise 1:

```
belief = (cooperations − defections) / (1/p + rounds seen)
```

An agent leaves when belief < −θ, where θ is its **tolerance** and p its openness to a new
partner. This one inequality reproduces the original phases' main results without
simulation:

| Original result | Closed form | Value |
| --- | --- | --- |
| Phase 3: betrayal tolerated per prior good round | (1 + θ) / (1 − θ) | 2.37 at θ = 0.41 |
| Phase 4: partners who defect less often than this are never left | q* = (1 + θ) / 2 | 0.72 at θ = 0.43 |
| Phase 5: a balanced split of effort beats specialising when | γρ < 1 | conjunctive (ρ < 0): always |

`model.py` contains one configurable simulation that covers every world we've run.
`experiments.py` runs everything in about 90 seconds. The v1 setting reproduces the
original lag figures exactly (1, 25, 96, 191 and 380 rounds), and re-evolving it gives the
same genome as the original Phase 2 within error:

| Gene | Original (Phase 2) | Simplified model |
| --- | --- | --- |
| tolerance θ | 0.41 ± 0.08 | 0.36 ± 0.07 |
| retaliation | 0.70 ± 0.02 | 0.67 ± 0.04 |
| first-contact threshold | 18.7 ± 4.3 | 16.3 ± 1.0 |

## 2. Falsification

For each claim we wrote down, before running anything, the result that would kill it.

| Claim | Falsifier | Result | Verdict |
| --- | --- | --- | --- |
| **A.** Tolerated betrayal grows linearly with shared history | Any forgetting makes it level off | Pure betrayal tolerated after 40, 160 and 320 good rounds: 10, 10 and 10 rounds (v2 agents) | **Overturned** for agents that forget; holds only with perfect memory |
| **B.** Exit cost doesn't change what evolves (Phase 2) | A large enough cost shifts the genome | 30-round cooldown: θ 0.36 → **0.64** | **Overturned**: the original null came from a weak manipulation |
| **C.** Evolution leaves the blind spot because agents never meet inconsistent partners (Phase 6) | Add inconsistent partners; θ should move | θ 0.43 → 0.40, no change | **Overturned**: exposure wasn't the problem |
| **D.** Conjunctive scoring favours balance (Phase 5) | — | Follows from γρ < 1 | True by construction, not an empirical finding |
| **E.** Tit for tat has a fat losing tail | — | Re-run: 1.60 per round vs 2.36 for evolving agents | Holds, by definition: tit for tat can't exit |

**B in practice.** When leaving costs 30 rounds out of the market, evolution makes agents
far more tolerant. Each good round then buys 4.6 rounds of betrayal instead of 2.1. The
cutoff for inconsistent partners rises from 0.68 to 0.82. Costly exit doesn't just
lengthen bad relationships; it changes the kind of agent that wins.

**Why C failed.** The real limit is the tally itself. Its cutoff is q* = (1 + θ) / 2,
which is at least 0.5 for any tolerance. No setting of θ can catch a partner who defects
40% of the time, so experience alone can't fix the blind spot. In the population, partners
who are inconsistent from round one are mostly caught at once, by first impressions. The
dangerous ones are those who are good first and turn later, and a tally weighs that good
history most heavily. Fixing it took new machinery: section 3.

## 3. Rebuild with human-like dynamics

**What's added.** The v2 world keeps Phase 6's high stakes. Cooperation pays 4 × the
partner's value / 50, being exploited costs 4, and mutual defection costs 1. Partners
also shape each other's value: 5% pull toward the partner, +0.5 for each cooperation they
play and −1 for each defection.

On top of that, v2 adds three human-like traits. Each one is a gene that evolution can
switch on or leave off:

| Trait | Psychology it models | Gene |
| --- | --- | --- |
| Negativity bias | "Bad is stronger than good" | A defection counts as −β instead of −1 |
| Fading memory | Recent events dominate | Belief always updates by at least a floor *f*, so old evidence fades |
| Comparison with alternatives | People with better prospects leave sooner | The exit bar rises with the agent's own value and youth |

The population mixes four kinds of partner besides the evolving agents: tit for tat,
pure defectors, *inconsistent* partners who defect 20–80% of the time, and *turners* who
cooperate until a random point in life, then defect 50–100% of the time. Lives last 120
rounds.

### Results

**Evolution switched on two of the three traits.**

| Gene | v2 without human traits | v2 with human traits | Neutral drift would give |
| --- | --- | --- | --- |
| negativity bias β | fixed at 1 | **2.03 ± 0.33** | ~2.5 (middle of the range) |
| memory floor *f* | fixed at 0 | **0.08 ± 0.01** | ~0.15 |
| comparison with alternatives | fixed at 0 | 0.48 ± 0.11 | ~0.5 |
| tolerance θ | 0.40 ± 0.13 | 0.59 ± 0.07 | |
| retaliation | 0.21 ± 0.03 | 0.18 ± 0.02 | |

A defection ends up counting about twice as much as a cooperation, and memory spans
about 12 rounds (1 / *f*). Both values sit clearly away from where neutral drift would
leave them. Comparison with alternatives stayed at its drift value: there was no selection
on it. Evolution also raised tolerance to offset the negativity bias.

**1. History stops buying unlimited grace.**

| Good rounds first | 0 | 10 | 20 | 40 | 80 | 160 | 320 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Perfect memory (v1): rounds of betrayal tolerated | 1 | 22 | 44 | 86 | 170 | 335 | 664 |
| Human-like (v2): rounds of betrayal tolerated | 1 | 8 | 9 | 10 | 10 | 10 | 10 |

With fading memory, a good history raises belief to a ceiling and no further. The grace
period then has a closed form that doesn't depend on history:

```
grace = ln((1 + β) / (β − θ)) / −ln(1 − f)
```

That gives 8.1 rounds for the mean genome; the measured median is 10. In this model, 320
good rounds buy no more benefit of the doubt than 40 do.

**2. The blind spot mostly closes, at a price.** This table shows the share of agents who
leave within 1,000 rounds after 20 good rounds, with perfect observation:

| Partner defects | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 100% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tally agents (v2 without human traits) | 0% | 0% | 0% | 0% | 4% | 26% | 49% | 72% | 96% |
| Human-like agents | 16% | 36% | 52% | 69% | 80% | 88% | 94% | 96% | 99% |

The negativity bias moves the average cutoff from 0.70 to 0.54. The memory floor lets a
run of bad luck cross the line even below that cutoff, so partners who defect 30% of the
time are now left half the time. The cost is twitchiness: 16% of agents walk away from a
partner who is bad only 10% of the time. That is about how a good partner looks through
10% misperception. Staying a little sceptical forever is evolutionarily stable here,
which answers an open question from earlier: healthy scepticism holds up.

**3. A bad first partner leaves a smaller mark.** Here we compare evolving agents whose
first partner was a defector, a mostly-defecting inconsistent partner, or a turner who
had already turned, against everyone else (last 10 generations, pooled over seeds):

| | Tally agents | Human-like agents |
| --- | --- | --- |
| Lifetime payoff per round, good first partner | 4.19 | 4.38 |
| Lifetime payoff per round, bad first partner | 3.26 (−22%) | 3.88 (−11%) |
| Value gained over life, good vs bad first partner | +34.2 vs +27.2 | +36.4 vs +32.7 |

Human-like belief halves the lasting cost of a bad start. A bad first partner is dropped
sooner, before the value scarring compounds.

**4. Comparison with alternatives did nothing.** Bad partners were left within a median of
1 round regardless of the agent's age or value. Mostly that happened at first contact,
before prospects could matter. In this world, who stays in bad relationships isn't
explained by their options.

## Limitations

- **The controlled tests use perfect observation.** The evolution runs used 10%
  misperception.
- **Section 3's first-partner comparison has no confidence intervals.** It pools agents
  across seeds, and agents are not randomly assigned to a bad first partner.
- **Comparison with alternatives may need a different world.** Lives are short, and most
  bad partners are caught at first contact. A longer-lived world, where agents commit
  before learning who their partner is, might give the trait something to do.
- **Mean payoff doesn't clearly improve with human traits.** It was 4.30 ± 0.24 against
  4.04 ± 0.31, which is within noise, so the traits' benefit is in the tails, not the
  average.
- **Channels from Study 5 are not modelled in v2.** The specialists-vs-generalists result
  stands only as the closed-form rule in section 1.

## Reproducing

```sh
pip install numpy
python experiments.py   # all results in this writeup → results.json, genomes.json (~90 s on 8 cores)
```

The original scripts and their outputs are in `legacy/`. Run them from inside that
folder.
