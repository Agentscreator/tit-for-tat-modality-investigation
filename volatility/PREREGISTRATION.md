# Study 7 pre-registration: does volatility set the memory of trust?

Written and committed **before** the learned agent was trained or the decay simulation run.
Only the Bayes-optimal predictions (`predict.py` → `predictions.json`) existed at this point.
They are the theory being tested, not results.

## Step 0: theory commitment

- **Prior.** The essay's reconciliation asymmetry, where a known partner takes 7× longer to
  re-accept than a stranger, and our original "betrayal tolerated grows linearly with history"
  both hold only when partners never change. Any volatility should cap how much history counts.
- **Mechanism.** If a partner can turn with hazard λ per round, a good history can't push
  the posterior P(partner is bad) below a floor of about λ. Past that floor, more good
  rounds add nothing, so the grace period saturates.
- **Falsifier.** An agent that learns from scratch in a volatile world keeps extending grace
  with history, or doesn't shorten grace as volatility rises.

## Step 2: the essay's map

1. **Core problem.** Why relationships end, and why reversing a judgment is hard.
2. **One irreducible idea.** Perceived value is a Bayesian posterior about a *fixed* truth.
3. **Most fragile assumption.** The truth is fixed: partners don't change.
4. **What breaks if wrong.** "Confidence is the obstacle" to reconciliation, the linear
   growth of grace, and "decline is learning, not decay" all weaken.
5. **What can be falsified now.**
   - "A decay model cannot produce a falling hazard tail."
   - The 7× asymmetry under volatility.
6. **Learning theory assumed.** Stationary Bayesian updating, where all evidence is
   exchangeable.
7. **Competing theory ruled out.** Volatility-tracking learning: humans raise their learning
   rate in volatile environments (Behrens et al. 2007, *Nat. Neurosci.*). A believer in
   that theory would see the essay's asymmetries as artifacts of a static world.

## Environment (`env.py`)

- **Partners.** A new partner is bad from the start with probability 0.2. Each round, a
  good partner turns bad with probability λ ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}.
- **Behaviour.** Good partners defect 10% of the time and bad partners 50%.
- **Payoffs.** Staying pays +1 per cooperation and −2 per defection.
- **Leaving.** Leaving costs 5 rounds of search. Lives last 300 rounds.

## Agents

- **Bayes.** Exact posterior with known λ, plus the reward-maximising leave threshold.
  This is the normative benchmark.
- **Tally.** The original model's belief, (C − D) / (1/p + n), with θ and p tuned per λ.
- **Human.** The v2 heuristic: negativity bias β, doubt floor *f* and tolerance θ, all
  tuned per λ.
- **GRU.** An 8-unit recurrent network that sees only "the partner defected / cooperated"
  and outputs stay or leave. It is trained from scratch per λ by evolution strategies, 3
  seeds each. No belief structure is built in.

## Bayes-optimal predictions (from `predictions.json`)

This table shows grace: rounds until leaving after G good rounds, once the partner turns
and starts defecting half the time.

| λ | G = 0 | 10 | 20 | 40 | 160 | Saturation G* = ln(floor / prior) / LLR_C |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 15 | 28 | 49 | 190 | ∞ (linear) |
| 0.001 | 4 | 10 | 10 | 10 | 10 | 9.0 |
| 0.003 | 4 | 8 | 8 | 8 | 8 | 7.1 |
| 0.01 | 4 | 6 | 6 | 6 | 6 | 5.1 |
| 0.03 | 3 | 4 | 4 | 4 | 4 | 3.2 |
| 0.1 | 2 | 2 | 2 | 2 | 2 | 1.2 |

The Bayes negativity ratio is LLR_D / |LLR_C| = ln 5 / ln 1.8 = 2.74: one defection is worth
2.74 cooperations because it is more diagnostic.

## Hypotheses and falsifiers

The GRU is evaluated with the mean of its final search distribution. Grace is the median
over 400 scripted sequences, averaged over 3 seeds.

| ID | Hypothesis | Falsified if |
| --- | --- | --- |
| H1 | In volatile worlds (λ ≥ 0.003), the learned agent's grace saturates with history | GRU grace(G=160) > 1.5 × grace(G=20) for any λ ≥ 0.003 |
| H2 | Saturated grace shortens as volatility rises | GRU grace(G=160) at λ = 0.003 is less than 2× its value at λ = 0.1, or it rises in more than one of the four steps between them |
| H3 | The learned agent represents the Bayes posterior, not a tally | At λ ≥ 0.01, held-out R² of a linear probe from GRU hidden state to Bayes log-odds is < 0.8, or not at least 0.1 above R² to the tally belief |
| H4 | Perfect memory (tally) is only near-optimal in stable worlds | Tuned tally earns ≥ 90% of Bayes reward at λ ≥ 0.03, or < 95% at λ = 0 |
| H5 | Competence check: the GRU learns the task | GRU earns < 90% of Bayes reward at some λ; H1–H3 are then *inconclusive* at that λ, not confirmed |
| H6 (exploratory) | At λ = 0, a bounded learner won't reproduce the unbounded Bayes growth, because the training data never shows a long good history followed by a turn | GRU grace(G=160) ≥ 2 × grace(G=20) at λ = 0 |

## Part A: the essay's hazard claim (`decay_vs_learning.py`)

| ID | Hypothesis | Falsified if |
| --- | --- | --- |
| A1 | A pure decay model with heterogeneous decay rates produces a hazard that rises, peaks and falls (contradicting the essay) | Its annual hazard is monotone |
| A2 | It also produces "survivors were always better": higher starting value among long survivors | Survivors' starting value isn't higher than the cohort's |
| A3 | A martingale test separates the models: mean within-couple change in perceived value is ≈ 0 under calibrated learning, < 0 at every duration under decay, and < 0 early then ≈ 0 under learning with an optimistic prior | Calibrated learning drifts (\|mean change\| > 0.5 points/yr in any year with ≥ 200 couples), or decay doesn't |

Empirical anchor for A3: within-couple relationship satisfaction declines on average,
steepest in the early years (Bühler, Krauss & Orth 2021, *Psychological Bulletin*, 147,
1012–1053). Under A3, that rules out calibrated learning as the whole story. It leaves
open optimistic-prior learning, where the decline flattens, and decay, where it continues.
