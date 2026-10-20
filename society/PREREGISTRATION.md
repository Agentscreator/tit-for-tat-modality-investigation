# Study 8 pre-registration: what partner preferences does a society evolve?

Written before any run of `model.py`.

## The question

The essay claims that people judge partners against an ideal point, not a weighted sum. It
also claims that warmth gates competence. Study 8 doesn't build either into the agents.
Each agent's preference for each trait is heritable and can take any shape:
- flat
- monotone (the ideal sits beyond the population's range)
- a true interior peak, narrow or wide

A gate gene decides whether competence is judged through warmth. Then we ask what evolution
builds and whether it matches the real fitness landscape.

## World (one model, three switches)

**Traits.** There are three heritable traits:
- warmth W: the fraction of your investment you actually deliver
- competence K: production
- boldness B: higher production, but higher mortality

**Relationships.** A relationship is a ladder:
goodwill → willingness → commitment → dedication → devotion.
- Each partner holds their own rung and invests more of their production the higher it is.
- The shared pot grows ×1.5, feeds the children, and the rest is split evenly.
- Children need both partners at commitment or above.
- Warmth is never seen directly. It is learned from one noisy reading a year.

**Conditions.**
- `main`
- `flat_warmth`: everyone delivers the same share, so warmth does nothing
- `safe_bold`: boldness has no mortality cost

## Predictions

| ID | Prediction | Falsified if |
| --- | --- | --- |
| P1 | Boldness preference evolves an interior ideal point in `main` | Evolved ideal for B is above the 90th percentile of the population's B |
| P2 | Without its cost (`safe_bold`), the B preference becomes monotone | B ideal in `safe_bold` is within the population's 10th–90th percentile |
| P3 | Warmth and competence preferences are monotone (ideal above the 90th percentile) | Either ideal lies inside the 10th–90th percentile range |
| P4 | The warmth gate on competence evolves because a cold partner's output never reaches you | Mean ζ in `main` isn't higher than in `flat_warmth` (seed-level, 95% CI) |
| P5 | The exit hazard by duration rises, peaks between years 2 and 10, then falls, with no decay in the model | Peak at year ≤ 1 or ≥ 15, or hazard at years 20–30 ≥ peak / 2 |
| P6 | Selection, not growth: the mean true warmth of partners rises with duration among intact pairs | Year-20 survivors' partner W ≤ year-0 cohort's |
| P7 | The evolved preference functions are better fit by an ideal-point model than a linear one | Mean held-out R²(ideal) − R²(linear) < 0.05 |
