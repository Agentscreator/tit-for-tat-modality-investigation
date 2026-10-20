# Study 8: an evolving society

A small society where what people want in a partner is heritable and free to take any shape.
Nothing in the agents is the essay's theory. Ideal points, warmth gating and selection-driven
divorce curves either evolve or they don't. We then compare what evolved to the real fitness
landscape, which we can measure directly.

## The model (`model.py`, ~250 lines)

**Population.** About 1,000 people (≈550 adults). They are born, pair, have children and die.
There are 4,000 years (~160 generations) and 8 seeds per condition.

| Trait | What it does |
| --- | --- |
| Warmth W | You deliver sig(W) of what your rung says you'll invest |
| Competence K | Your share of the land (production ∝ e^0.3K), under mutation load |
| Boldness B | More production (e^0.15B), but extra mortality (0.004·e^0.8B per year) |
| Fit F (per pair) | The pot grows ×e^0.6F, and conception rate is 0.5·sig(1+2F) |

**Preferences.** Each agent carries an ideal point, a width and a weight for each of W, K, B
and F. A gate gene ζ decides whether competence is read through perceived warmth.

**The ladder.** Goodwill → willingness → commitment → dedication → devotion.
- Each partner holds their own rung and invests INVEST[rung] = 5/20/40/60/80% of production.
- The pot grows ×1.5, feeds the household's children first, and the rest is split evenly.
- Children need both partners at commitment or above.
- Each year you drop a rung if partner value is below your `drop` gene, otherwise climb if it
  is above `climb`. Dropping below goodwill means leaving.

**Information.** Boldness and competence are visible with a little noise. Warmth and fit are
never seen directly. Each is a Kalman belief: a noisy first impression plus one noisy reading
a year.

**No decay anywhere.** Nobody's traits and no pair's fit ever change.

**Conditions.**
- `main`
- `flat_warmth`: everyone delivers 0.6 regardless of W
- `safe_bold`: boldness has no mortality cost

## Results (`python run.py`, 21 s on 8 cores → `results.json`)

Values are means ± 95% CI across 8 seeds. The pre-registration is in `PREREGISTRATION.md`.

| | Prediction | Result |
| --- | --- | --- |
| P1 | Interior ideal point for boldness | **Confirmed.** It is interior in 7/8 seeds (peak 0.25, population 10th–90th percentile −0.65 to 1.34). The real landscape agrees: partners in the boldest decile raise fewer children (0.82 vs 0.88). |
| P2 | Free boldness → monotone preference | **Not supported.** The population ran to B ≈ 3.6, but the preference was left behind (peak 2.1, below the 10th percentile). Its weight decayed from 0.50 to 0.24. |
| P3 | Warmth and competence preferences monotone | **Falsified.** Both evolved interior peaks in 8/8 seeds. See below. |
| P4 | Warmth gates competence | **Confirmed.** ζ = 0.49 ± 0.07 vs 0.28 ± 0.08 in `flat_warmth`. The real landscape has a W×K interaction of +0.015 ± 0.006 in `main` and +0.002 ± 0.006 without it. (ζ started at 0.5: gating was *kept* where it pays and *eroded* where it doesn't.) |
| P5 | Exit hazard rises, peaks, then falls, with no decay | **Confirmed for committed pairs.** See the hazard table below. |
| P6 | Survivors' partners were always better | **Confirmed, but almost all of it is fit.** Pair fit is +0.41 ± 0.07 SD higher among year-20 survivors than at formation. Partner warmth is only +0.02 ± 0.01. |
| P7 | An ideal-point model beats a linear one | **Confirmed.** Held-out R² is 0.965 vs 0.865 (the essay's simulation found 0.89 vs 0.66). |

The exit hazard per year, for pairs that reached commitment:

| Years | 1–2 | 3–4 | 5–6 | 7–8 | 9–10 | 11–14 | 15–19 | 20–29 | 30+ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hazard | .001 | **.019** | .010 | .006 | .005 | .003 | .002 | .001 | .000 |

Divorce rates:
- 27% of all pairs end by exit; most of those exits come in the first year (dating).
- 8% of committed pairs end by exit.

### What the society teaches beyond the predictions

1. **Ideal-point preferences evolve, but not always for the essay's reason.** The true value
   of a partner, measured by children raised per partner-trait decile, has three shapes:

   | Trait | True shape | Evolved preference |
   | --- | --- | --- |
   | Boldness | A real peak | Matches |
   | Warmth | Rises, then plateaus (top three deciles flat at 0.89–0.90) | Peaks and dips mildly (24% fall from peak to the 90th percentile), roughly matching the plateau |
   | Competence | Rises all the way | Peaks at 1.3 and falls 65% by the 90th percentile |

   So a peaked preference can be:
   - correct (boldness)
   - a fair approximation of saturation (warmth)
   - simply wrong (competence)

2. **The competence error is lag.** Under weak selection, preferences trail the traits they
   judge. Population competence was still rising at year 4,000. The preference tracked it at
   first (ideal 0.2 → 1.2), but the trait overtook it. `safe_bold` shows the extreme: once
   boldness was free, it ran away and the preference never caught up. This is the
   mismatch/lag story from evolutionary psychology, reproduced from nothing.

3. **Fit dominates and starves the trait preferences of selection.** Fit's weight evolved to
   about 0.83, against roughly 0.2–0.3 for each trait. That's because a pair's compatibility
   explains far more fitness (0.54 children per SD) than any partner trait (about 0.1). It is
   also why trait preferences can drift out of calibration.

4. **The ladder evolves into a bang-bang switch.** The evolved `drop` threshold (0.48) sits
   above `climb` (0.38). Agents never hold steady: partner value above 0.48 means climb,
   below means drop. Leaving a devoted partner takes five consecutive bad years, so exits
   from high commitment are slow and rare. That's the essay's asymmetry between building and
   unwinding, here produced by investment rather than by Bayesian confidence.

## Caveats, including deviations from the pre-registration

- **Calibration happened after the pre-registration, before the main run.** First runs had
  near-zero divorce. Leaving never paid, because household pots never limited child survival.
  I made four changes:
  - raised child need
  - added pair fit, which is the essay's own "heterogeneous match quality"
  - let fit affect conception (infertility is a leading cause of divorce cross-culturally;
    Betzig 1989)
  - fixed a bug where climbing overrode dropping

  None of these targeted a prediction's outcome, but they were made after seeing exit rates.
- **The P5 rise is partly mechanical.** Committed pairs can't exit before year 3, because the
  ladder needs time. The *falling tail* is the real result. Exits also bunch in alternate
  years because the ladder steps one rung a year, so hazards are reported in 2-year bins.
- **Reporting choices made after seeing calibration runs.** Splitting the hazard by commitment
  and the decile/decline analyses were both decided then.
- **Competence hasn't reached equilibrium by year 4,000.**
- **Simplifications.** There are no sexes, no kin beyond parents, and no extra-pair mating.
  P2 and P3 failed as stated, and I've left them that way.
