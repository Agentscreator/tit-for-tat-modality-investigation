# tit-for-tat-modality-investigation

An agent-based model of a repeated prisoner's dilemma with an **exit** option. It asks
whether leaving a relationship is the mirror image of entering one.

**Short answer:** it depends on memory.
- **Perfect memory.** Agents with perfect memory give a partner about 2 rounds of
  betrayal for every good round, with no upper limit. They also never leave a partner who
  defects less than half the time.
- **Human-like memory evolves when it can.** When it's available, evolution gives agents
  a ~2× negativity bias and a ~12-round memory. After that, history buys at most ~10
  rounds of grace, and inconsistent partners usually get caught, at the cost of some
  twitchiness.

Read the full writeup in [`WRITEUP.md`](WRITEUP.md).

## Results

- **One equation explains the original phases.** The agents' belief is a damped tally:
  (cooperations − defections) / (1/p + rounds). It gives the 2.37 slope and the 0.72
  cutoff in closed form.
- **Falsification tests overturned three of five claims:**
  - Tolerated betrayal grows linearly only with perfect memory.
  - Exit cost does matter once it's large: tolerance goes 0.36 → 0.64.
  - The blind spot wasn't caused by a lack of exposure to inconsistent partners. The
    tally belief itself can't catch a partner who defects less than half the time.
- **Human-like rebuild:**
  - Negativity bias and fading memory evolve; comparison with alternatives doesn't.
  - Grace saturates at about 10 rounds whatever the history.
  - 52% of agents leave a 30%-bad partner, against 0% before.
  - A bad first partner costs 11% of lifetime payoff, against 22% for tally agents.

## Files

| File | Contents |
| --- | --- |
| `model.py` | The whole model: one `World` config covers the original and human-like worlds, plus closed forms |
| `experiments.py` | Replication, falsification tests and v2 experiments → `results.json`, `genomes.json` |
| `WRITEUP.md` | The writeup |
| `legacy/` | The original Phase 1–6 scripts, results and writeup |

## Running

```sh
pip install numpy
python experiments.py   # ~90 s on 8 cores
```
