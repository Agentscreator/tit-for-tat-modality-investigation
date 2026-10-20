import sys, json, warnings
sys.path.insert(0, '/home/claude/study7')
import numpy as np
from engine import evolve, run_generation
from core import random_genome, GENE_BOUNDS, OBS_NOISE_VAR
warnings.filterwarnings('ignore')

SEEDS = [1,2,3,4,5]
def ci(v):
    v = np.array(v, float)
    return float(v.mean()), float(1.96*v.std(ddof=1)/np.sqrt(len(v)))

R = {}

# ---------------- PHASE 1: tit-for-tat's failure mode ----------------
print("PHASE 1  benchmark: mean vs tail")
acc = {k: [] for k in ['evo_mean','tft_mean','def_mean','evo_p10','tft_p10','stuck_share','stuck_len']}
for s in SEEDS:
    rng = np.random.default_rng(s)
    genomes = [random_genome(rng) for _ in range(200)]
    evo, d = run_generation(genomes, 40, 80, 80, seed=s,
                            exit_cost_mode='entangled', perception_noise=0.10)
    ev = [a.payoff for a in evo]
    acc['evo_mean'].append(np.mean(ev)); acc['evo_p10'].append(np.percentile(ev,10))
    acc['tft_mean'].append(np.mean(d['tft_payoffs'])); acc['tft_p10'].append(np.percentile(d['tft_payoffs'],10))
    acc['def_mean'].append(np.mean(d['def_payoffs']))
    st = np.array(d['stuck'] if 'stuck' in d else d['tft_stuck'])
    acc['stuck_share'].append(100*np.mean(st>0))
    acc['stuck_len'].append(float(st[st>0].mean()) if (st>0).any() else 0.0)
R['phase1'] = {k: ci(v) for k,v in acc.items()}
for k,(m,e) in R['phase1'].items(): print(f"   {k:12s} {m:8.1f} ± {e:.1f}")

# ---------------- PHASE 2: does exit cost change the optimum? ----------------
print("\nPHASE 2  evolution under exit-cost regimes")
R['phase2'] = {}
for mode in ['free','flat','entangled']:
    finals=[]
    for s in SEEDS:
        hist,_ = evolve(n_gen=100, n_evo=200, n_tft=40, n_defect=80, n_rounds=80,
                        seed=s, exit_cost_mode=mode, perception_noise=0.10)
        tail = hist[-10:]
        row = {k: float(np.mean([h['genome'][k] for h in tail])) for k in GENE_BOUNDS}
        row['evo_mean'] = float(np.mean([h['evo_mean'] for h in tail]))
        row['tft_mean'] = float(np.mean([h['tft_mean'] for h in tail]))
        finals.append(row)
    R['phase2'][mode] = {k: ci([f[k] for f in finals]) for k in finals[0]}
    print(f"  --- {mode} ---")
    for k,(m,e) in R['phase2'][mode].items(): print(f"    {k:22s} {m:8.2f} ± {e:.2f}")

json.dump(R, open('/home/claude/study7/results.json','w'),
          indent=2, default=lambda o: list(o))
print("\nsaved")
