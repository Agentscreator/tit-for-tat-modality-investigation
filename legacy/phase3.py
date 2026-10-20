import sys, json; 
import numpy as np
from engine import evolve
from core import GENE_BOUNDS, OBS_NOISE_VAR

SEEDS=[1,2,3,4,5]
MAXR=400

def time_to_recognise(genome, good_rounds, max_rounds=MAXR):
    """Partner cooperates `good_rounds`, then defects forever.
       Returns rounds of continuous betrayal needed before agent exits (None = never)."""
    mean, var = 0.0, genome['prior_variance']
    thr = genome['belief_exit_threshold']
    for t in range(max_rounds):
        obs = 1.0 if t < good_rounds else -1.0
        gain = var/(var+OBS_NOISE_VAR)
        mean += gain*(obs-mean)
        var  *= (1-gain)
        if mean < thr and t >= good_rounds:
            return t-good_rounds+1
    return None

# evolve under FREE exit, so no economic reason to stay
print("evolving populations (free exit)...")
pops=[]
for s in SEEDS:
    hist, genomes = evolve(n_gen=100, n_evo=200, n_tft=40, n_defect=80, n_rounds=80,
                           seed=s, exit_cost_mode='free', perception_noise=0.10)
    pops.append(genomes)
    print("  seed", s, "done")

GOOD=[0,1,2,3,5,10,20,40,80,160]
print(f"\n{'good_rounds_first':>18} {'median_rounds_to_exit':>22} {'share_never_exits':>18}")
table={}
for g in GOOD:
    per_seed_med=[]; per_seed_never=[]
    for genomes in pops:
        res=[time_to_recognise(gm,g) for gm in genomes]
        fin=[r for r in res if r is not None]
        per_seed_never.append(100*(1-len(fin)/len(res)))
        per_seed_med.append(np.median(fin) if fin else np.nan)
    med=np.nanmean(per_seed_med); nev=np.mean(per_seed_never)
    table[g]={'median':float(med) if not np.isnan(med) else None,'never_pct':float(nev)}
    ms = f"{med:.1f}" if not np.isnan(med) else "n/a"
    print(f"{g:>18} {ms:>22} {nev:>17.1f}%")

# Study 6 comparison: stranger (wide prior) vs ex-partner (converged prior)
print("\nStudy 6 replication inside the evolved population:")
rows=[]
for genomes in pops:
    stranger=[time_to_recognise(gm,0) for gm in genomes]
    ex=[time_to_recognise(gm,20) for gm in genomes]
    sf=[r for r in stranger if r]; ef=[r for r in ex if r]
    if sf and ef: rows.append(np.median(ef)/np.median(sf))
print(f"  ratio (20 good rounds vs stranger): {np.mean(rows):.1f}x  ± {1.96*np.std(rows,ddof=1)/np.sqrt(len(rows)):.1f}")
json.dump(table, open('phase3.json','w'), indent=2)
