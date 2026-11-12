"""κ side run: AGOP eigenvalue trajectory on the stored cells (README, caveat c).

For each concept of the check run and each block, at the stored σ and normalize, recompute M_t for
t = 1..TMAX for the real labels and B permutations (seed SEED, drawn per concept in the same order
as Check Ia), and record λ1, λ2, λ3, the trace, s = λ1/tr and the top-2, top-3 and top-5 shares.
Gate: the largest |s - stored s| against Check Ia's `Ia-shuffle-label-on-blocks_sT.npz` in
--source-run (read only), printed at the end. Every concept must be present there.

Writes `kappa.csv` into --run, a check run made by `python -m rfm_checks.checkrun new` (its
concepts come from run.json); appends one step to its run.json. Refuses an existing kappa.csv
unless RFM_FORCE=1, and refuses pinned runs (paths.PINNED_RUNS) in any case.

Usage (from the repo root):
    .venv/bin/python rfm_checks/kappa_side_run.py --run DIR [--source-run DIR]
"""
import argparse
import sys,os,json,time
import numpy as np, pandas as pd
_r = os.path.dirname(os.path.abspath(__file__))   # walk up to the repo root (holds rfm_guppy/paths.py)
while not os.path.isfile(os.path.join(_r, "rfm_guppy", "paths.py")):
    if _r == os.path.dirname(_r): raise RuntimeError("rfm_guppy/paths.py not found above " + __file__)
    _r = os.path.dirname(_r)
sys.path.insert(0, _r)
from rfm_guppy.paths import CACHE_DIR, CHECKS_RUN_2026_10_09
from rfm_guppy.rfm import _agop_step,_preprocess
from rfm_guppy.runs import display_path
from rfm_checks.checkrun import KAPPA, load_check_run, out_file, read_check_run

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--run", required=True, metavar="DIR",
                help="check run to write kappa.csv into (made by `python -m rfm_checks.checkrun new`)")
ap.add_argument("--source-run", default=CHECKS_RUN_2026_10_09, metavar="DIR",
                help="run whose Ia-shuffle-label-on-blocks_sT.npz is the gate (read only; "
                     "default rfm_checks/runs/2026-10-09)")
args = ap.parse_args()

C=CACHE_DIR; B,TMAX,LAM,SEED=20,10,1e-3,20261002
SRC=os.path.join(os.path.abspath(args.source_run),'Ia-shuffle-label-on-blocks_sT.npz')
if not os.path.isfile(SRC): sys.exit(f"error: {display_path(SRC)} not found (--source-run)")
Z0=np.load(SRC)
try: _, cfg = read_check_run(args.run)            # checks the run folder; writes nothing
except (PermissionError, FileNotFoundError, ValueError) as e: sys.exit(f"error: {e}")
miss=[c for c in cfg['concepts'] if not any(k.startswith(f'{c}|') for k in Z0.files)]
if miss: sys.exit(f"error: {display_path(SRC)} has no cells for {miss}; pass --source-run with a "
                  "run whose Check Ia covered these concepts, in the same concept order")
run_dir, cfg = load_check_run(KAPPA, params=dict(B=B, TMAX=TMAX, LAM=LAM, SEED=SEED,
                              source_run=display_path(os.path.abspath(args.source_run))),
                              run_dir=args.run, script=__file__)
CON=cfg['concepts']
OUT=out_file(os.path.join(run_dir,'kappa.csv'))
rng=np.random.default_rng(SEED); rows=[]; gate=0; t0=time.time()
for c in CON:
    X=np.load(f'{C}/{c}_train_activations.npy').astype(float)
    Y=np.array([d['label'] for d in json.load(open(f'{C}/concept_{c}.json'))['train']],float)
    V=dict(np.load(f'{C}/{c}_vectors.npz'))
    perms=[rng.permutation(len(Y)) for _ in range(B)]
    for l in range(6):
        L,Ts,nz=int(V['hp_L'][l]),int(V['hp_T'][l]),int(V['hp_normalize'][l])
        Zc,_=_preprocess(X[:,l,:],center=True,normalize=bool(nz))
        for b in range(B+1):
            yy=Y if b==0 else Y[perms[b-1]]
            M=np.eye(384)
            for t in range(1,TMAX+1):
                M,_,_=_agop_step(Zc,yy,M,L,LAM)
                ev=np.sort(np.linalg.eigvalsh(M))[::-1]; ev=np.maximum(ev,0); tr=ev.sum()
                s=ev[0]/tr
                gate=max(gate,abs(s-Z0[f'{c}|{l}|{nz}|{L}'][b,t-1]))
                rows.append(dict(c=c,l=l,L=L,nz=nz,Ts=Ts,b=b,t=t,s=s,l1=ev[0],l2=ev[1],l3=ev[2],tr=tr,top2=ev[:2].sum()/tr,top3=ev[:3].sum()/tr,top5=ev[:5].sum()/tr))
    print(c,round(time.time()-t0),flush=True)
pd.DataFrame(rows).to_csv(OUT,index=False)
print('wrote',display_path(OUT),len(rows),'rows')
print('gate max |s-stored|',gate)
