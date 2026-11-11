import sys,os,json,time
import numpy as np, pandas as pd
_r = os.path.dirname(os.path.abspath(__file__))   # walk up to the repo root (holds rfm_guppy/paths.py)
while not os.path.isfile(os.path.join(_r, "rfm_guppy", "paths.py")):
    if _r == os.path.dirname(_r): raise RuntimeError("rfm_guppy/paths.py not found above " + __file__)
    _r = os.path.dirname(_r)
sys.path.insert(0, _r)
from rfm_guppy.paths import CACHE_DIR, CHECKS_RUN_2026_10_09
from rfm_guppy.rfm import _agop_step,_preprocess
C=CACHE_DIR; CON=['food','valence','env_social']; B,TMAX,LAM,SEED=20,10,1e-3,20261002
Z0=np.load(os.path.join(CHECKS_RUN_2026_10_09,'Ia-shuffle-label-on-blocks_sT.npz'))
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
pd.DataFrame(rows).to_csv(f'{os.path.dirname(__file__)}/kappa.csv',index=False)
print('gate max |s-stored|',gate)
