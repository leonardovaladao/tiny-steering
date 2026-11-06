"""Collect the numbers charted by the Checks I/Ia page into data.json. Run from report_page_checks_I_Ia/."""
import json, numpy as np, pandas as pd
R='../run_20261009/'
csv=lambda f: pd.read_csv(R+f+'.csv')
def recs(df,n=4): return json.loads(df.round(n).to_json(orient='records'))
D={}
A=csv('I-shuffled-label-control_regimeA'); B=csv('I-shuffled-label-control_regimeB')
na=np.load(R+'I-shuffled-label-control_results.npz'); nb=np.load(R+'I-shuffled-label-control_resultsB.npz')
A['lo']=[na[f'{r.c}_{r.b}_s'].min() for r in A.itertuples()]; A['hi']=[na[f'{r.c}_{r.b}_s'].max() for r in A.itertuples()]
B['lo']=[nb[f'{r.c}_{r.b}_s'].min() for r in B.itertuples()]; B['hi']=[nb[f'{r.c}_{r.b}_s'].max() for r in B.itertuples()]
D['IA']=recs(A[['c','b','L','T','nz','s','med','lo','hi','f95','p','rho','rho_sh','cos','refuted']])
D['IB']=recs(B[['c','b','s','med','lo','hi','f95','p','rho','rho_sh','rho_mx','refuted']])
D['Irho']=recs(csv('I-shuffled-label-control_rho_baselines'))
z=np.load(R+'Ia-shuffle-label-on-blocks_sT.npz'); cur={}
for k in z.keys():
    a=z[k]; s=a[1:]
    cur[k]=dict(real=np.round(a[0],4).tolist(),med=np.round(np.median(s,0),4).tolist(),lo=np.round(s.min(0),4).tolist(),hi=np.round(s.max(0),4).tolist())
D['Iacur']=cur
D['Iacell']=recs(csv('Ia-shuffle-label-on-blocks_per_cell'))
D['Iastored']=recs(csv('Ia-shuffle-label-on-blocks_stored_hp'))
# Aggregate over the 108 cells, per T: cell-level median across permutations, then median and q05-q95 across cells
X=np.stack([z[k] for k in z.keys()]); real=X[:,0]; med=np.median(X[:,1:],1)
q=lambda a,p: np.round(np.quantile(a,p,axis=0),4).tolist()
D['Iaagg']=dict(real=q(real,.5),real_lo=q(real,.05),real_hi=q(real,.95),shuf=q(med,.5),shuf_lo=q(med,.05),shuf_hi=q(med,.95),gap=q(real-med,.5))
json.dump(D,open('data.json','w'),separators=(',',':'))
print({k:v for k,v in D['Iaagg'].items() if k in('real','shuf','gap')})
print('cos range',A.cos.min(),A.cos.max(),'f95 max',A.f95.max(),B.f95.max())
