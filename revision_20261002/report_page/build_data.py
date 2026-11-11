"""Collect every number the page charts from run_20261009/ into data.json. Run from report_page/."""
import json, os, sys, numpy as np, pandas as pd
_r = os.path.dirname(os.path.abspath(__file__))   # walk up to the repo root (holds rfm_guppy/paths.py)
while not os.path.isfile(os.path.join(_r, "rfm_guppy", "paths.py")):
    if _r == os.path.dirname(_r): raise RuntimeError("rfm_guppy/paths.py not found above " + __file__)
    _r = os.path.dirname(_r)
sys.path.insert(0, _r)
from rfm_guppy.paths import CHECKS_RUN_2026_10_09
R=os.path.join(CHECKS_RUN_2026_10_09,'')
C=['food','valence','env_social']
csv=lambda f: pd.read_csv(R+f+'.csv')
def recs(df,n=4): return json.loads(df.round(n).to_json(orient='records'))
D={}
# Check I
A=csv('I-shuffled-label-control_regimeA'); B=csv('I-shuffled-label-control_regimeB')
na=np.load(R+'I-shuffled-label-control_results.npz'); nb=np.load(R+'I-shuffled-label-control_resultsB.npz')
A['lo']=[na[f'{r.c}_{r.b}_s'].min() for r in A.itertuples()]
A['hi']=[na[f'{r.c}_{r.b}_s'].max() for r in A.itertuples()]
B['lo']=[nb[f'{r.c}_{r.b}_s'].min() for r in B.itertuples()]
B['hi']=[nb[f'{r.c}_{r.b}_s'].max() for r in B.itertuples()]
D['IA']=recs(A[['c','b','L','T','nz','s','med','lo','hi','f95','p','rho','rho_sh','cos','refuted']])
D['IB']=recs(B[['c','b','s','med','lo','hi','f95','p','rho','rho_sh','rho_mx','refuted']])
D['Irho']=recs(csv('I-shuffled-label-control_rho_baselines'))
# Check Ia
z=np.load(R+'Ia-shuffle-label-on-blocks_sT.npz'); cur={}
for k in z.keys():
    a=z[k]; s=a[1:]
    cur[k]=dict(real=np.round(a[0],4).tolist(),med=np.round(np.median(s,0),4).tolist(),lo=np.round(s.min(0),4).tolist(),hi=np.round(s.max(0),4).tolist())
D['Iacur']=cur
D['Iacell']=recs(csv('Ia-shuffle-label-on-blocks_per_cell'))
D['Iastored']=recs(csv('Ia-shuffle-label-on-blocks_stored_hp'))
# Check II
D['IIblock']=recs(csv('II-PCA-baseline_perblock'))
D['IIhead']=recs(csv('II-PCA-baseline_headline'))
D['IIboot']=recs(csv('II-PCA-baseline_bootstrap'))
D['IIverd']=recs(csv('II-PCA-baseline_verdict'))
# Check III
g=csv('III-steering-metric_generations'); cal=csv('III-steering-metric_calibration')
D['IIIgen']=recs(g[['concept','r','i','g','probe','lex','ppl','d2','coherent','on','old_success']],3)
D['IIIr']=recs(csv('III-steering-metric_per_r'),3)
D['IIIh1']=recs(csv('III-steering-metric_H1'),3)
D['IIIh3']=recs(csv('III-steering-metric_H3'),3)
D['IIIdup']=recs(csv('III-steering-metric_duplication'),3)
D['IIIjudge']=recs(csv('III-steering-metric_judge_validity'),3)
D['IIIold']=recs(csv('III-steering-metric_old_vs_new'),3)
D['pplmax']=float(cal.ppl.quantile(.95)); D['d2min']=float(cal.d2.quantile(.05))
ex=[]
for c in C:
    for r in [0,.25,1,-1]:
        t=g[(g.concept==c)&(g.r==r)].iloc[0]
        ex.append(dict(concept=c,r=r,text=t.text[:92]+('…' if len(t.text)>92 else ''),g=round(t.g,2),probe=round(t.probe,2),ppl=float(t.ppl)))
D['IIIex']=ex
# Check IV
s=csv('IV-small-n_summary'); D['IVsum']=recs(s[['block','concept','n','contrast','metric','delta','ci_lo','ci_hi','holm_declared']],5)
d=csv('IV-small-n_draws')
m=d.groupby(['concept','block','n','method'])[['auc','rho']].mean().reset_index(); D['IVmean']=recs(m,4)
t=csv('IV-small-n_T_sensitivity'); D['IVT']=recs(t[['block','concept','n','T','delta','ci_hi','is_T_frozen','is_T_star']],4)
D['IVhp']=recs(csv('IV-small-n_hp')[['concept','block','L','T','val_rho_rfm','C','val_rho_logreg']],3)
D['IVverd']=recs(csv('IV-small-n_verdicts')[['block','hypothesis','verdict']])
json.dump(D,open('data.json','w'),separators=(',',':'))
# ---- verifications printed for the record
print('size KB',round(len(json.dumps(D))/1024))
n1=d[d.n==1]; w=n1.pivot_table(index=['concept','block','r'],columns='method',values='auc')
print('C0 auc spread n=1',(w.max(axis=1)-w.min(axis=1)).max(),'max resid (all n)',d.resid.max(),'dup check draws per cell',d[(d.n==2)].groupby(['concept','block','method']).size().unique())
dup=csv('III-steering-metric_duplication'); h=dup[dup.r.abs()>=.5]; print('jaccard |r|>=.5',h.mean_jaccard.min(),h.mean_jaccard.max(),'n_distinct',h.n_distinct.min(),h.n_distinct.max())
a=A; print('T=1 cells med',a[a['T']==1].med.tolist(),'T>1 med',a[a['T']>1].med.min(),a[a['T']>1].med.max())
print('regimeB rho_sh',B.groupby('c').rho_sh.agg(['min','max']).round(3).to_dict('index'))
print('regimeA rho_sh',A.groupby('c').rho_sh.agg(['min','max']).round(3).to_dict('index'))
ss=s[(s.metric=='auc')]
print(ss[(ss.contrast=='RFM-LogReg')&(ss.concept=='valence')&(ss.block==1)&(ss.n==32)][['delta','ci_lo','ci_hi']].round(4).to_dict('records'))
print(ss[(ss.contrast=='RFM-DM')&(ss.block==3)&(ss.n==32)][['concept','delta','ci_lo','ci_hi']].round(6).to_dict('records'))
c=np.load(R+'I-shuffled-label-control_resultsC.npz'); print('C keys',[k for k in c.keys()][:3])
ok=[]
for k in c.keys():
    if k.endswith('_real'):
        cc,b,sig=k[:-5].rsplit('_',2)
        best=min(np.abs(c[k]-z[f'{cc}|{b}|{nz}|{sig}'][0]).max() for nz in (0,1)); ok.append(best)
print('Regime C vs Ia real curves max diff',max(ok),'n',len(ok))
