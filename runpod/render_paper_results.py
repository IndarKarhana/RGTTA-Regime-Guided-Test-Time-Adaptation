"""Render manuscript tables/figure directly from validated CSVs."""
from pathlib import Path
import json
import os
import tempfile
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'rgtta-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'benchmarks/results/final_validated';O=ROOT/'paper/generated'
N={'tta':'TTA','ewc':'EWC','dynatta':'Dyna-style','rgtta':'RG-TTA','rgtta_ewc':'RG-EWC',
   'rgtta_dynatta':'RG-Dyna','retrain':'Retrain','itransformer':'iTransformer','patchtst':'PatchTST','dlinear':'DLinear','gru_small':'GRU'}

def table(name, cols, headers, rows):
    text='\\begin{tabular}{'+cols+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (O/(name+'.tex')).write_text(text)

def pm(mean,sd,precision=2):return f'${mean:.{precision}f}\\pm{sd:.{precision}f}$'

def main():
    assert json.loads((R/'validation.json').read_text())['validated']
    O.mkdir(exist_ok=True)
    m=pd.read_csv(R/'main_tables.csv'); x=m.pivot(index=['model','dataset','horizon'],columns='policy',values='mse_mean')
    t=m.pivot(index=['model','dataset','horizon'],columns='policy',values='seconds_mean')
    rows=[]
    for p in ['tta','ewc','dynatta','retrain','rgtta','rgtta_ewc','rgtta_dynatta']:
        row=[N[p]]
        for synth in [False,True]:
            take=x.index.get_level_values('dataset').str.startswith('synth')==synth
            ratios=x.loc[take,p]/x.loc[take,'tta'];times=t.loc[take,p]/t.loc[take,'tta']
            row += [f'{np.exp(np.log(ratios).mean()):.3f}',f'{(ratios<1).sum()}/{len(ratios)}' if p!='tta' else '--',f'{np.exp(np.log(times).mean()):.2f}']
        rows.append(row)
    table('main_summary','lrrrrrr',['Policy','MSE/TTA','Wins','Time/TTA','MSE/TTA','Wins','Time/TTA'],rows)
    paired=pd.read_csv(R/'aggregate.csv');rows=[]
    for _,r in paired[paired.scope=='all'].iterrows():
        rows.append([N[r.policy],f'{r.geometric_mse_ratio:.3f}',f'{int(r.wins)}/224',f'{r.geometric_time_ratio:.2f}'])
    table('matched_summary','lrrr',['Controller','MSE/base','Wins','Time/base'],rows)
    s=pd.read_csv(R/'reviewer_tables.csv');rows=[]
    variants={'small_head':'Compact/head','large_head':'Larger/head','large_adapter':'Larger/adapter+head','large_full':'Larger/full'}
    for model in ['itransformer','patchtst']:
        for v in variants:
            sel=s[(s.study=='scale')&(s.model==model)&(s.variant==v)].set_index('policy');a=sel.loc['tta'];b=sel.loc['rgtta']
            rows.append([N[model],variants[v],f'{int(b.parameters_mean):,}',f'{int(b.adapted_mean):,}',
                         pm(a.mse_mean,a.mse_sd),pm(b.mse_mean,b.mse_sd),f'{a.seconds_mean:.2f}/{b.seconds_mean:.2f}',f'{b.peak_mib_mean:.0f}'])
    table('scale','llrrrrrr',['Backbone','Update scope','Total','Adapted','TTA MSE','RG MSE','Time T/R (s)','Peak MiB'],rows)
    rows=[]
    for variant in ['target','marginal','dependence']:
        row=[{'target':'Target 5D','marginal':'Marginal','dependence':'Dependence'}[variant]]
        for ds in ['ETTh1','Weather','CovShift']:
            q=s[(s.variant==variant)&(s.dataset==ds)&(s.policy=='rgtta')].iloc[0]
            row.append(pm(q.mse_mean,q.mse_sd,3))
        rows.append(row)
    table('features','lrrr',['Signature','ETTh1','Weather','CovShift'],rows)
    rows=[]
    for var in ['frozen','full','no_reuse','no_loss_gate','no_regime_lr']:
        row=[{'frozen':'Frozen','full':'Full RG-TTA','no_reuse':'No retrieval','no_loss_gate':'No loss gate','no_regime_lr':'Constant RG LR'}[var]]
        for ds in ['ETTh1','Weather']:
            sel=s[(s.dataset==ds)&(s.policy=='rgtta')]
            q=sel[(sel.study=='features')&(sel.variant=='marginal')].iloc[0] if var=='full' else sel[(sel.study=='ablation')&(sel.variant==var)].iloc[0]
            row.append(pm(q.mse_mean,q.mse_sd))
        rows.append(row)
    table('ablations','lrr',['Variant','ETTh1 MSE','Weather MSE'],rows)
    rows=[]
    for variant in ['current','fast_ema','paper_warmup']:
        row=[{'current':r'$\eta=.1, w=60$','fast_ema':r'$\eta=.7, w=60$','paper_warmup':r'$\eta=.1, w=96$'}[variant]]
        for ds in ['ETTh1','Weather']:
            q=s[(s.study=='dynatta_sensitivity')&(s.variant==variant)&(s.dataset==ds)&(s.policy=='dynatta')].iloc[0]
            row.append(pm(q.mse_mean,q.mse_sd))
        rows.append(row)
    table('sensitivity','lrr',['Local baseline setting','ETTh1','Weather'],rows)
    off=pd.read_csv(R/'official_tables.csv');rows=[]
    for ds in ['ETTh1','ETTh2']:
        for p in ['frozen','tafas','dynatta']:
            q=off[(off.dataset==ds)&(off.policy==p)].iloc[0]
            rows.append([ds,{'frozen':'Frozen','tafas':'TAFAS','dynatta':'DynaTTA'}[p],pm(q.mse_mean,q.mse_sd,4),pm(q.mae_mean,q.mae_sd,4),f'{q.seconds_mean:.2f}'])
    table('official','llrrr',['Data','Policy','MSE','MAE','Eval. (s)'],rows)
    pairs=pd.read_csv(R/'paired_comparisons.csv');rg=pairs[pairs.policy=='rgtta']
    grid=rg.assign(logratio=np.log(rg.mse_ratio)).groupby(['model','dataset']).logratio.mean().unstack().apply(np.exp)
    models=['itransformer','patchtst','dlinear','gru_small'];datasets=['ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange']+[d for d in grid if d.startswith('synth_')]
    grid=grid.loc[models,datasets]
    fig,ax=plt.subplots(figsize=(10,2.7));im=ax.imshow(np.log2(grid),cmap='RdBu_r',vmin=-.5,vmax=.5,aspect='auto')
    for i in range(len(models)):
        for j in range(len(datasets)):ax.text(j,i,f'{grid.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
    ax.set_yticks(range(4),[N[m] for m in models]);ax.set_xticks(range(len(datasets)),[d.replace('synth_','').replace('_',' ') for d in datasets],rotation=45,ha='right',fontsize=8)
    fig.colorbar(im,ax=ax,label='log2(MSE ratio)',fraction=.025,pad=.02);fig.tight_layout();fig.savefig(O/'rgtta_ratios.pdf');fig.savefig(O/'rgtta_ratios.png',dpi=160);plt.close(fig)
    print('Generated 7 tables and ratio figure in',O)

if __name__=='__main__':main()
