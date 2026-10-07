"""Coverage extension: add the five saved daily PF futures books to their family.

Registered separately after the first ensemble outputs; no prior run overwritten.
This tests omitted compatible clocks, not a selection based on their profits.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
import pandas as pd
from full_catalogue_options import save,sha
from ensemble_intraday import paper_daily,read_shadow,execute
HERE=Path(__file__).resolve().parent;OUT=HERE/'ensemble/all_clocks'

def causal_daily_state(intraday,daily,weights):
    right=pd.DataFrame({'known_at':pd.DatetimeIndex(daily.bar_end),'weight':weights})
    left=pd.DataFrame({'known_at':pd.DatetimeIndex(intraday.bar_end)})
    z=pd.merge_asof(left,right,on='known_at',direction='backward',tolerance=pd.Timedelta(days=4))
    if z.weight.isna().any():raise ValueError('Missing or stale daily shadow observation')
    return z.weight.to_numpy()

def main(root):
    folder=OUT/root;folder.mkdir(parents=True,exist_ok=True);proto=OUT/'protocol.json'
    if not proto.exists():save(proto,{'registered_at':datetime.now(timezone.utc).isoformat(),
      'scope':'Coverage extension after first ensemble outputs: add ALL five previously tested daily PF futures configurations (fourES,oneCL), irrespective of returns. GC has no additional dailyPF book. Earlier fixed ensembles stay immutable.',
      'method':'EqualFamiliesAllClocks. Average saved daily and15m PFcloseexposures within ONE PFfamily; same paperfamilies andESquantfamily as first futures blend. Daily state usable only after declared completed UTCday; backwardjoin max4calendar days. Every member gets equal weight within PF, every family gets equal vote. No newparametersearch.',
      'execution':'Same actualcontract integeraccount, dailyfirstobservedUTCbar, prior15mclose,95%targetcap,$1m,zero yield,$2.50+1tick;3tick andextra15mbarlag stresses. Full2023-Jun2026 andeachyear; same95%grosspassivefuturebenchmark.',
      'qualified':False})
    f=pd.read_parquet(HERE/'next_batch'/f'{root}_execution_15m.parquet');f=f[f.timestamp>=pd.Timestamp('2021-06-01',tz='UTC')].reset_index(drop=True)
    t,names=paper_daily(f);sources=json.loads((HERE/f'ensemble/intraday/{root}/sources.json').read_text());groups={};audit=list(sources['shadow_books'])
    for row in audit:
        w,_=read_shadow(f,row['name'],root);groups.setdefault(row['family'],[]).append(w)
    configs=[r for r in json.loads((HERE/'extension/patternfindr_configs.json').read_text()) if r.get('config',{}).get('ticker')==root+'=F' and r['config'].get('interval')=='1d' and (HERE/'next_batch'/(r['id']+'_equity.parquet')).exists()]
    if not configs:raise ValueError('No new compatible clock')
    feed=HERE/'next_batch'/f'{root}_execution_daily.parquet';daily=pd.read_parquet(feed)
    for r in configs:
        h=pd.read_parquet(HERE/'next_batch'/(r['id']+'_equity.parquet'));d=daily[daily.bar_end>=h.timestamp.min()].reset_index(drop=True)
        w,a=read_shadow(d,r['id'],root);groups['PatternFindr'].append(causal_daily_state(f,d,w));audit.append(dict(a,family='PatternFindr',clock='daily'))
    for family,values in groups.items():t=np.column_stack([t,np.mean(values,axis=0)]);names.append(family)
    stream=t.mean(axis=1);rows=[];first=int(np.flatnonzero(f.timestamp>=pd.Timestamp('2023-01-01',tz='UTC'))[0])
    periods=[('full',first,len(f)-1)]+[(str(y),int(ix[0]),int(ix[-1])) for y in range(2023,2027) if len(ix:=np.flatnonzero(f.timestamp.dt.year==y))>1]
    for block,a,b in periods:
        for case,ticks,delay in [('base',1,1),('three_ticks',3,1),('extra_bar_delay',1,2)]:
            bm,_,_=execute(f,None,a,b,root,ticks,benchmark=True);m,eq,fills=execute(f,stream,a,b,root,ticks,delay)
            rows.append({'root':root,'method':'EqualFamiliesAllClocks','block':block,'case':case,**m,'benchmark_return':bm['total_return'],'benchmark_drawdown':bm['max_drawdown'],'pass':bool(m['total_return']>bm['total_return'] and m['max_drawdown']<=bm['max_drawdown'])})
            if block=='full':pd.DataFrame({'timestamp':f.bar_end.iloc[a:b+1],'equity':eq}).to_parquet(folder/(case+'.parquet'),index=False);save(folder/(case+'_fills.json'),fills)
    save(folder/'results.json',rows);save(folder/'sources.json',{'intraday_feed_sha256':sources['feed_sha256'],'daily_feed_sha256':sha(feed),'shadow_books':audit})
    save(folder/'manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),'execution_engine_sha256':sha(HERE/'ensemble_intraday.py'),'protocol_sha256':sha(proto),'source_sha256':sha(folder/'sources.json'),'additional_daily_books':len(configs),'candidate_cases':len(rows),'all_gates_pass':all(r['pass'] for r in rows),'execution_qualified':False})
    print(root,'all clocks complete',len(rows),flush=True)

if __name__=='__main__':main(sys.argv[1])
