"""Predeclared nearby-parameter diagnostics, not a search for replacement winners."""
from datetime import datetime,timezone
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from cross_project_inventory import OUT,save,sha
from expanded_study import load_jobs,prepare,simulate
from run_study import panel_from
from tensortrade_lab.portfolio import Ledger,performance

def main():
    specs=[{'family':'GoldDiversification','gold':g,'frequency':q} for g in [.1,.15,.2,.25,.3] for q in [1,3,12]]
    specs +=[{'family':'TLT_MA','fast':f,'slow':s} for f,s in [(40,160),(50,200),(60,240)]]
    specs +=[{'family':'BTC_Momentum','lookback':n} for n in [168,252,336]]
    blocks=[('full','2017-01-01','2026-09-24'),('2017_2019','2017-01-01','2020-01-01'),
            ('2020_2022','2020-01-01','2023-01-01'),('2023_latest','2023-01-01','2026-09-24')]
    save('lead_protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),'specs':specs,
          'blocks':blocks,'cost_multipliers':[1,2],'decision':'Diagnose sensitivity; no parameter replacement based on these outcomes.'})
    jobs,_=load_jobs();data={}
    for market,frames,_,crypto in jobs:
        if market not in ['MultiAsset','TLT','BTC']:continue
        if market=='MultiAsset':frames={k:frames[k] for k in ['SPY','GLD']}
        p=panel_from(frames,not crypto,365 if crypto else 252);data[market]=(frames,p)
    rows=[]
    for spec in specs:
        family=spec['family'];market={'GoldDiversification':'MultiAsset','TLT_MA':'TLT','BTC_Momentum':'BTC'}[family]
        frames,p=data[market];f=next(iter(frames.values()))
        if family=='TLT_MA':sig=(f.close.rolling(spec['fast']).mean()>f.close.rolling(spec['slow']).mean()).to_numpy(float)
        elif family=='BTC_Momentum':
            tr=pd.Series(p.total_index[:,0]);sig=(tr>tr.shift(spec['lookback'])).to_numpy(float)
        for block,begin,finish in blocks:
            ix=np.where((p.timestamp>=pd.Timestamp(begin,tz='UTC'))&(p.timestamp<pd.Timestamp(finish,tz='UTC')))[0]
            start=max(int(ix[0]),337);end=int(ix[-1])
            for mult in [1,2]:
                fee=(.001 if market=='BTC' else .0005)*mult;slip=.0005*mult
                bm,_,_,_=simulate(p,frames,None,'BuyHold',start,end,fee,slip)
                ledger=Ledger(p,start,end,fee,slip);last=None
                for t in range(start,end+1):
                    j=t-1;w=None
                    if family=='GoldDiversification':
                        now=p.timestamp[t];before=p.timestamp[t-1]
                        q=spec['frequency']
                        if t==start or (now.year*12+now.month-1)//q!=(before.year*12+before.month-1)//q:
                            w=np.array([1-spec['gold'],spec['gold']])
                    elif sig[j]!=last:w=np.array([sig[j]]);last=sig[j]
                    ledger.step(w)
                m=performance(ledger)
                rows.append({'spec':spec,'market':market,'block':block,'cost_multiple':mult,**m,
                    'benchmark_return':bm['return'],'benchmark_cagr':bm['cagr'],'benchmark_drawdown':bm['max_drawdown'],
                    'pass':bool(m['total_return']>bm['return'] and m['max_drawdown']<=bm['max_drawdown'])})
        print('lead diagnostic',spec,flush=True)
    save('lead_sensitivity.json',rows);save('lead_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'specs':len(specs),'runs':len(rows),'script_sha256':sha(__file__),
        'disclosure':'Retrospective neighborhood tests. Original selected rules remain unchanged.'})

if __name__=='__main__':main()
