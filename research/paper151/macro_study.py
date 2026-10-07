"""Paper 19.3 inflation allocation using dated ALFRED vintages and ETF prices."""
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import replace
import json, zipfile
import numpy as np
import pandas as pd
from full_catalogue_options import save, sha
from run_study import read_frame, panel_from
from tensortrade_lab.portfolio import Ledger, performance

HERE=Path(__file__).resolve().parent
OUT=HERE/'data_expansion'

def vintages(series):
    with zipfile.ZipFile(OUT/(series+'_vintages.zip')) as z:
        f=pd.read_csv(z.open('obs._by_real-time_period.csv'))
    for c in ['period_start_date','realtime_start_date']:
        f[c]=pd.to_datetime(f[c],utc=True)
    f['available_at']=f.realtime_start_date+pd.Timedelta(days=1)
    f=f.rename(columns={series:'value'})
    if f.duplicated(['period_start_date','available_at']).any():raise ValueError('Duplicate vintage')
    return f

def known_at(f,at):
    # End of the vintage validity interval can reveal future revisions. Ignore it.
    return f[f.available_at<=at].sort_values('available_at').drop_duplicates('period_start_date',keep='last').set_index('period_start_date')

def allocation(headline,core,at):
    a,b=known_at(headline,at),known_at(core,at)
    periods=a.index.intersection(b.index)
    if not len(periods):return None,{'reason':'no common known period'}
    period=periods.max();prior=period-pd.DateOffset(years=1)
    if prior not in a.index or prior not in b.index:return None,{'reason':'missing prior-year observation'}
    if at-period>pd.Timedelta(days=90):return None,{'reason':'observation older than90days'}
    if any(at-f.loc[period,'available_at']>pd.Timedelta(days=75) for f in [a,b]):return None,{'reason':'release older than75days'}
    h=float(a.loc[period,'value']/a.loc[prior,'value']-1)
    c=float(b.loc[period,'value']/b.loc[prior,'value']-1)
    weight=float(np.clip((h-c)/h,0,1)) if abs(h)>1e-12 else 0.
    return weight,{'reason':None,'period':str(period),'headline_yoy':h,'core_yoy':c,
        'vintages_used':[{ 'series':s,'period':str(q),'available_at':str(f.loc[q,'available_at']),'value':float(f.loc[q,'value'])} for s,f in [('headline',a),('core',b)] for q in [period,prior]]}

def main():
    proto=OUT/'macro_protocol.json'
    if not proto.exists():save(proto,{'registered_at':datetime.now(timezone.utc).isoformat(),'section':'19.3',
      'formula':'Commodity allocation=max(0,min((headlineYoY-coreYoY)/headlineYoY,1)); abs(headlineYoY)<=1e-12 =>0. Formula includes deflation without parameter changes.',
      'data':'ALFRED CPIAUCSL and CPILFESL, all vintages2015-Aug2026. At decision use latest known version of latest common observation and its year-earlier value. Dated vintage+1calendar day availability; no lookahead using realtime_end_date. Seasonal-adjusted YoY adapter, not exact unadjusted CPI replication.',
      'execution':'Commodity ETF DBC plus SPY remainder; rebalance monthly after first observed session close, next open. Missing/stale data =>all cash. $10k,5bpcommission+5bpslippage, explicit dividends and prior-known cash yield. No tax or fund-holdings replication.',
      'missing':'Need current and12months-prior observations in bothseries, latest observation<=90days old and release<=75days old. No filling unavailable values.',
      'evaluation':'2017-latest full and2017-19,2020-22,2023-latest;5bp+5bpbase,doublecost,extra-bar-delay,nointerest. Match fully invested SPY. Retrospective fixed adapter;noqualification.',
      'source_sha256':{s:sha(OUT/(s+'_vintages.zip')) for s in ['CPIAUCSL','CPILFESL']}})
    frames={}
    for s,path in [('SPY','research/paper151/full_catalogue/data/SPY.parquet'),('DBC','research/paper151/data_expansion/DBC.parquet')]:
        f,_=read_frame(path);frames[s]=f
    start=max(f.timestamp.min() for f in frames.values());end=min(f.timestamp.max() for f in frames.values())
    frames={s:f[f.timestamp.between(start,end)].reset_index(drop=True) for s,f in frames.items()}
    p=panel_from(frames);a,b=vintages('CPIAUCSL'),vintages('CPILFESL')
    targets=np.full((len(p.timestamp),2),np.nan);decisions=[]
    for j,at in enumerate(p.bar_end):
        if j and p.timestamp[j].month==p.timestamp[j-1].month:continue
        w,audit=allocation(a,b,at);targets[j]=[1-w,w] if w is not None else [0,0]
        decisions.append({'decision_at':str(at),'target':targets[j].tolist(),**audit})
    save(OUT/'macro_decisions.json',decisions)
    rows=[]
    for block,left,right in [('full','2017-01-01','2026-09-24'),('2017_2019','2017-01-01','2020-01-01'),('2020_2022','2020-01-01','2023-01-01'),('2023_latest','2023-01-01','2026-09-24')]:
        ix=np.flatnonzero((p.timestamp>=pd.Timestamp(left,tz='UTC'))&(p.timestamp<pd.Timestamp(right,tz='UTC')));lo,hi=int(ix[0]),int(ix[-1])
        for case,cost,delay,zero in [('base',1,1,False),('double_cost',2,1,False),('extra_bar_delay',1,2,False),('no_interest',1,1,True)]:
            pp=replace(p,cash_rate=np.zeros(len(p.timestamp))) if zero else p
            led=Ledger(pp,lo,hi,commission=.0005*cost,slippage=.0005*cost)
            bh=Ledger(pp,lo,hi,commission=.0005*cost,slippage=.0005*cost)
            for t in ix:
                j=t-delay;w=targets[j] if np.isfinite(targets[j]).all() else None
                if t==lo and w is None:
                    known=np.flatnonzero(np.isfinite(targets[:j+1]).all(axis=1));w=targets[known[-1]] if len(known) else np.zeros(2)
                led.step(w);bh.step(np.array([1.,0.]) if t==lo else None,drip=True)
            m,bm=performance(led),performance(bh)
            rows.append({'strategy':'InflationAllocation','section':'19.3','block':block,'case':case,**m,'benchmark_return':bm['total_return'],'benchmark_drawdown':bm['max_drawdown'],'pass':bool(m['total_return']>bm['total_return'] and m['max_drawdown']<=bm['max_drawdown'])})
            if block=='full':
                for name,book in [('InflationAllocation',led),('InflationBuyHold',bh)]:
                    pd.DataFrame(book.history).to_parquet(OUT/(name+'_'+case+'.parquet'),index=False);save(OUT/(name+'_'+case+'_fills.json'),book.fills)
    save(OUT/'macro_results.json',rows)
    save(OUT/'macro_manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'candidate_cases':len(rows),'code_sha256':sha(__file__),'protocol_sha256':sha(proto),'vintage_records':len(a)+len(b),'all_gates_pass':all(r['pass'] for r in rows),'execution_qualified':False,'prices_sha256':{s:sha(HERE/'full_catalogue/data/SPY.parquet' if s=='SPY' else OUT/'DBC.parquet') for s in frames}})
    print('macro complete',len(rows),flush=True)

if __name__=='__main__':main()
