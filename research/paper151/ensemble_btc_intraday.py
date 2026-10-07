"""Daily target blends of all saved BTC15m PF/quant books, re-executed on spot."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from full_catalogue_options import save,sha
from ensemble_intraday import paper_daily,fit_weights,METHODS
from run_study import panel_from
from signed_study import SignedLedger
from tensortrade_lab.portfolio import Ledger

HERE=Path(__file__).resolve().parent;OUT=HERE/'ensemble/intraday/BTC'

def execute(p,stream,start,end,cost=1,delay=1,benchmark=False):
    led=Ledger(p,start,end,commission=.001*cost,slippage=.0005*cost) if benchmark else SignedLedger(p,start,end,fee=.001*cost,slip=.0005*cost,borrow=.05*cost)
    for t in range(start,end+1):
        w=None
        if benchmark:
            if t==start:w=np.ones(1)
        elif t==start or p.timestamp[t].normalize()!=p.timestamp[t-1].normalize():w=np.array([stream[t-delay]])
        led.step(w)
    h=pd.DataFrame(led.history);e=h.equity.to_numpy();d=h.set_index('timestamp').equity.resample('1D').last().dropna().to_numpy()
    d=np.r_[10000.,d]
    m={'total_return':float(e[-1]/10000-1),'max_drawdown':float(np.max(1-d/np.maximum.accumulate(d))),
        'max_intraday_drawdown':float(np.max(1-e/np.maximum.accumulate(e))),'fees':led.fees,
        'slippage':led.slip_paid if benchmark else led.slippage,'borrow':0. if benchmark else led.borrow_paid,
        'collateral_liquidations':0 if benchmark else led.margin_liquidations,'bankrupt':False if benchmark else led.bankrupt}
    return m,h,led.fills

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    proto=OUT/'protocol.json'
    if not proto.exists():save(proto,{'registered_at':datetime.now(timezone.utc).isoformat(),
      'market':'BITSTAMP BTCUSD15m; warmupMar2024-Dec2024, evaluation2025-May2026',
      'experts':'All7savedPF15m+savedquantBTC_TVRange_Reversal_64k; averageheldcloseexposurewithinorigin. Papercompleteddaytrend,IBS,12%voltarget365dayconvention. Existing expertsettings were selected retrospectively.',
      'execution':'New signed spot account; firstUTCbar eachday trades prior completed15m close target.95%targetcap,$10k,10bpfee+5bpslip,zero cash yield. Do not copy intrabar protectiveorders. Matched100%spot buy-and-hold. 5%annualborrow base,10%doublecoststress;1.5times shortproceeds collateral; availabilityunverified.',
      'methods':METHODS,'weights':'Equalfamilies,previous63observeddays inversevol,clippedPastSharpesoftmaxpluscash,omitPF,omitQuant. Weights use causally re-executed individualfamily returns.',
      'cases':'full,2025,2026 separately; base,doublecost+borrow,extra15mbarlag. No tuning, no PPO in this intraday branch. Separate daily study tests TensorTrade.',
      'qualification':False})
    feed=HERE/'next_batch/BTC_BITSTAMP_15m.parquet'
    if sha(feed)!=json.loads((HERE/'next_batch/btc_manifest.json').read_text())['data_sha256']:raise ValueError('BTC source changed')
    f=pd.read_parquet(feed);f=f[f.timestamp>=pd.Timestamp('2024-03-01',tz='UTC')].reset_index(drop=True)
    if (f.timestamp.diff().dropna()!=pd.Timedelta(minutes=15)).any():raise ValueError('Missing BTC bars')
    f=f.assign(dividend=0.,split=1.,scale=1.);p=panel_from({'BTC':f},False,365*96)
    targets,names=paper_daily(f)
    daily=f.set_index('timestamp').close.resample('1D').last().dropna()
    risk=pd.DataFrame({'known_at':daily.index+pd.Timedelta(days=1),'risk':np.minimum(.95,.12/(daily.pct_change().rolling(63).std()*np.sqrt(365))).fillna(0).to_numpy()})
    targets[:,2]=pd.merge_asof(pd.DataFrame({'known_at':pd.DatetimeIndex(f.bar_end)}),risk,on='known_at',direction='backward').risk.fillna(0).to_numpy()
    paths=[(r['id'],'PatternFindr') for r in json.loads((HERE/'extension/patternfindr_configs.json').read_text()) if r.get('config',{}).get('ticker')=='BTC-USD' and r['config'].get('interval')=='15m']+[('BTC_TVRange_Reversal_64k','Quant')]
    groups={};audit=[]
    for name,family in paths:
        hp=HERE/'next_batch'/(name+'_equity.parquet');fp=HERE/'next_batch'/(name+'_fills.json')
        h=pd.read_parquet(hp).set_index('timestamp').reindex(pd.DatetimeIndex(f.bar_end));trades=pd.Series(0.,index=pd.DatetimeIndex(f.timestamp))
        if h.equity.isna().any() or (h.equity<=0).any():raise ValueError('Invalid shadow equity')
        for row in json.loads(fp.read_text()):
            at=pd.Timestamp(row['timestamp'])
            if at not in trades.index:raise ValueError('Unmatched shadow fill')
            trades.loc[at]+=row['units']
        w=np.clip(trades.cumsum().to_numpy()*f.close.to_numpy()/h.equity.to_numpy(),-.95,.95)
        groups.setdefault(family,[]).append(w);audit.append({'name':name,'family':family,'curve_sha256':sha(hp),'fills_sha256':sha(fp)})
    for family,ww in groups.items():targets=np.column_stack([targets,np.mean(ww,axis=0)]);names.append(family)
    save(OUT/'sources.json',{'feed_sha256':sha(feed),'shadow_books':audit,'families':names})
    returns=[]
    for i in range(len(names)):
        _,h,_=execute(p,targets[:,i],1,len(f)-1);returns.append(h.equity.pct_change().fillna(0).to_numpy())
    returns=np.column_stack(returns)
    streams={m:np.sum(fit_weights(f,returns,names,m)[:,:-1]*targets,axis=1) for m in METHODS}
    rows=[];first=int(np.flatnonzero(f.timestamp>=pd.Timestamp('2025-01-01',tz='UTC'))[0])
    periods=[('full',first,len(f)-1)]+[(str(y),int(ix[0]),int(ix[-1])) for y in [2025,2026] if len(ix:=np.flatnonzero(f.timestamp.dt.year==y))>1]
    for block,a,b in periods:
        for case,cost,delay in [('base',1,1),('double_cost',2,1),('extra_bar_delay',1,2)]:
            bm,bh,bf=execute(p,None,a,b,cost,benchmark=True)
            for method,s in streams.items():
                m,h,ff=execute(p,s,a,b,cost,delay)
                rows.append({'market':'BTC15m','method':method,'block':block,'case':case,**m,'benchmark_return':bm['total_return'],'benchmark_drawdown':bm['max_drawdown'],'pass':bool(m['total_return']>bm['total_return'] and m['max_drawdown']<=bm['max_drawdown'] and not m['bankrupt'])})
                if block=='full':h.to_parquet(OUT/(method+'_'+case+'.parquet'),index=False);save(OUT/(method+'_'+case+'_fills.json'),ff)
            if block=='full':bh.to_parquet(OUT/('BuyHold_'+case+'.parquet'),index=False);save(OUT/('BuyHold_'+case+'_fills.json'),bf)
    save(OUT/'results.json',rows);save(OUT/'manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'candidate_cases':len(rows),'shadow_books':len(paths),'code_sha256':sha(__file__),'protocol_sha256':sha(proto),'source_sha256':sha(OUT/'sources.json'),'all_gates_pass':[m for m in METHODS if all(r['pass'] for r in rows if r['method']==m)],'execution_qualified':False})
    print('BTC15m complete',len(rows),flush=True)

if __name__=='__main__':main()
