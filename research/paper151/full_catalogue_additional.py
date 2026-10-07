"""Two additional directly specified paper ETF/ETN implementations."""
import numpy as np
import pandas as pd
import full_catalogue_prices as engine
from full_catalogue_options import OUT,save,sha
from run_study import panel_from

def etn_target(p,j):
    r=p.total_index[j-125:j+1,:2]/p.total_index[j-126:j,:2]-1
    x=r[:,1]-r[:,1].mean();y=r[:,0]-r[:,0].mean()
    beta=float(np.clip(x@y/max(x@x,1e-14),0,3))
    w=np.zeros(len(p.symbols));w[0]=-1/(1+beta);w[1]=beta/(1+beta)
    return w,{'beta':beta,'last_training_close':str(p.bar_end[j])}

def main():
    proto=OUT/'additional_protocol.json'
    if not proto.exists():save(proto,{
      'ETNCarry':{'section':'7.3','universe':['VXX','VXZ','SPY'],'rule':'Monthly short VXX, long VXZ; 126 prior daily-return regression beta clipped [0,3], dollar gross 1. No splice to predecessor ETNs; only retrieved series after 2018 inception. Evaluate 2019 onward.',
        'borrow':'5% annual base, 20% stress; no historical locate/recall proof, diagnostic only.'},
      'InfrastructureMix':{'section':'20','universe':['SPY','IGF'],'rule':'Monthly 80% SPY and 20% IGF listed global infrastructure ETF; compare 100% SPY. A public-fund diversification test allowed by §20, not project-level infrastructure cashflows.'},
      'execution':'Same next-session-open signed/long cash-share ledgers; fees 5bp and slippage 5bp each side, doubled-cost, extra-day and zero-cash tests; no parameter fitting on outcomes.',
      'registered_before_scoring':True})
    rows=[];errors=[];sources={};original=engine.target
    for name,section,symbols,route in [('ETNCarry','7.3',['VXX','VXZ','SPY'],'ShortLeveragedPair'),('InfrastructureMix','20',['SPY','IGF'],'REITMixed')]:
        frames,sources[name]=engine.load_new(symbols)
        # Preserve the shared actual observations; do not fill missing market bars.
        start=max(f.timestamp.iloc[0] for f in frames.values());end=min(f.timestamp.iloc[-1] for f in frames.values())
        frames={s:f[(f.timestamp>=start)&(f.timestamp<=end)].reset_index(drop=True) for s,f in frames.items()}
        p=panel_from(frames,True,252)
        engine.target=lambda rule,pp,j,prepared:etn_target(pp,j) if name=='ETNCarry' and rule=='ShortLeveragedPair' else original(rule,pp,j,prepared)
        blocks=[('full','2019-01-01','2026-09-24'),('2019_2020','2019-01-01','2021-01-01'),('2021_2023','2021-01-01','2024-01-01'),('2024_latest','2024-01-01','2026-09-24')] if name=='ETNCarry' else engine.BLOCKS
        for window,left,right in blocks:
            ix=np.flatnonzero((p.timestamp>=pd.Timestamp(left,tz='UTC'))&(p.timestamp<pd.Timestamp(right,tz='UTC')));a=max(127,int(ix[0]));b=int(ix[-1])
            cases={'base':{}}
            if window=='full':cases.update(double_cost={'fee':.001,'slip':.001},extra_day={'delay':2},zero_cash={'zero_cash':True},borrow_20pct={'borrow':.2})
            for case,args in cases.items():
                settings={'borrow':.05,**args}
                try:
                    m,h,fills,decisions=engine.simulate(p,route,None,a,b,**settings)
                    bench,bh,bfills,_=engine.simulate(p,'BuyHold',None,a,b,**settings)
                    rows.append({'section':section,'strategy':name,'window':window,'case':case,**m,
                      'benchmark_return':bench['return'],'benchmark_max_drawdown':bench['max_drawdown'],
                      'pass':bool(m['return']>bench['return'] and m['max_drawdown']<=bench['max_drawdown'] and not m['bankrupt']),'qualified':False})
                    if window=='full' and case=='base':
                        folder=OUT/'price_ledgers';h.to_parquet(folder/(name+'.parquet'),index=False);bh.to_parquet(folder/(name+'_benchmark.parquet'),index=False)
                        save(folder/(name+'_fills.json'),fills);save(folder/(name+'_decisions.json'),decisions);save(folder/(name+'_benchmark_fills.json'),bfills)
                except (ValueError,ArithmeticError) as e:errors.append({'strategy':name,'window':window,'case':case,'reason':str(e)})
        print('Completed',name,flush=True)
    engine.target=original
    save(OUT/'additional_results.json',rows);save(OUT/'additional_errors.json',errors)
    save(OUT/'additional_manifest.json',{'runs':len(rows),'benchmark_runs':len(rows),'errors':len(errors),'sources':sources,'code_sha256':sha(__file__),'protocol_sha256':sha(proto)})
if __name__=='__main__':main()
