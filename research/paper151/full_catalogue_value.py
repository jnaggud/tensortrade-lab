"""§9.4 two-commodity near-contract proxy for five-year physical spot value."""
import numpy as np
import pandas as pd
import futures_extension as engine
from full_catalogue_options import HERE,OUT,save,sha

def signals(data):
    dates,books,front,*_=data;targets={};decisions=[]
    for j,d in enumerate(dates):
        cutoff=d-pd.DateOffset(years=5);past=dates.searchsorted(cutoff,side='right')-1
        w=np.zeros(5);score={}
        if past>=0:
            for root in ['GC','CL']:
                old=books[dates[past]][front[past][root]]['close'];now=books[d][front[j][root]]['close']
                if old<=0 or now<=0:raise ValueError('Nonpositive proxy price')
                score[root]=old/now
            ranked=sorted(score,key=lambda r:(score[r],r));w[engine.ROOTS.index(ranked[0])]=-.5;w[engine.ROOTS.index(ranked[-1])]=.5
        targets[j]=w;decisions.append({'decision_at':str(d+pd.Timedelta(days=1)),'prior_price_date':str(dates[past]) if past>=0 else None,'scores':score,'weights':w.tolist()})
    return targets,decisions

def main():
    protocol=OUT/'value_protocol.json'
    if not protocol.exists():save(protocol,{'section':'9.4','registered_before_scoring':True,'rule':'Monthly long higher five-calendar-year old/current near-contract price ratio, short lower, GC/CL only; equal absolute weights.',
      'data':'Unadjusted prior-volume-selected raw near futures prices as a declared proxy for physical spot prices; both actual roll legs in P&L. NOT a physical spot value replication.',
      'costs':'$1m capital, existing <=95% gross actual-contract engine; $2.50 per side and 1/3/5 ticks, extra-day signal lag.',
      'start':'2022-02-01','end_exclusive':'2026-06-29','blocks':['2022_2023','2024_latest'],'qualification':'No execution qualification; two commodities and futures-for-spot proxy.'})
    f=pd.read_parquet(HERE/'extension/actual_futures_daily.parquet');data=engine.prepare(f);dates=data[0]
    targets,audits=signals(data);save(OUT/'value_decisions.json',audits)
    original=engine.target_weights
    engine.target_weights=lambda rule,j,tr,ret,carry:targets[j] if rule=='CommodityValue5Y' else original(rule,j,tr,ret,carry)
    rows=[];errors=[]
    for window,left,right in [('full','2022-02-01','2026-06-29'),('2022_2023','2022-02-01','2024-01-01'),('2024_latest','2024-01-01','2026-06-29')]:
        ix=np.flatnonzero((dates>=pd.Timestamp(left,tz='UTC'))&(dates<pd.Timestamp(right,tz='UTC')))
        for case,ticks,delay in [('base',1,1),('three_ticks',3,1),('five_ticks',5,1),('extra_day',1,2)]:
            try:
                m,h,fills=engine.run(data,'CommodityValue5Y',int(ix[0]),int(ix[-1]),ticks=ticks,delay=delay)
                b,_,_=engine.run(data,'PassiveCommodity',int(ix[0]),int(ix[-1]),ticks=ticks,delay=delay)
                rows.append({'section':'9.4','strategy':'CommodityValue5Y','window':window,'case':case,**m,
                  'benchmark_return':b['total_return'],'benchmark_max_drawdown':b['max_drawdown'],'pass':bool(m['total_return']>b['total_return'] and m['max_drawdown']<=b['max_drawdown']),'qualified':False})
                if window=='full' and case=='base':pd.DataFrame(h).to_parquet(OUT/'value_equity.parquet',index=False);save(OUT/'value_fills.json',fills)
            except ValueError as e:errors.append({'window':window,'case':case,'reason':str(e)})
    save(OUT/'value_results.json',rows);save(OUT/'value_errors.json',errors)
    save(OUT/'value_manifest.json',{'runs':len(rows),'benchmark_runs':len(rows),'failed_closed':len(errors),'code_sha256':sha(__file__),'data_sha256':sha(HERE/'extension/actual_futures_daily.parquet'),'protocol_sha256':sha(protocol)})
    print('Value cases',len(rows),'errors',len(errors),flush=True)
if __name__=='__main__':main()
