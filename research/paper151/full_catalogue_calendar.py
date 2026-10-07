"""§10.2 retry: select the most liquid prior-known deferred contract.

Original CalendarSlope2 and its missing-price failures are preserved unchanged.
This is a distinct fixed liquidity adaptation, not a filled-in old backtest.
"""
import numpy as np
import pandas as pd
import futures_extension as engine
from full_catalogue_options import HERE,OUT,save,sha

def main():
    protocol=OUT/'calendar_protocol.json'
    if not protocol.exists():save(protocol,{'section':'10.2','registered_before_scoring':True,
      'change':'Most liquid eligible deferred delivery month by preceding completed UTC-day volume, instead of earliest later maturity; close prices never used to select contract availability.',
      'signal':'Same prior near/deferred slope sign; GC/CL equal gross spread allocations; both raw-price legs, no imputed bar or mid-price.',
      'windows':engine.BLOCKS,'costs':'$2.50 per contract side plus 1/3/5 ticks; extra-day signal sensitivity; $1m capital and <=95% target gross.',
      'qualification':'Modeled fills only. Any missing held/preselected contract price fails the entire case. Original calendar failures retained.'})
    frame=pd.read_parquet(HERE/'extension/actual_futures_daily.parquet');data=list(engine.prepare(frame));dates,books,front,second,tr,ret,carry,_,_=data
    choices=[]
    for i,day in enumerate(dates):
        before=books[dates[max(0,i-1)]]
        for k,root in enumerate(engine.ROOTS):
            eligible=[s for s,z in before.items() if z['root']==root and z['close']>0 and engine.month(front[i][root],day)<engine.month(s,day)<day+pd.DateOffset(years=2)]
            eligible.sort(key=lambda s:(-before[s]['volume'],s));deferred=eligible[0] if eligible else None
            second[i][root]=deferred
            carry.iloc[i,k]=books[day][front[i][root]]['close']/books[day][deferred]['close'] if deferred in books[day] and front[i][root] in books[day] and books[day][deferred]['close']>0 else np.nan
            if root in ['GC','CL']:choices.append({'date':str(day),'root':root,'near':front[i][root],'deferred':deferred,'selected_from_completed_day':str(dates[max(0,i-1)])})
    save(OUT/'calendar_selections.json',choices);rows=[];errors=[]
    for window,left,right in engine.BLOCKS:
        ix=np.flatnonzero((dates>=pd.Timestamp(left,tz='UTC'))&(dates<pd.Timestamp(right,tz='UTC')))
        for case,ticks,delay in [('base',1,1),('three_ticks',3,1),('five_ticks',5,1),('extra_day',1,2)]:
            try:
                m,h,fills=engine.run(data,'CalendarSlope2',int(ix[0]),int(ix[-1]),ticks=ticks,delay=delay)
                b,_,_=engine.run(data,'PassiveCommodity',int(ix[0]),int(ix[-1]),ticks=ticks,delay=delay)
                rows.append({'section':'10.2','strategy':'CalendarLiquidSlope2','window':window,'case':case,**m,
                  'benchmark_return':b['total_return'],'benchmark_max_drawdown':b['max_drawdown'],
                  'pass':bool(m['total_return']>b['total_return'] and m['max_drawdown']<=b['max_drawdown']),'qualified':False})
                if window=='full' and case=='base':pd.DataFrame(h).to_parquet(OUT/'calendar_equity.parquet',index=False);save(OUT/'calendar_fills.json',fills)
            except ValueError as e:errors.append({'section':'10.2','strategy':'CalendarLiquidSlope2','window':window,'case':case,'reason':str(e)})
    save(OUT/'calendar_results.json',rows);save(OUT/'calendar_errors.json',errors)
    save(OUT/'calendar_manifest.json',{'completed':len(rows),'failed_closed':len(errors),'code_sha256':sha(__file__),'data_sha256':sha(HERE/'extension/actual_futures_daily.parquet'),'protocol_sha256':sha(protocol)})
    print('Calendar retry',len(rows),'completed,',len(errors),'failed closed',flush=True)
if __name__=='__main__':main()
