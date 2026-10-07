"""Actual-contract daily-bar research for paper §§9.1,9.5,10.2,10.3,10.4.

Cash-account P&L uses unadjusted contract prices and both roll legs. Daily OHLC
fills remain a model, not observed simultaneous executable quotes. No margin
leverage: target gross notional <=95% equity, integer contracts, $1m reference.
"""
from pathlib import Path
from datetime import datetime,timezone
import re
import numpy as np
import pandas as pd
from cross_project_inventory import OUT,save,sha

SPECS={'ES':(50.,.25),'NQ':(20.,.25),'GC':(100.,.1),'CL':(1000.,.01),'ZN':(1000.,1/64)}
ROOTS=list(SPECS)
RECIPES=['FutTrendSign252','FutTrendSmooth252','FutTrendNeutral252','FutContrarian5',
         'FutContrarianVol5','FutContrarianVar5','CommodityCarryRank2','CommoditySkew252Rank2','CalendarSlope2']
BLOCKS=[('full','2018-01-01','2026-06-29'),('2018_2019','2018-01-01','2020-01-01'),
        ('2020_2022','2020-01-01','2023-01-01'),('2023_latest','2023-01-01','2026-06-29')]

def month(symbol,day):
    m=re.search(r'([FGHJKMNQUVXZ])(\d{1,2})$',symbol)
    mo='FGHJKMNQUVXZ'.index(m[1])+1;yy=int(m[2]);base=day.year-1
    if len(m[2])==2:
        year=(day.year//100)*100+yy
        if year-day.year>50:year-=100
        if day.year-year>50:year+=100
    else:year=next(y for y in range(base,base+12) if y%10==yy)
    return pd.Timestamp(year=year,month=mo,day=1,tz='UTC')

def prepare(frame):
    frame=frame.copy();frame['ts_event']=pd.to_datetime(frame.ts_event,utc=True)
    frame=frame[frame.volume>0]
    dates=sorted(set.intersection(*(set(frame.loc[frame.root==r,'ts_event']) for r in ROOTS)))
    books={t:g.set_index('symbol').to_dict('index') for t,g in frame[frame.ts_event.isin(dates)].groupby('ts_event')}
    front=[];second=[];signal=np.zeros((len(dates),len(ROOTS)));returns=np.zeros_like(signal);carry=np.zeros_like(signal)
    changes=[];gaps=[]
    for i,t in enumerate(dates):
        current={};deferred={}
        # Decisions for day's selected contracts use the preceding completed day's volume.
        before=books[dates[max(0,i-1)]]
        for k,r in enumerate(ROOTS):
            # CL expires before its delivery month. A deliberately early 45-day
            # cutoff avoids delivery/expiry without inventing historical calendars.
            buffer=45 if r=='CL' else 7
            candidates=[(s,z) for s,z in before.items() if z['root']==r and
                        t+pd.Timedelta(days=buffer)<month(s,t)<t+pd.DateOffset(years=2) and z['close']>0]
            candidates.sort(key=lambda x:(-x[1]['volume'],x[0]))
            if not candidates:raise ValueError(f'No eligible prior-known contract: {r} {t}')
            chosen=candidates[0][0];current[r]=chosen
            latter=[(s,z) for s,z in candidates if month(s,t)>month(chosen,t)]
            latter.sort(key=lambda x:(month(x[0],t),-x[1]['volume']))
            deferred[r]=latter[0][0] if latter else None
            if chosen not in books[t]:
                gaps.append({'date':str(t),'root':r,'symbol':chosen})
                signal[i,k]=np.nan;returns[i,k]=np.nan;carry[i,k]=np.nan;continue
            price=books[t][chosen]['close']
            if i==0:signal[i,k]=100.
            else:
                prev=before[chosen]['close'];ret=price/prev-1
                returns[i,k]=ret;signal[i,k]=signal[i-1,k]*(1+ret)
            back=deferred[r]
            carry[i,k]=price/books[t][back]['close'] if back in books[t] and books[t][back]['close']>0 else np.nan
            if i and chosen!=front[-1][r]:changes.append({'date':str(t),'root':r,'old':front[-1][r],'new':chosen})
        front.append(current);second.append(deferred)
    return pd.DatetimeIndex(dates),books,front,second,pd.DataFrame(signal),pd.DataFrame(returns),pd.DataFrame(carry),changes,gaps

def target_weights(rule,j,tr,ret,carry):
    if rule=='PassiveFutures':return np.ones(len(ROOTS))/len(ROOTS)
    if rule=='PassiveCommodity':return np.array([.5 if r in ('GC','CL') else 0 for r in ROOTS])
    if rule=='Cash':return np.zeros(len(ROOTS))
    vol=ret.iloc[max(0,j-62):j+1].std().to_numpy()*np.sqrt(252)
    if rule.startswith('FutTrend'):
        momentum=tr.iloc[j].to_numpy()/tr.iloc[j-252].to_numpy()-1
        sign=np.tanh(momentum/max(np.std(momentum),1e-8)) if rule=='FutTrendSmooth252' else np.sign(momentum)
        w=sign/np.maximum(vol,.01)
        if rule=='FutTrendNeutral252':w-=w.mean()
    elif rule.startswith('FutContrarian'):
        r=tr.iloc[j].to_numpy()/tr.iloc[j-5].to_numpy()-1
        power={'FutContrarian5':0,'FutContrarianVol5':1,'FutContrarianVar5':2}[rule]
        w=(r.mean()-r)/np.maximum(vol,.01)**power
    elif rule in ('CommodityCarryRank2','CommoditySkew252Rank2'):
        ix=[ROOTS.index(r) for r in ('GC','CL')]
        score=carry.iloc[j].to_numpy() if rule=='CommodityCarryRank2' else -ret.iloc[j-251:j+1].skew().to_numpy()
        if not np.isfinite(score[ix]).all():return np.zeros(len(ROOTS))
        ranked=sorted(ix,key=lambda k:(score[k],k));w=np.zeros(len(ROOTS));w[ranked[-1]]=.5;w[ranked[0]]=-.5
    elif rule=='CalendarSlope2':
        w=np.zeros(len(ROOTS))
        for r in ('GC','CL'):
            k=ROOTS.index(r);w[k]=np.sign(carry.iloc[j,k]-1)*.25 if np.isfinite(carry.iloc[j,k]) else 0
        return w
    else:raise ValueError(rule)
    return w/abs(w).sum() if abs(w).sum()>0 else np.zeros(len(ROOTS))

def run(data,rule,start,end,ticks=1,fee=2.5,delay=1):
    dates,books,front,second,tr,ret,carry,_,_=data
    cash=1_000_000.;positions={};marks={};fills=[];history=[];last_key=None;w=np.zeros(len(ROOTS));rejections=0
    reserve_cap=0.95
    for t in range(start,end+1):
        day=dates[t];j=t-delay;book=books[day]
        # Held contracts must have observed prices, even if a new contract is more liquid.
        missing=set(positions)-set(book)
        if missing:raise ValueError(f'Missing held-contract daily bar at {day}: {sorted(missing)}')
        for s,q in positions.items():cash+=q*SPECS[book[s]['root']][0]*(book[s]['open']-marks[s])
        week=dates[j].isocalendar();key=(week.year,week.week) if rule.startswith('FutContrarian') else (dates[j].year,dates[j].month)
        if rule in ('PassiveFutures','PassiveCommodity','Cash'):key='fixed_weights'
        rebalance=key!=last_key or t==start
        if rebalance:w=target_weights(rule,j,tr,ret,carry);last_key=key
        # Contract choice for execution t was fixed from completed t-1 volume.
        desired={}
        if cash>0:
            for k,r in enumerate(ROOTS):
                s=front[t][r]
                if not w[k]:continue
                if s not in book:raise ValueError(f'Missing preselected contract {s} at {day}')
                # Sizes use known prior closes, with an opening-price collateral check below.
                prior=books[dates[t-1]][s]['close']
                q=int(np.trunc(reserve_cap*cash*w[k]/(abs(prior)*SPECS[r][0])))
                if q:desired[s]=q
                if rule=='CalendarSlope2' and q:
                    back=second[t][r]
                    if back not in book:raise ValueError(f'Missing calendar deferred contract at {day}')
                    desired[back]=-q
            # Rebalance exposure monthly/weekly; roll unchanged quantities between decisions.
            if not rebalance and rule!='Cash':
                held_by_root={r:[] for r in ROOTS}
                for s,q in positions.items():held_by_root[book[s]['root']].append((s,q))
                desired={}
                for r,pairs in held_by_root.items():
                    if rule=='CalendarSlope2':
                        # Calendar spreads reset when either selected maturity changes.
                        pairs.sort(key=lambda z:month(z[0],day))
                        if pairs:
                            q=pairs[0][1];desired[front[t][r]]=q
                            if second[t][r] is None:raise ValueError('Missing deferred maturity')
                            desired[second[t][r]]=-q
                    elif pairs:desired[front[t][r]]=sum(q for _,q in pairs)
            if any(s not in book for s in desired):raise ValueError('Cannot invent a new-contract fill')
            def gross(target):return sum(abs(q*book[s]['open']*SPECS[book[s]['root']][0]) for s,q in target.items())
            if gross(desired)>max(0,cash)*.99:
                factor=max(0,cash)*.95/gross(desired);desired={s:int(q*factor) for s,q in desired.items() if int(q*factor)};rejections+=1
        for s in sorted(set(positions)|set(desired)):
            delta=desired.get(s,0)-positions.get(s,0)
            if not delta:continue
            mult,tick=SPECS[book[s]['root']];slip=abs(delta)*tick*ticks*mult;comm=abs(delta)*fee
            cash-=slip+comm
            fills.append({'date':str(day),'decision_at':str(dates[t-1]+pd.Timedelta(days=1)),
                          'symbol':s,'units':delta,'reference_open':book[s]['open'],
                          'modeled_fill':book[s]['open']+np.sign(delta)*tick*ticks,'fee':comm,'slippage':slip})
        positions=desired
        for s,q in positions.items():cash+=q*SPECS[book[s]['root']][0]*(book[s]['close']-book[s]['open'])
        marks={s:book[s]['close'] for s in positions}
        if t==end:
            for s,q in positions.items():
                mult,tick=SPECS[book[s]['root']];comm=abs(q)*fee;slip=abs(q)*tick*ticks*mult;cash-=comm+slip
                fills.append({'date':str(day),'symbol':s,'units':-q,'reference_close':book[s]['close'],
                              'fee':comm,'slippage':slip,'reason':'terminal_liquidation'})
        history.append({'timestamp':day+pd.Timedelta(days=1),'equity':cash,'gross_notional':sum(abs(q*marks[s]*SPECS[book[s]['root']][0]) for s,q in positions.items())})
    eq=np.r_[1_000_000.,[x['equity'] for x in history]];days=(dates[end]-dates[start]).days+1
    m={'total_return':eq[-1]/eq[0]-1,'cagr':(eq[-1]/eq[0])**(365.25/days)-1 if eq[-1]>0 else -1,
       'max_drawdown':float((1-eq/np.maximum.accumulate(eq)).max()),'fees':sum(x['fee'] for x in fills),
       'slippage':sum(x['slippage'] for x in fills),'fills':len(fills),'capital':1_000_000.,
       'start':str(dates[start]),'end':str(dates[end]),'opening_size_reductions':rejections,
       'ever_insolvent':bool(eq.min()<=0),'interest':0.}
    return m,history,fills

def main():
    save('futures_protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),'recipes':RECIPES,'blocks':BLOCKS,
        'contracts':SPECS,'capital':1_000_000,'target_gross_notional':.95,'commission_per_contract_side':2.5,
        'cases':['base_1_tick','stress_3_ticks','stress_5_ticks','extra_signal_day'],
        'roll':'Highest preceding completed UTC-day volume; exclude delivery months starting within 7 days (45 days for CL, which expires before delivery month). Both actual roll legs charged. This intentionally early roll is not the original cache roll.',
        'time':'UTC daily OHLC buckets. First traded price after the daily boundary is a modeled opening fill; its exact timestamp/size is unavailable here. No simultaneous quote claim.',
        'commodity_universe':'Only GC and CL: two-commodity rank adaptation, not original wide-universe tercile/quintile replication.',
        'calendar':'Buy near/sell deferred when prior near/deferred ratio exceeds 1, inverse otherwise. Slope signal is a declared adaptation, not provided fundamental forecasts.',
        'collateral':'Unlevered gross exposure; no historical exchange margin schedule or collateral yield claim. Cash interest zero in both strategy and rolled benchmark.',
        'qualification':'Historical modeled screen only. Executable quote/size validation and margin history still required before promotion.'})
    f=pd.read_parquet(OUT/'actual_futures_daily.parquet');data=prepare(f)
    dates=data[0];save('futures_roll_audit.json',{'changes':data[-2],'missing_preselected_prices':data[-1],
        'source_sha256':sha(OUT/'actual_futures_daily.parquet'),'common_dates':len(dates),
        'available_dates_by_root':f.groupby('root').ts_event.nunique().to_dict()})
    # Missing preselected bars are not imputed. Entire affected cases fail closed.
    rows=[];errors=[]
    for block,begin,finish in BLOCKS:
        ix=np.where((dates>=pd.Timestamp(begin,tz='UTC'))&(dates<pd.Timestamp(finish,tz='UTC')))[0]
        start=max(int(ix[0]),253);end=int(ix[-1])
        for case,ticks,delay in [('base',1,1),('three_ticks',3,1),('five_ticks',5,1),('delay_two',1,2)]:
            for rule in ['PassiveFutures','PassiveCommodity','Cash']+RECIPES:
                try:
                    m,h,fill=run(data,rule,start,end,ticks=ticks,delay=delay)
                    rows.append({'origin':'paper_actual_contract_adaptation','strategy':rule,'case':case,'block':block,**m})
                    if block=='full' and case=='base':
                        dest=OUT/'futures_ledgers';dest.mkdir(exist_ok=True)
                        pd.DataFrame(h).to_parquet(dest/(rule+'.parquet'),index=False)
                        save('futures_ledgers/'+rule+'_fills.json',fill)
                except ValueError as e:errors.append({'strategy':rule,'case':case,'block':block,'error':str(e)})
            print('futures case',block,case,'completed',len(rows),'failed_closed',len(errors),flush=True)
    save('futures_results.json',rows);save('futures_errors.json',errors)
    save('futures_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(OUT/'actual_futures_daily.parquet'),
        'script_sha256':sha(__file__),'completed_runs':len(rows),'failed_closed_runs':len(errors),
        'new_recipes':RECIPES,'status':'Actual-contract historical daily-bar modeled fills; not execution-qualified'})

if __name__=='__main__':main()
