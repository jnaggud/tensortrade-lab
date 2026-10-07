"""§7.4/7.4.1 quote-sample adaptations: IV/RV gate and hourly delta hedging.

Model IV is used for signals/Greeks only; all hypothetical fills use bid/ask.
These samples never qualify a strategy for deployment.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from full_catalogue_options import HERE,OUT,save,sha,episodes,quotes_at

def black76(f,k,t,vol,right):
    d1=(np.log(f/k)+.5*vol*vol*t)/(vol*np.sqrt(t));d2=d1-vol*np.sqrt(t)
    if right=='C':return f*norm.cdf(d1)-k*norm.cdf(d2),norm.cdf(d1)
    return k*norm.cdf(-d2)-f*norm.cdf(-d1),norm.cdf(d1)-1

def iv_delta(f,k,t,premium,right):
    if t<=0 or min(f,k,premium)<=0:raise ValueError('Invalid IV input')
    try:vol=brentq(lambda v:black76(f,k,t,v,right)[0]-premium,.0001,5.)
    except ValueError:raise ValueError('No finite model IV inside fixed bounds')
    return vol,black76(f,k,t,vol,right)[1]

def signal(selection,quote,clock,history):
    mapping=selection['mapping'];uid=mapping['F:0:0'];cid=mapping['C:0:0'];pid=mapping['P:0:0']
    midpoint=lambda iid:float((quote[iid].bid_px_00+quote[iid].ask_px_00)/2)
    expiry=pd.Timestamp(selection['near']);t=(expiry-clock).total_seconds()/(365.25*86400)
    f=midpoint(uid);k=selection['center']
    c,dc=iv_delta(f,k,t,midpoint(cid),'C');p,dp=iv_delta(f,k,t,midpoint(pid),'P')
    prices=history[(history.symbol==selection['underlying'])&(history.ts_event+pd.Timedelta(days=1)<=clock)].sort_values('ts_event').close.tail(21).to_numpy()
    if len(prices)!=21 or np.any(prices<=0):raise ValueError('Need 21 completed actual-underlying daily closes')
    rv=float(np.std(np.diff(np.log(prices)),ddof=1)*np.sqrt(252))
    return {'implied_vol':float((c+p)/2),'realized_vol':rv,'enter':bool((c+p)/2>rv),'future_hedge':int(np.rint(dc+dp)),
       'estimated_option_delta':float(-dc-dp),'decision_at':str(clock),'rate_assumption':0.}

def run(f,s,day,history,hedge,stress):
    mapping=s['mapping'];ids=[mapping[k] for k in ['F:0:0','C:0:0','P:0:0']];uid,cid,pid=ids
    grid=pd.DatetimeIndex([t for d in pd.date_range(day,periods=3) for t in pd.date_range(d+pd.Timedelta(hours=14),d+pd.Timedelta(hours=19,minutes=55),freq='5min')])
    q,valid=quotes_at(f,ids,[1,1,1],grid)
    if not valid[0] or not valid[-1]:raise ValueError('Missing entry or exit sized quotes')
    dq,dv=quotes_at(f,ids,[1,1,1],grid-pd.Timedelta(minutes=5))
    if not dv[0]:raise ValueError('Missing pre-entry signal quotes')
    a=signal(s,{iid:dq[iid].iloc[0] for iid in ids},grid[0]-pd.Timedelta(minutes=5),history)
    active=a['enter'];tick=.25*stress;fee=2.5;capital=1e6;cash=capital;position=0;mark=None;fills=[];equity=[];missing_hedges=[]
    if active:
        for iid in [cid,pid]:
            price=float(q[iid].bid_px_00.iloc[0])-tick
            if price<0:raise ValueError('Negative short option fill')
            cash+=price*50-fee;fills.append({'timestamp':str(grid[0]),'instrument_id':iid,'kind':'option','units':-1,'price':price,'fee':fee})
    for i,time in enumerate(grid):
        if not valid[i]:continue
        bid=float(q[uid].bid_px_00.iloc[i]);ask=float(q[uid].ask_px_00.iloc[i]);mid=(bid+ask)/2
        if position and mark is not None:cash+=position*(mid-mark)*50
        mark=mid
        if active and hedge and time.minute==0:
            try:
                if not dv[i]:raise ValueError('Missing pre-hedge quotes')
                sig=signal(s,{iid:dq[iid].iloc[i] for iid in ids},time-pd.Timedelta(minutes=5),history)
                target=sig['future_hedge'];delta=target-position
                if delta:
                    side_size=q[uid].ask_sz_00.iloc[i] if delta>0 else q[uid].bid_sz_00.iloc[i]
                    if side_size<abs(delta):raise ValueError('Insufficient hedge quote size')
                    fill=ask+tick if delta>0 else bid-tick
                    cash-=delta*(fill-mid)*50+abs(delta)*fee;position=target
                    fills.append({'timestamp':str(time),'instrument_id':uid,'kind':'future','units':delta,'price':fill,'fee':abs(delta)*fee,'decided_at':sig['decision_at']})
            except ValueError as e:missing_hedges.append({'timestamp':str(time),'reason':str(e)})
        option_close=sum((float(q[iid].ask_px_00.iloc[i])+tick)*50+fee for iid in [cid,pid]) if active else 0
        future_liq=position*((bid-tick if position>0 else ask+tick)-mid)*50-abs(position)*fee
        nav=cash-option_close+future_liq
        benchmark=capital+(bid-tick-float(q[uid].ask_px_00.iloc[0])-tick)*50-2*fee
        equity.append({'timestamp':time,'equity':nav,'benchmark_equity':benchmark,'futures_hedge':position})
    if active:
        for iid in [cid,pid]:fills.append({'timestamp':str(grid[-1]),'instrument_id':iid,'kind':'option','units':1,'price':float(q[iid].ask_px_00.iloc[-1])+tick,'fee':fee})
    if position:
        fills.append({'timestamp':str(grid[-1]),'instrument_id':uid,'kind':'future','units':-position,'price':float(q[uid].bid_px_00.iloc[-1])-tick if position>0 else float(q[uid].ask_px_00.iloc[-1])+tick,'fee':abs(position)*fee})
    h=pd.DataFrame(equity);eq=np.r_[capital,h.equity.to_numpy()];bh=np.r_[capital,h.benchmark_equity.to_numpy()]
    dd=lambda x:float(np.max(1-x/np.maximum.accumulate(x)))
    # Signed cashflows of all closed legs must reconcile with variation-margin model.
    signed_cash=sum(-x['units']*x['price']*50-x['fee'] for x in fills)
    if not np.isclose(signed_cash,eq[-1]-capital,rtol=0,atol=1e-6):raise ValueError('Volatility hedge accounting does not reconcile')
    return {'return':float(eq[-1]/capital-1),'max_sampled_drawdown':dd(eq),'benchmark_return':float(bh[-1]/capital-1),
       'benchmark_max_sampled_drawdown':dd(bh),'entered':active,'fills':len(fills),'fees':sum(x['fee'] for x in fills),
       'missing_marks':int((~valid).sum()),'missed_hedges':len(missing_hedges),'pass_in_episode_only':bool(eq[-1]>bh[-1] and dd(eq)<=dd(bh)),
       'qualified':False,'signal':a},h,fills,missing_hedges

def register():
    p=OUT/'volatility_protocol.json'
    if not p.exists():save(p,{'sections':['7.4','7.4.1'],'registered_before_scoring':True,
       'samples':'Same twelve fixed three-day episodes and selected ES weekly contracts as options_protocol.json.',
       'gate':'At 13:55 Monday, average ATM call/put Black76 midpoint IV (zero discount rate) greater than 20 daily log-return realized volatility on the actual underlying contract; enter 14:00 using bid/ask. No position otherwise.',
       'gamma_variant':'Short one ATM call and put; each hour decide from observed quotes five minutes earlier; round target ES futures delta hedge to nearest integer, <=1 unit. Hold previous hedge if signal/fill missing. Close all positions Wednesday 19:55.',
       'execution':'$50 per point; $1m reference capital; $2.50 per contract side and bid/ask. Extra 0.25-point adverse tick stress. No model premiums used as fill prices.',
       'limits':'Sparse episodes, model Greeks, integer imperfect delta hedging, zero discount/collateral yield assumptions and no dated margin or intraday tick queue history. Never promotion-qualified.'})

def main():
    register();history=pd.read_parquet(HERE/'extension/actual_futures_daily.parquet');rows=[];errors=[]
    for day in episodes():
        folder=OUT/'options'/day.strftime('%Y%m%d')
        try:
            import json
            s=json.loads((folder/'selection.json').read_text())
            f=pd.concat([pd.read_parquet(folder/f'quotes_{d.strftime("%Y%m%d")}.parquet') for d in pd.date_range(day,periods=3)],ignore_index=True)
        except (ValueError,FileNotFoundError) as e:
            errors.append({'episode':str(day.date()),'reason':str(e)});continue
        for section,hedge in [('7.4',False),('7.4.1',True)]:
            for stress in [0,1]:
                key={'section':section,'episode':str(day.date()),'case':'extra_tick' if stress else 'base'}
                try:
                    r,h,fills,miss=run(f,s,day,history,hedge,stress);rows.append(dict(key,**r));name='vol_'+section.replace('.','_')+'_'+key['case']
                    h.to_parquet(folder/(name+'_equity.parquet'),index=False);save(folder/(name+'_fills.json'),fills);save(folder/(name+'_missed_hedges.json'),miss)
                except (ValueError,KeyError) as e:errors.append(dict(key,reason=str(e)))
    save(OUT/'volatility_results.json',rows);save(OUT/'volatility_errors.json',errors)
    save(OUT/'volatility_manifest.json',{'completed':len(rows),'failed_closed':len(errors),'entered_cases':sum(r['entered'] for r in rows),
       'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'volatility_protocol.json'),'qualified':0})
    print('Volatility sample cases',len(rows),'errors',len(errors),flush=True)
if __name__=='__main__':main()
