"""Fixed new paper adapters on frozen daily ETF prices; local research only."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scipy.linalg import solve
from expanded_study import load_jobs,BLOCKS
from run_study import panel_from,read_frame,Ledger
from signed_study import SignedLedger
from signed_study import features as signed_features, target as signed_target
from factor_study import load_factors,monthly_returns,factor_window
from full_catalogue_options import HERE,OUT,save,sha

RULES={
 'MomentumLowVolCombo':{'section':'3.6','market':'Sectors','description':'Fixed 50/50 combination of the paper momentum and low-volatility signed ETF factors; monthly rebalance. Uses two price-derived factors, not a value/fundamentals replication.'},
 'OptimizedContrarian':{'section':'3.18','market':'Sectors','description':'Dollar-neutral Eq.358; negative last-day return forecasts; 126 prior daily returns covariance with fixed 50% diagonal shrinkage; gross 1; daily rebalance.'},
 'SelectivityAlpha':{'section':'4.3','market':'Sectors','description':'Nine fixed sector ETFs: lowest-R² tercile, highest alpha one fund; 36 archived-availability monthly observations, three Fama-French factors; monthly rebalance. Proxy for active-fund selectivity, not original active-fund universe.'},
 'ShortLeveragedPair':{'section':'4.5','market':'LETF','description':'Monthly -25% SPXL, -25% SPXS, +50% SHY; explicit distributions, 2% modeled annual borrow and collateral. Point-in-time locate availability is absent; diagnostic only.'},
 **{f'HP_{s}':{'section':'8.1','market':s,'description':'Causal trailing-252 HP(lambda=1600), 20/50 mean crossover, long/cash. Currency ETF adaptation, not spot/forward FX; ETF fees and distributions embedded.'} for s in ['FXE','FXB','FXY','FXA']},
 'REITMixed':{'section':'16.2','market':'REITMix','description':'Monthly 80% SPY/20% VNQ; listed-REIT proxy, not direct-property cashflows.'},
 'REITGeography':{'section':'16.3.3','market':'REITGeo','description':'Monthly 50% VNQ/50% VNQI; US/non-US listed-REIT proxy, not geographic and property-type ownership data.'},
}

def hp_weights(window=252,lam=1600,fast=20,slow=50):
    d=np.diff(np.eye(window),n=2,axis=0)
    contrast=np.zeros(window);contrast[-fast:]+=1/fast;contrast[-slow:]-=1/slow
    return solve(np.eye(window)+lam*d.T@d,contrast,assume_a='pos')

def hp_signal(close):
    c=np.asarray(close,float);w=hp_weights();out=np.zeros(len(c))
    if len(c)>=252:out[251:]=(np.lib.stride_tricks.sliding_window_view(c,252)@w>0).astype(float)
    return out

def optimized_weights(p,j):
    ret=p.total_index[j-126+1:j+1,:9]/p.total_index[j-126:j,:9]-1
    covariance=np.cov(ret,rowvar=False);covariance=.5*covariance+.5*np.diag(np.diag(covariance))
    covariance+=np.eye(9)*1e-10
    forecast=-ret[-1];a=np.linalg.solve(covariance,forecast);b=np.linalg.solve(covariance,np.ones(9))
    w=a-b*(a.sum()/b.sum());w=w/max(abs(w).sum(),1e-15)
    result=np.zeros(len(p.symbols));result[:9]=w
    if abs(w.sum())>1e-8 or abs(w).sum()>1+1e-8:raise ValueError('Optimization normalization failed')
    return result

def selectivity(p,j,monthly,factors):
    f,audit=factor_window('archived',p.bar_end[j],p.timestamp[j+1],factors)
    y=monthly.loc[f.index].iloc[:,:9].to_numpy()-f.RF.to_numpy()[:,None]
    x=np.column_stack([np.ones(len(f)),f[['Mkt-RF','SMB','HML']].to_numpy()])
    if not np.isfinite(y).all():raise ValueError('Incomplete regression window')
    b=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@b
    r2=1-(res**2).sum(axis=0)/np.maximum(((y-y.mean(axis=0))**2).sum(axis=0),1e-12)
    candidates=np.argsort(r2,kind='stable')[:3];winner=candidates[np.argsort(-b[0,candidates],kind='stable')[0]]
    w=np.zeros(len(p.symbols));w[winner]=1
    return w,dict(audit,r_squared=r2.tolist(),alpha=b[0].tolist())

def load_new(symbols):
    frames={};sources={}
    for symbol in symbols:
        f,m=read_frame(OUT/'data'/(symbol+'.parquet'));frames[symbol]=f;sources[symbol]=m
    return frames,sources

def target(rule,p,j,prepared):
    w=np.zeros(len(p.symbols));audit={}
    if rule=='MomentumLowVolCombo':return .5*(signed_target('LSMomentum',j,p,prepared)+signed_target('LSLowVol',j,p,prepared)),audit
    if rule=='OptimizedContrarian':return optimized_weights(p,j),audit
    if rule=='SelectivityAlpha':return selectivity(p,j,*prepared)
    if rule=='ShortLeveragedPair':w[:3]=[-.25,-.25,.5]
    elif rule.startswith('HP_'):w[0]=prepared[j]
    elif rule=='REITMixed':w[:2]=[.8,.2]
    elif rule=='REITGeography':w[:2]=[.5,.5]
    elif rule=='BuyHold':w[p.symbols.index('SPY') if 'SPY' in p.symbols else 0]=1
    else:raise ValueError(rule)
    return w,audit

def simulate(p,rule,prepared,start,end,fee=.0005,slip=.0005,borrow=.02,delay=1,zero_cash=False):
    pp=replace(p,cash_rate=np.zeros_like(p.cash_rate)) if zero_cash else p
    signed=rule in ['OptimizedContrarian','ShortLeveragedPair','MomentumLowVolCombo']
    ledger=SignedLedger(pp,start,end,fee,slip,borrow) if signed else Ledger(pp,start,end,fee,slip)
    decisions=[];pending={};last=None
    for t in range(start,end+1):
        # Monthly decision schedules independent of execution delay.
        due=t==start or (rule in ['SelectivityAlpha','ShortLeveragedPair','REITMixed','REITGeography','MomentumLowVolCombo'] and p.timestamp[t].month!=p.timestamp[t-1].month) or rule=='OptimizedContrarian' or rule.startswith('HP_')
        if due:
            j=t-1;w,a=target(rule,p,j,prepared)
            if rule=='BuyHold' and t!=start:w=None
            if rule.startswith('HP_') and last is not None and np.array_equal(w,last):w=None
            if w is not None:
                execute=t+delay-1
                if execute<=end:
                    if p.bar_end[j]>p.timestamp[execute]:raise ValueError('Future information')
                    pending[execute]=w;last=w.copy()
                    decisions.append({'decision':str(p.bar_end[j]),'execute':str(p.timestamp[execute]),'weights':w.tolist(),**a})
        w=pending.get(t)
        if signed:ledger.step(w)
        else:ledger.step(w,drip=rule=='BuyHold')
    h=pd.DataFrame(ledger.history);eq=h.equity.to_numpy();days=(p.bar_end[end]-p.timestamp[start]).total_seconds()/86400
    result={'return':float(eq[-1]/10000-1),'cagr':float((eq[-1]/10000)**(365.25/days)-1) if eq[-1]>0 else -1.,
        'max_drawdown':float((1-eq/np.maximum.accumulate(eq)).max()),'fees':ledger.fees,
        'slippage':ledger.slippage if signed else ledger.slip_paid,'borrow':ledger.borrow_paid if signed else 0.,
        'interest':ledger.interest,'fills':len(ledger.fills),'bankrupt':ledger.bankrupt if signed else False,
        'collateral_liquidations':ledger.margin_liquidations if signed else 0,
        'start':str(p.timestamp[start]),'end':str(p.bar_end[end])}
    return result,h,ledger.fills,decisions

def main(only=None):
    protocol=OUT/'prices_protocol.json'
    if not protocol.exists():save(protocol,{'rules':RULES,'windows':BLOCKS,'cases':['base','double_cost','extra_day','zero_cash','borrow_10pct'],
       'capital':10000,'base_fee_bps':5,'base_slippage_bps':5,'borrow_apr':.02,
       'qualification':'Must pass higher after-cost return and no worse daily-close maximum drawdown in full sample and all three chronological blocks, plus full-sample cost/delay/cash/borrow stresses. Revised public data, proxy universes and absent short locates prevent live-execution qualification.'})
    jobs,old_sources=load_jobs();_,sector,_,_=next(j for j in jobs if j[0]=='Sectors')
    markets={'Sectors':sector};sources={'Sectors':{k:v for k,v in old_sources.items() if k.startswith('sector_')}}
    for name,symbols in {'LETF':['SPXL','SPXS','SHY','SPY'],**{s:[s] for s in ['FXE','FXB','FXY','FXA']},'REITMix':['SPY','VNQ'],'REITGeo':['VNQ','VNQI']}.items():
        markets[name],sources[name]=load_new(symbols)
    factors,factor_source=load_factors();rows=[];errors=[];prefix=[]
    if only:
        save(OUT/('prices_protocol_'+only+'.json'),{'rule':RULES[only],'costs_windows_and_execution':'Unchanged from prices_protocol.json','registered_before_scoring':True})
        rows=[r for r in json.loads((OUT/'prices_results.json').read_text()) if r['strategy']!=only]
        errors=[r for r in json.loads((OUT/'prices_errors.json').read_text()) if r['strategy']!=only]
        prefix=[r for r in json.loads((OUT/'prices_prefix_checks.json').read_text()) if r['strategy']!=only]
    for market,frames in markets.items():
        p=panel_from(frames,True,252);rules=[r for r,x in RULES.items() if x['market']==market and (only is None or r==only)]
        for rule in rules:
            prepared=(monthly_returns(p),factors) if rule=='SelectivityAlpha' else hp_signal(p.close[:,0]) if rule.startswith('HP_') else signed_features(p,frames) if rule=='MomentumLowVolCombo' else None
            # Prefix invariance: later observations cannot change a historical decision.
            j=int(np.flatnonzero(p.timestamp>=pd.Timestamp('2020-01-01',tz='UTC'))[0])
            if rule.startswith('HP_'):
                assert np.array_equal(prepared[:j+1],hp_signal(p.close[:j+1,0]))
            else:
                truncated=replace(p,timestamp=p.timestamp[:j+2],bar_end=p.bar_end[:j+2],opening=p.opening[:j+2],close=p.close[:j+2],dividend=p.dividend[:j+2],total_index=p.total_index[:j+2],volatility=p.volatility[:j+2],cash_rate=p.cash_rate[:j+2])
                assert np.allclose(target(rule,p,j,prepared)[0],target(rule,truncated,j,prepared)[0])
            prefix.append({'strategy':rule,'passed':True,'prefix_bar':j})
            for window,left,right in BLOCKS:
                ix=np.flatnonzero((p.timestamp>=pd.Timestamp(left,tz='UTC'))&(p.timestamp<pd.Timestamp(right,tz='UTC')));start=max(254,int(ix[0]));end=int(ix[-1])
                cases={'base':{}}
                if window=='full':cases.update(double_cost={'fee':.001,'slip':.001},extra_day={'delay':2},zero_cash={'zero_cash':True},borrow_10pct={'borrow':.1})
                for case,args in cases.items():
                    try:
                        m,h,fills,decisions=simulate(p,rule,prepared,start,end,**args)
                        b,bh,bfills,_=simulate(p,'BuyHold',None,start,end,**args)
                        row={'section':RULES[rule]['section'],'strategy':rule,'market':market,'window':window,'case':case,**m,
                            'benchmark_return':b['return'],'benchmark_max_drawdown':b['max_drawdown'],
                            'pass':bool(m['return']>b['return'] and m['max_drawdown']<=b['max_drawdown'] and not m['bankrupt'])}
                        rows.append(row)
                        if window=='full' and case=='base':
                            folder=OUT/'price_ledgers';folder.mkdir(exist_ok=True)
                            h.to_parquet(folder/(rule+'.parquet'),index=False);bh.to_parquet(folder/(rule+'_benchmark.parquet'),index=False)
                            save(folder/(rule+'_fills.json'),fills);save(folder/(rule+'_benchmark_fills.json'),bfills);save(folder/(rule+'_decisions.json'),decisions)
                    except (ValueError,ArithmeticError) as e:errors.append({'strategy':rule,'window':window,'case':case,'reason':str(e)})
            print('Completed',rule,'cases',sum(x['strategy']==rule for x in rows),flush=True)
    save(OUT/'prices_results.json',rows);save(OUT/'prices_errors.json',errors);save(OUT/'prices_prefix_checks.json',prefix)
    save(OUT/'prices_manifest.json',{'runs':len(rows),'benchmark_runs':len(rows),'failed_closed':len(errors),'rules':RULES,
       'sources':sources,'factors':factor_source,'code_sha256':sha(__file__),'protocol_sha256':sha(protocol)})
if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else None)
