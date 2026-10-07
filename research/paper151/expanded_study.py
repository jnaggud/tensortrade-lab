"""Frozen expanded screen. No live orders; original snapshots are read-only."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import sys
import warnings
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(HERE))
from tensortrade_lab.portfolio import Ledger
from run_study import read_frame, panel_from, SECTORS

OUT = HERE/'expanded_results'
SINGLE = ['SMA200','MA50_200','MA20_50_200','Momentum252','Donchian55_20',
          'ChannelBounce20','PivotBounce','IBS','Vol12','TrendVol12','AlphaCombo','KNN']
PORTFOLIO = ['SectorRotation','SectorRotationMA','SectorRotationDual','SectorMomentumSkip21',
             'SectorLowVol','SectorIBS','SectorContrarian']
MULTI = ['MultiTrend','MultiTrendVol','MultiTrendVar','GoldDiversification']
NATIVE = set(SINGLE)-{'KNN'}
BLOCKS = [('full','2017-01-01','2026-09-24'),('2017_2019','2017-01-01','2020-01-01'),
          ('2020_2022','2020-01-01','2023-01-01'),('2023_latest','2023-01-01','2026-09-24')]

def model_features(f, p):
    # Predictors measured strictly before the decision bar, as in §3.17.
    knn = np.column_stack([f.close.rolling(k).mean().shift(1) for k in (5,20,60)]
                          + [f.volume.rolling(20).mean().shift(1)])
    tr = pd.Series(p.total_index[:,0])
    ann = np.column_stack([tr.pct_change(k).shift(1) for k in (1,5,21,63)])
    y = np.full(len(f), np.nan)
    y[:-2] = (f.open.to_numpy()[2:] + f.dividend.to_numpy()[2:]) / f.open.to_numpy()[1:-1] - 1
    return knn, ann, y

def causal_knn(x, y, cost, lookback=756, neighbors=21, minimum=126):
    pred=np.full(len(x),np.nan)
    for j in range(minimum+65,len(x)):
        ix=np.arange(max(0,j-lookback), j-1)  # i+2<=j: label is already known
        valid=np.isfinite(x[ix]).all(axis=1)&np.isfinite(y[ix])
        ix=ix[valid]
        if len(ix)<minimum or not np.isfinite(x[j]).all():continue
        xx=x[ix]; scale=np.maximum(np.ptp(xx,axis=0),1e-12)
        distance=np.square((xx-x[j])/scale).sum(axis=1)
        near=np.argsort(distance,kind='stable')[:neighbors]
        pred[j]=y[ix[near]].mean()
    return (pred>cost).astype(float), pred

def causal_ann(x,y,dates):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    pred=np.full(len(x),np.nan); audits=[]; model=scaler=None; last_year=None
    for j in range(316,len(x)):
        year=dates[j].year
        if year!=last_year:
            ix=np.arange(max(0,j-1095), j-1)
            ix=ix[np.isfinite(x[ix]).all(axis=1)&np.isfinite(y[ix])]
            if len(ix)>=252 and len(np.unique(y[ix]>0))==2:
                scaler=StandardScaler().fit(x[ix])
                model=MLPClassifier(hidden_layer_sizes=(8,),activation='tanh',solver='adam',
                    alpha=0.01,learning_rate_init=0.001,max_iter=500,random_state=17,shuffle=False,
                    early_stopping=False)
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always');model.fit(scaler.transform(x[ix]),y[ix]>0)
                audits.append({'fit_at':str(dates[j]),'max_training_label_known_at':str(dates[ix[-1]+2]),
                    'training_rows':len(ix),'iterations':model.n_iter_,'warnings':[str(w.message) for w in ws]})
                last_year=year
        if model is not None and np.isfinite(x[j]).all():
            pred[j]=model.predict_proba(scaler.transform(x[j:j+1]))[0,1]
    return (pred>0.55).astype(float),audits

def prepare(f,p,crypto=False,models=True):
    c=f.close; tr=pd.Series(p.total_index[:,0]); ret=tr.pct_change().fillna(0)
    d={f'ma{k}':c.rolling(k).mean().to_numpy() for k in (10,20,40,50,200)}
    for k in (10,20,55):
        d[f'hi{k}']=f.high.rolling(k).max().shift(1).to_numpy()
        d[f'lo{k}']=f.low.rolling(k).min().shift(1).to_numpy()
    d['momentum']=(tr>tr.shift(252)).to_numpy()
    d['ibs']=np.divide(c-f.low,f.high-f.low,out=np.full(len(f),0.5),where=f.high!=f.low)
    pivot=(f.high.shift(1)+f.low.shift(1)+c.shift(1))/3
    d['support']=(2*pivot-f.high.shift(1)).to_numpy()
    d['resistance']=(2*pivot-f.low.shift(1)).to_numpy()
    d['vol']=ret.rolling(63).std().to_numpy()*np.sqrt(365 if crypto else 252)
    d['volweight']=np.nan_to_num(np.minimum(1,0.12/np.maximum(d['vol'],1e-12)))
    d['returns']=ret.to_numpy()
    # Autonomous Donchian state for combo voting, rather than portfolio state.
    trend=False; states=[]
    for j in range(len(f)):
        if c.iloc[j]>d['hi55'][j]:trend=True
        elif c.iloc[j]<d['lo20'][j]:trend=False
        states.append(trend)
    d['channel_state']=np.array(states)
    audits=[]
    if models:
        x,xa,y=model_features(f,p)
        d['KNN'],d['knn_prediction']=causal_knn(x,y,0.003 if crypto else 0.002)
        if crypto:d['ANN'],audits=causal_ann(xa,y,p.timestamp)
    return d,audits

def single_target(rule,j,held,in_position,f,d):
    c=f.close.iloc[j]
    if rule=='SMA200': return float(c>d['ma200'][j])
    if rule=='MA50_200':return float(d['ma50'][j]>d['ma200'][j])
    if rule=='MA20_50_200':return float(d['ma20'][j]>d['ma50'][j]>d['ma200'][j])
    if rule=='Momentum252':return float(d['momentum'][j])
    if rule=='Vol12':return d['volweight'][j]
    if rule=='TrendVol12':return d['volweight'][j]*d['momentum'][j]
    if rule=='AlphaCombo':return float(int(c>d['ma200'][j])+int(d['momentum'][j])+int(d['channel_state'][j])>=2)
    if rule in ('KNN','ANN'):return d[rule][j]
    if rule=='Donchian55_20':
        if in_position:return float(not(c<d['lo20'][j]))
        return float(c>d['hi55'][j])
    if rule=='IBS':
        if in_position:return float(not(d['ibs'][j]>0.8 or held>=5))
        return float(d['ibs'][j]<0.2)
    if rule=='PivotBounce':
        if in_position:return float(not(c>=d['resistance'][j] or held>=5))
        return float(f.low.iloc[j]<=d['support'][j]<c)
    if rule=='ChannelBounce20':
        if in_position:return float(not(c>=d['hi20'][j] or held>=5))
        return float(f.low.iloc[j]<=d['lo20'][j]<c)
    raise ValueError(rule)

def portfolio_target(rule,j,p,frames):
    w=np.zeros(len(p.symbols)); c=p.close[j]; momentum=p.total_index[j]/p.total_index[j-252]-1
    ma=p.close[j-199:j+1].mean(axis=0)
    if rule=='GoldDiversification':
        w[p.symbols.index('SPY')]=0.8;w[p.symbols.index('GLD')]=0.2;return w
    if rule.startswith('MultiTrend'):
        power={'MultiTrend':0,'MultiTrendVol':1,'MultiTrendVar':2}[rule]
        raw=np.where((momentum>0)&(c>ma),momentum/np.maximum(p.volatility[j],0.01)**power,0)
        return raw/raw.sum() if raw.sum()>0 else w
    if rule=='SectorRotationDual' and c[p.symbols.index('SPY')]<=ma[p.symbols.index('SPY')]:
        w[p.symbols.index('IEF')]=1;return w
    if rule=='SectorMomentumSkip21':score=p.total_index[j-21,:9]/p.total_index[j-252,:9]-1
    elif rule=='SectorLowVol':score=-p.volatility[j,:9]
    elif rule=='SectorIBS':
        score=np.array([-(fr.close.iloc[j]-fr.low.iloc[j])/(fr.high.iloc[j]-fr.low.iloc[j]) if fr.high.iloc[j]!=fr.low.iloc[j] else -0.5 for fr in list(frames.values())[:9]])
    elif rule=='SectorContrarian':
        r=p.total_index[j,:9]/p.total_index[j-1,:9]-1
        raw=np.maximum(r.mean()-r,0)
        if raw.sum()>0:w[:9]=raw/raw.sum()
        return w
    else:score=momentum[:9]
    ix=np.argsort(-score,kind='stable')[:3]
    for k in ix:
        if rule!='SectorRotationMA' or c[k]>ma[k]:w[k]=1/3
    return w

def simulate(p,frames,d,rule,start,end,fee,slip,delay=1,zero_cash=False,record=False):
    # Isolate mutable rate choices from other cases.
    from dataclasses import replace
    pp=replace(p,cash_rate=np.zeros_like(p.cash_rate)) if zero_cash else p
    ledger=Ledger(pp,start,end,fee,slip)
    f=next(iter(frames.values()));held=0;last=None;decisions=[];peak=10000.;bound=0.
    benchmark=rule in ('BuyHold','PassiveBasket')
    for t in range(start,end+1):
        j=t-delay;w=None
        assert p.bar_end[j]<=p.timestamp[t], 'Signal bar must have completed before execution'
        if benchmark:
            if t==start:
                w=np.zeros(len(p.symbols))
                if rule=='BuyHold':w[p.symbols.index('SPY') if len(p.symbols)>1 else 0]=1
                else:
                    n=9 if len(p.symbols)==11 else len(p.symbols);w[:n]=1/n
        elif rule in SINGLE or rule=='ANN':
            in_position=ledger.shares[0]>1e-10;held=held+1 if in_position else 0
            value=single_target(rule,j,held,in_position,f,d)
            if value!=last or rule in ('Vol12','TrendVol12'):w=np.array([value]);last=value
        else:
            daily=rule in ('SectorIBS','SectorContrarian')
            if daily or t==start or p.timestamp[t].month!=p.timestamp[t-1].month:
                w=portfolio_target(rule,j,p,frames)
        if w is not None and record:
            decisions.append({'decided_at':str(p.bar_end[j]),'execute_at':str(p.timestamp[t]),'weights':w.tolist()})
        ledger.step(w,drip=benchmark,liquidate=False)
        # Reconstruct pre-close-reinvestment exposure on passive distribution days.
        cash=ledger.cash;units=ledger.shares.copy()
        for fill in ledger.fills[-len(p.symbols):]:
            if fill['reason']=='dividend_reinvestment' and fill['timestamp']==p.bar_end[t]:
                units[p.symbols.index(fill['symbol'])]-=fill['units'];cash+=fill['units']*fill['price']+fill['fee']
        high=cash+sum(units[k]*fr.high.iloc[t] for k,fr in enumerate(frames.values()))
        low=cash+sum(units[k]*fr.low.iloc[t] for k,fr in enumerate(frames.values()))
        peak=max(peak,high,ledger.equity);bound=max(bound,1-low/peak,1-ledger.equity/peak)
        if t==end:
            ledger._execute(-ledger.shares,p.close[t],'terminal_liquidation');ledger.history.pop();ledger.record()
            bound=max(bound,1-ledger.equity/peak)
    hist=pd.DataFrame(ledger.history);eq=hist.equity.to_numpy()
    days=(p.bar_end[end]-p.timestamp[start]).total_seconds()/86400
    metrics={'return':eq[-1]/10000-1,'cagr':(eq[-1]/10000)**(365.25/days)-1,
        'max_drawdown':float((1-eq/np.maximum.accumulate(eq)).max()),'ohlc_drawdown_bound':bound,
        'fills':len(ledger.fills),'fees':ledger.fees,'slippage':ledger.slip_paid,'dividends':ledger.dividends,
        'interest':ledger.interest,'turnover':ledger.turnover/10000,'days':days}
    return metrics,hist,ledger.fills,decisions

def load_jobs():
    jobs=[];manifest={}
    paths={'SPY':'data/rotation/SPY.parquet','QQQ':'data/yahoo/QQQ_1d.parquet',
           'AAPL':'data/yahoo/AAPL_1d.parquet','TLT':'data/rotation/TLT.parquet',
           'GLD':'data/rotation/GLD.parquet','EFA':'data/rotation/EFA.parquet',
           'BTC':'research/paper151/data/BTCUSD_yahoo_20260929.parquet'}
    for symbol,path in paths.items():
        f,m=read_frame(path);manifest[symbol]=m
        end='2026-09-21' if symbol=='EFA' else '2026-09-23'
        f=f[f.timestamp.dt.date<=pd.Timestamp(end).date()].reset_index(drop=True)
        jobs.append((symbol,{symbol:f},SINGLE+(['ANN'] if symbol=='BTC' else []),symbol=='BTC'))
    frames={}
    for symbol in SECTORS+['SPY','IEF']:
        f,m=read_frame(f'data/rotation/{symbol}.parquet');manifest['sector_'+symbol]=m
        frames[symbol]=f[f.timestamp.dt.date<=pd.Timestamp('2026-09-21').date()].reset_index(drop=True)
    jobs.append(('Sectors',frames,PORTFOLIO,False))
    frames={}
    for symbol in ['SPY','EFA','EEM','TLT','IEF','GLD']:
        f,m=read_frame(f'data/rotation/{symbol}.parquet');manifest['multi_'+symbol]=m
        frames[symbol]=f[(f.timestamp.dt.date>=pd.Timestamp('2004-11-18').date()) & (f.timestamp.dt.date<=pd.Timestamp('2026-09-21').date())].reset_index(drop=True)
    jobs.append(('MultiAsset',frames,MULTI,False))
    return jobs,manifest

def add_benchmarks(rows):
    result=pd.DataFrame(rows)
    b=result[result.strategy=='BuyHold'][['market','window','case','return','cagr','max_drawdown','ohlc_drawdown_bound']]
    result=result.merge(b,on=['market','window','case'],suffixes=('','_bh'),validate='many_to_one')
    result['excess_cagr']=result.cagr-result.cagr_bh
    result['drawdown_deterioration']=result.max_drawdown-result.max_drawdown_bh
    result['pass']=(result['return']>result.return_bh)&(result.max_drawdown<=result.max_drawdown_bh)
    return result

def rank(result):
    rows=[]
    for name,g in result[~result.strategy.isin(['BuyHold','PassiveBasket'])].groupby('strategy'):
        blocks=g[(g.window!='full')&(g['case']=='base')]
        stress=g[(g.window=='full')&(g['case']=='double_cost')]
        full=g[(g.window=='full')&(g['case']=='base')]
        portable=name in NATIVE or name in PORTFOLIO or name in MULTI
        rows.append({'strategy':name,'markets':len(full),'scored_blocks':len(blocks),
            'qualified':bool(len(blocks)>0 and blocks['pass'].all() and stress['pass'].all()),
            'block_pass_fraction':float(blocks['pass'].mean()),'median_excess_cagr':float(blocks.excess_cagr.median()),
            'worst_excess_cagr':float(blocks.excess_cagr.min()),
            'worst_drawdown_deterioration':float(blocks.drawdown_deterioration.max()),
            'double_cost_pass_fraction':float(stress['pass'].mean()),'full_pass_fraction':float(full['pass'].mean()),
            'pine_compatible':portable,'pine_kind':'native_strategy' if name in NATIVE else 'portfolio_signal_indicator' if portable else 'model_port_pending'})
    rankings=pd.DataFrame(rows).sort_values(['qualified','block_pass_fraction','median_excess_cagr','worst_drawdown_deterioration','strategy'],ascending=[False,False,False,True,True])
    rankings.insert(0,'rank',np.arange(1,len(rankings)+1))
    return rankings

def main():
    OUT.mkdir(exist_ok=True);(OUT/'ledgers').mkdir(exist_ok=True)
    jobs,manifest=load_jobs();allrows=[];audits={}
    for market,frames,rules,crypto in jobs:
        p=panel_from(frames,cash_interest=not crypto,periods=365 if crypto else 252)
        d,audit=prepare(next(iter(frames.values())),p,crypto) if len(frames)==1 else ({},[])
        audits[market]=audit
        benchmarks=['BuyHold']+(['PassiveBasket'] if len(frames)>1 else [])
        for window,a,b in BLOCKS:
            ix=np.flatnonzero((p.timestamp>=pd.Timestamp(a,tz='UTC'))&(p.timestamp<pd.Timestamp(b,tz='UTC')))
            start,end=max(254,int(ix[0])),int(ix[-1])
            cases=[('base',1,1,False)]
            if window=='full':cases+=[('double_cost',2,1,False),('extra_bar_delay',1,2,False),('zero_cash_yield',1,1,True)]
            for case,cost,delay,zero in cases:
                for name in benchmarks+rules:
                    keep=window=='full' and case=='base'
                    m,h,fills,decisions=simulate(p,frames,d,name,start,end,(0.001 if crypto else 0.0005)*cost,0.0005*cost,delay,zero,keep)
                    allrows.append({'market':market,'strategy':name,'window':window,'case':case,'start':str(p.timestamp[start]),'end':str(p.bar_end[end]),**m})
                    if keep:
                        key=market+'__'+name
                        h.to_parquet(OUT/'ledgers'/f'{key}.parquet',index=False)
                        (OUT/'ledgers'/f'{key}.json').write_text(json.dumps({'fills':fills,'decisions':decisions},default=str))
        print('Completed',market,'recipes',len(rules),flush=True)
        pd.DataFrame(allrows).to_json(OUT/'progress.json',orient='records',indent=2)
    results=add_benchmarks(allrows);rankings=rank(results)
    results.to_json(OUT/'metrics.json',orient='records',indent=2)
    rankings.to_json(OUT/'ranking.json',orient='records',indent=2)
    promoted=rankings[rankings.pine_compatible].head(10).to_dict('records')
    (OUT/'promotion.json').write_text(json.dumps(promoted,indent=2))
    manifest={'sources':manifest,'protocol_sha256':hashlib.sha256((HERE/'EXPANDED_PROTOCOL.md').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_fits':audits,
        'disclosure':'Retrospective screening, no untouched holdout. Portfolio and single-instrument universes differ; block denominators are shown.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    print(rankings[['rank','strategy','qualified','block_pass_fraction','median_excess_cagr','pine_kind']].to_string(index=False),flush=True)

if __name__=='__main__':main()
