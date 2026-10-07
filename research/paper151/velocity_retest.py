"""Frozen, causal adaptations of saved PatternFindr daily recipes.

Signal mathematics are copied from the source projects. Execution is deliberately
stricter: next open, gap-aware stops, prior-bar trailing levels, daily marked equity.
This is not claimed to reproduce earlier optimistic report percentages.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from cross_project_inventory import OUT, PF, snapshot_functions, save, sha
from run_study import read_frame, panel_from
from tensortrade_lab.portfolio import Ledger, performance

def module(name):
    p=OUT/'sources'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

SUPPORTED_FLAGS={'use_accel_exit','use_jerk_confirm','use_trailing_stop','use_breakeven_stop',
                 'use_macd_confirm','use_bb_filter','use_next_bar_entry'}
BLOCKS=[('full','2017-01-01','2026-09-24',1,1,False),
        ('2017_2019','2017-01-01','2020-01-01',1,1,False),
        ('2020_2022','2020-01-01','2023-01-01',1,1,False),
        ('2023_latest','2023-01-01','2026-09-24',1,1,False),
        ('2026_Apr_Sep','2026-04-01','2026-09-24',1,1,False),
        ('double_cost','2017-01-01','2026-09-24',2,1,False),
        ('delay_two_bars','2017-01-01','2026-09-24',1,2,False),
        ('no_cash_interest','2017-01-01','2026-09-24',1,1,True)]

def exclusion(d):
    bad=[k for k,v in d.items() if k.startswith('use_') and v is True and k not in SUPPORTED_FLAGS]
    if bad:return 'Unsupported auxiliary rules: '+', '.join(bad)
    if any(d.get(k,0) for k in ['oz_wall_bias','oz_extreme_boost']):return 'Requires historical options-zone inputs'
    if d.get('signal_type','any_reversal') not in ['any_reversal','velocity_crossover_or_zone',
            'velocity_crossover_and_zone','zone_only','momentum','divergence','double_bottom','breakout']:
        return 'Unknown signal type'
    return None

def oscillator(f,d):
    kind=d.get('oscillator_type','composite')
    if kind in ('composite','composite_smooth'):
        if d.get('optimization_method',{}).get('script')=='velocity_walkforward_validation.py' or d.get('oscillator_type'):
            return module('pf_oscillators').create_composite_oscillator_features(f).osc_composite_smooth
        return module('pf_legacy').create_composite_oscillator(f).composite_smooth
    fun=getattr(module('pf_novel'),'calculate_'+kind,None)
    if fun is None:raise ValueError('Unsupported oscillator '+kind)
    return fun(f)

def make_signals(f,d,osc=None):
    osc=oscillator(f,d) if osc is None else osc
    o=osc.rolling(int(d.get('vel_smoothing',1))).mean()
    v=o.diff();a=v.diff();jerk=a.diff()
    up=(v>0)&(v.shift(1)<=0);down=(v<0)&(v.shift(1)>=0)
    os=o<d.get('oversold_threshold',-.2);ob=o>d.get('overbought_threshold',.2)
    mult=d.get('extreme_zone_mult',2.)
    eos=o<d.get('oversold_threshold',-.2)*mult;eob=o>d.get('overbought_threshold',.2)*mult
    st=v.rolling(int(d.get('velocity_std_window',10)),min_periods=2).std()
    mu=d.get('momentum_multiplier',1.5);strongup=v>st*mu;strongdown=v<-st*mu
    kind=d.get('signal_type','any_reversal')
    if kind=='any_reversal':buy=up|eos|(strongup&os);sell=down|eob|(strongdown&ob)
    elif kind=='velocity_crossover_or_zone':buy=up|eos;sell=down|eob
    elif kind=='velocity_crossover_and_zone':buy=up&os;sell=down&ob
    elif kind=='zone_only':buy=eos&(v>0);sell=eob&(v<0)
    elif kind=='momentum':buy=strongup&(o<0);sell=strongdown&(o>0)
    elif kind=='breakout':
        buy=(o>d.get('oversold_threshold',-.2))&(o.shift(1)<=d.get('oversold_threshold',-.2))
        sell=(o<d.get('overbought_threshold',.2))&(o.shift(1)>=d.get('overbought_threshold',.2))
    elif kind=='divergence':
        n=int(d.get('divergence_lookback',5))
        buy=(f.close<f.close.rolling(n).min().shift(1))&(o>o.rolling(n).min().shift(1))&os
        sell=(f.close>f.close.rolling(n).max().shift(1))&(o<o.rolling(n).max().shift(1))&ob
    elif kind=='double_bottom':
        n=int(d.get('double_bottom_lookback',10));buy=(up.rolling(n).sum()>=2)&os;sell=(down.rolling(n).sum()>=2)&ob
    else:raise ValueError(kind)
    if d.get('require_accel'):buy&=a>0;sell&=a<0
    rsi_filter=d.get('rsi_filter','none')
    if rsi_filter!='none':
        delta=f.close.diff();n=int(d.get('rsi_period',14))
        gain=delta.clip(lower=0).rolling(n).mean();loss=(-delta.clip(upper=0)).rolling(n).mean()
        rsi=100-100/(1+gain/loss)
        if rsi_filter in ('oversold_only','both'):buy&=rsi<d.get('rsi_oversold',30)
        if rsi_filter in ('overbought_only','both'):sell&=rsi>d.get('rsi_overbought',70)
    if d.get('use_macd_confirm'):
        macd=f.close.ewm(span=12,adjust=False).mean()-f.close.ewm(span=26,adjust=False).mean()
        hist=macd-macd.ewm(span=9,adjust=False).mean();buy&=hist>hist.shift(1);sell&=hist<hist.shift(1)
    if d.get('use_bb_filter'):
        ma=f.close.rolling(20).mean();sd=f.close.rolling(20).std();buy&=f.close<ma-2*sd;sell&=f.close>ma+2*sd
    # No startup backfill or full-sample standard deviation. Warmup excludes these bars.
    return pd.DataFrame({'buy':buy.fillna(False),'sell':sell.fillna(False),'osc':o,'accel':a,'jerk':jerk})

def close_exit(d,s,j,entry,entry_bar,f):
    if j-entry_bar<int(d.get('min_hold_bars',1)):return False
    pnl=(f.close.iloc[j]/entry-1)*100
    if d.get('use_accel_exit') and (pnl>=d.get('accel_exit_min_pnl',.5) or pnl<0):
        n=int(d.get('accel_exit_lookback',1));a=s.accel.iloc[j];kind=d.get('accel_exit_type','sign_reversal')
        negative=(s.accel.iloc[max(0,j-n+1):j+1]<0).all()
        trigger={'sign_reversal':negative,'magnitude':a<-d.get('accel_exit_threshold',0),
                 'both':negative and abs(a)>d.get('accel_exit_threshold',0)}[kind]
        if d.get('use_jerk_confirm') and d.get('jerk_confirm_threshold',0)>0:
            trigger=trigger and s.jerk.iloc[j]<-d['jerk_confirm_threshold']
        if trigger:return True
    return bool((d.get('exit_on_opposite_signal',True) and s.sell.iloc[j]) or
                (d.get('exit_on_midline_cross',False) and s.osc.iloc[j]>0))

def protective_fill(opening,high,low,entry,previous_high,d):
    """Old orders only. Gaps fill at open; ambiguous stop+target bars lose first."""
    stop=entry*(1-d.get('stop_loss_pct',5)/100)
    target=entry*(1+d.get('take_profit_pct',10)/100)
    gain=(previous_high/entry-1)*100
    if d.get('use_trailing_stop') and gain>=d.get('trailing_stop_activation_pct',.3):
        stop=max(stop,previous_high*(1-d.get('trailing_stop_pct',1)/100))
    if d.get('use_breakeven_stop') and gain>=d.get('breakeven_trigger_pct',.3):
        stop=max(stop,entry*(1+d.get('breakeven_offset_pct',.05)/100))
    if opening<=stop:return opening,'stop_gap'
    if opening>=target:return opening,'target_gap'
    if low<=stop:return stop,'stop'
    if high>=target:return target,'target'
    return None,None

def run(p,f,s,d,start,end,fee,slip,delay=1,benchmark=False,cash_only=False):
    ledger=Ledger(p,start,end,fee,slip)
    entry=0.;entry_bar=-1;hwm=0.;last_exit=-10000;entries=0
    for t in range(start,end+1):
        j=t-delay;weights=None
        assert p.bar_end[j]<=p.timestamp[t]
        was_held=ledger.shares[0]>1e-10
        if benchmark:
            if t==start:weights=np.array([0. if cash_only else 1.])
        elif was_held:
            if close_exit(d,s,j,entry,entry_bar,f):weights=np.array([0.]);last_exit=t
        elif s.buy.iloc[j] and j-last_exit>=int(d.get('min_bars_between',1)):
            weights=np.array([1.])
        ledger.step(weights,drip=benchmark and not cash_only,liquidate=False)
        held=ledger.shares[0]>1e-10
        if held and not was_held:
            entry=f.open.iloc[t]*(1+slip);entry_bar=t;hwm=entry;entries+=1
        if held and not benchmark:
            price,reason=protective_fill(f.open.iloc[t],f.high.iloc[t],f.low.iloc[t],entry,hwm,d)
            if price is not None:
                ledger._execute(-ledger.shares,np.array([price]),reason)
                last_exit=t
            else:hwm=max(hwm,f.high.iloc[t])
        if t==end and ledger.shares[0]>0:
            ledger._execute(-ledger.shares,p.close[t],'terminal_liquidation')
        ledger.history.pop();ledger.record()
    metrics=performance(ledger);metrics['entries']=entries
    return metrics,pd.DataFrame(ledger.history),ledger.fills

def prepare_sources():
    names={'calculate_rsi','calculate_williams_r','calculate_cci','calculate_stochastic','calculate_roc',
           'calculate_momentum','calculate_bb_position','calculate_adx_trend','calculate_mfi','create_composite_oscillator'}
    return snapshot_functions(PF/'oscillator_predictor_page.py','pf_legacy.py',names)

def main():
    from dataclasses import replace
    prepare_sources()
    configs=json.loads((OUT/'patternfindr_configs.json').read_text())
    rows=[];status=[];causality=[];sources={};cache={};bench={}
    for symbol,path in [('SPY','data/rotation/SPY.parquet'),('BTC-USD','research/paper151/data/BTCUSD_yahoo_20260929.parquet')]:
        f,meta=read_frame(path);f=f[f.timestamp<pd.Timestamp('2026-09-24',tz='UTC')].reset_index(drop=True)
        p=panel_from({symbol:f},cash_interest=symbol=='SPY',periods=252 if symbol=='SPY' else 365)
        sources[symbol]=meta;cache[symbol]=(f,p)
    protocol={'registered_at':datetime.now(timezone.utc).isoformat(),'blocks':BLOCKS,
       'sample':'Retrospective. April-Sep 2026 is a post-configuration diagnostic, NOT an untouched holdout.',
       'selection':'Every saved daily BTC/SPY recipe with supported observable inputs; no reoptimization.',
       'execution':'Long-only, unlevered, next open; standing gap-aware stops/targets; trailing levels use only prior bars. Stop-first if both touched.',
       'source_resolution':'Use research composite for saved optimization_method/explicit oscillator; use legacy production composite for older unspecified oscillator. composite and composite_smooth follow saved walkforward fallback. Divergence uses saved lookback in a causal implementation.',
       'costs':'SPY 5bp fee +5bp slippage per side, BTC 10bp fee +5bp slippage. Both benchmark sides charged. Fractional units.',
       'qualification':'Full sample, every three-year block, double costs, extra-bar delay and no-interest comparison must all beat matched buy-hold return with no worse daily maximum drawdown, plus beat cash. April-Sep separately reported.'}
    save('velocity_protocol.json',protocol)
    for symbol,(f,p) in cache.items():
        for name,begin,finish,mult,delay,zero in BLOCKS:
            ix=np.where((p.timestamp>=pd.Timestamp(begin,tz='UTC'))&(p.timestamp<pd.Timestamp(finish,tz='UTC')))[0]
            start=max(int(ix[0]),253);end=int(ix[-1]);pp=replace(p,cash_rate=np.zeros_like(p.cash_rate)) if zero else p
            fee=(.0005 if symbol=='SPY' else .001)*mult;slip=.0005*mult
            for label,cash in [('BuyHold',False),('Cash',True)]:
                m,_,_=run(pp,f,None,{},start,end,fee,slip,benchmark=True,cash_only=cash)
                bench[(symbol,name,label)]=m
                rows.append({'origin':'benchmark','strategy':label,'market':symbol,'case':name,**m})
    for row in configs:
        if not row['valid']:status.append({'strategy':row['id'],'status':'invalid_config'});continue
        d=row['config'];symbol=d.get('ticker')
        if d.get('interval')!='1d' or symbol not in cache:
            status.append({'strategy':row['id'],'status':'needs_separate_intraday_or_contract_test'});continue
        reason=exclusion(d)
        if reason:status.append({'strategy':row['id'],'status':'blocked_auxiliary_inputs','reason':reason});continue
        f,p=cache[symbol]
        try:
            osc=oscillator(f,d);s=make_signals(f,d,osc)
            # Truncate the actual observations, independently recalculate, and compare.
            cut=len(f)-137;short=make_signals(f.iloc[:cut].copy(),d)
            ok=np.allclose(s.iloc[253:cut].to_numpy(float),short.iloc[253:].to_numpy(float),equal_nan=True,rtol=1e-10,atol=1e-10)
            causality.append({'strategy':row['id'],'prefix_bars':cut,'passed':bool(ok)})
            if not ok:raise ValueError('Prefix invariance failed')
        except Exception as e:
            status.append({'strategy':row['id'],'status':'blocked_calculation','reason':str(e)});continue
        for name,begin,finish,mult,delay,zero in BLOCKS:
            ix=np.where((p.timestamp>=pd.Timestamp(begin,tz='UTC'))&(p.timestamp<pd.Timestamp(finish,tz='UTC')))[0]
            start=max(int(ix[0]),253);end=int(ix[-1]);pp=replace(p,cash_rate=np.zeros_like(p.cash_rate)) if zero else p
            fee=(.0005 if symbol=='SPY' else .001)*mult;slip=.0005*mult
            m,hist,fills=run(pp,f,s,d,start,end,fee,slip,delay)
            b=bench[(symbol,name,'BuyHold')];c=bench[(symbol,name,'Cash')]
            rows.append({'origin':'PatternFindr_adaptation','strategy':row['id'],'market':symbol,'case':name,
                **m,'benchmark_return':b['total_return'],'benchmark_drawdown':b['max_drawdown'],
                'cash_return':c['total_return'],'pass':bool(m['total_return']>b['total_return'] and m['max_drawdown']<=b['max_drawdown']),
                'beats_cash':bool(m['total_return']>c['total_return'])})
            if name=='full':
                dest=OUT/'velocity_ledgers';dest.mkdir(exist_ok=True)
                hist.to_parquet(dest/(row['id']+'.parquet'),index=False)
                save('velocity_ledgers/'+row['id']+'_fills.json',fills)
        status.append({'strategy':row['id'],'status':'historically_tested_adaptation','source_sha256':row['source_sha256']})
        print('tested',row['id'],flush=True)
        save('velocity_progress.json',{'tested':sum(x['status']=='historically_tested_adaptation' for x in status)})
    save('velocity_results.json',rows);save('velocity_status.json',status);save('velocity_causality.json',causality)
    save('velocity_manifest.json',{'asof':datetime.now(timezone.utc).isoformat(),'sources':sources,
        'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'velocity_protocol.json'),
        'recipes_tested':sum(x['status']=='historically_tested_adaptation' for x in status),
        'runs':len(rows),'prefix_checks':len(causality),'checks_passed':sum(x['passed'] for x in causality)})
    print('DONE',len(rows),'runs',flush=True)

if __name__=='__main__':main()
