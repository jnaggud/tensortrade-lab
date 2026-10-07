"""Causal adaptations shared by this new batch; original frozen studies unchanged."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,importlib.util,sys,ast
import numpy as np
import pandas as pd
from velocity_retest import make_signals,oscillator,protective_fill
from cross_project_inventory import PF,QUANT,sanitize
HERE=Path(__file__).resolve().parent;OUT=HERE/'next_batch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(sanitize(x),indent=2,default=str,allow_nan=False)+'\n')
def pure_copy(path,name,names):
    src=Path(path).read_text();tree=ast.parse(src);parts=[ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    p=OUT/'sources'/name;p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text('from __future__ import annotations\nimport pandas as pd\nimport numpy as np\nfrom scipy import stats\n'+ '\n\n'.join(parts))
    spec=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    save(name+'.source.json',{'original':str(path),'source_sha256':sha(path),'copy_sha256':sha(p),'functions':sorted(names)})
    return m

def causal_rsc(f,lookback=100):
    """Past-window clustering; current observation predicted, never backfilled."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    returns=f.close.pct_change();feat=pd.DataFrame({'return':returns,'vol':returns.rolling(20).std(),'momentum':f.close.pct_change(20)}).dropna()
    out=pd.Series(0.,index=f.index);model=None;mapping=None
    with threadpool_limits(limits=1):
        for i in range(lookback,len(feat)):
            if (i-lookback)%10==0:
                model=KMeans(n_clusters=3,random_state=42,n_init=10).fit(feat.iloc[i-lookback:i].to_numpy())
                mapping={int(k):j-1 for j,k in enumerate(np.argsort(model.cluster_centers_[:,2]))}
            out.loc[feat.index[i]]=mapping[int(model.predict(feat.iloc[i:i+1].to_numpy())[0])]
    return out

def pf_signals(f,d):
    if d.get('regime_aware') or d.get('regime_strategies'):raise ValueError('Nested regime state machine needs separate implementation')
    if d.get('use_wavelet_denoise'):raise ValueError('Full-series wavelet revises history; cannot faithfully use as a trading signal')
    allowed={'use_accel_exit','use_jerk_confirm','use_trailing_stop','use_breakeven_stop','use_macd_confirm','use_bb_filter','use_next_bar_entry','use_regime_filter','use_fragility_filter','use_entropy_filter'}
    bad=[k for k,v in d.items() if k.startswith('use_') and v is True and k not in allowed]
    if bad:raise ValueError('Unsupported auxiliary rules: '+','.join(bad))
    s=make_signals(f,d)
    velocity=s.osc.diff()
    if d.get('vel_threshold',0)>0:s.loc[abs(velocity)<d['vel_threshold'],['buy','sell']]=False
    if d.get('require_accel') and d.get('accel_threshold',0)>0:s.loc[abs(s.accel)<d['accel_threshold'],['buy','sell']]=False
    if d.get('use_regime_filter'):s['buy']&=causal_rsc(f)>d.get('regime_threshold',0)
    if d.get('use_fragility_filter') or d.get('use_entropy_filter'):
        m=pure_copy(PF/'novel_indicators_v2.py','pf_filters.py',{'_scale_window','calculate_sei','calculate_mfi2'})
        if d.get('use_fragility_filter'):s['buy']&=m.calculate_mfi2(f)<d.get('fragility_threshold',.5)
        if d.get('use_entropy_filter'):s['buy']&=m.calculate_sei(f)<d.get('entropy_threshold',.7)
    return s

def close_exit_array(d,s,j,entry,entry_bar,close):
    if j-entry_bar<int(d.get('min_hold_bars',1)):return False
    pnl=(close[j]/entry-1)*100
    if d.get('use_accel_exit') and (pnl>=d.get('accel_exit_min_pnl',.5) or pnl<0):
        a=s['accel'][j];n=int(d.get('accel_exit_lookback',1));kind=d.get('accel_exit_type','sign_reversal');negative=np.all(s['accel'][max(0,j-n+1):j+1]<0)
        trigger={'sign_reversal':negative,'magnitude':a<-d.get('accel_exit_threshold',0),'both':negative and abs(a)>d.get('accel_exit_threshold',0)}[kind]
        if d.get('use_jerk_confirm') and d.get('jerk_confirm_threshold',0)>0:trigger=trigger and s['jerk'][j]<-d['jerk_confirm_threshold']
        if trigger:return True
    return bool((d.get('exit_on_opposite_signal',True) and s['sell'][j]) or (d.get('exit_on_midline_cross',False) and s['osc'][j]>0))

def metrics(frame,equity,initial,fees,slip,borrow,entries,fills):
    eq=np.asarray(equity);daily=pd.Series(eq,index=pd.DatetimeIndex(frame.bar_end)).resample('1D').last().dropna().to_numpy()
    dd=lambda x:float(np.max(1-np.r_[initial,x]/np.maximum.accumulate(np.r_[initial,x])))
    days=(frame.bar_end.iloc[-1]-frame.timestamp.iloc[0]).total_seconds()/86400
    return {'total_return':float(eq[-1]/initial-1),'cagr':float((eq[-1]/initial)**(365.25/days)-1) if eq[-1]>0 else -1,
      'max_drawdown':dd(daily),'bar_close_drawdown':dd(eq),'fees':fees,'slippage':slip,'borrow':borrow,'entries':entries,'fills':len(fills),
      'start':str(frame.timestamp.iloc[0]),'end':str(frame.bar_end.iloc[-1]),'ever_insolvent':bool(min(eq)<=0),'initial_capital':initial}

def spot_run(f,s,d,start,end,fee=.001,slip=.0005,delay=1,borrow_rate=.05,kind='pf'):
    a={k:f[k].to_numpy() for k in ['open','high','low','close']};sig={k:np.asarray(v) for k,v in s.items()} if s is not None else {}
    times=pd.DatetimeIndex(f.timestamp);cash=10000.;q=0.;entry=0.;entry_bar=-1;highwater=0.;last_exit=-100000
    fees=slippage=borrow=0.;entries=0;fills=[];history=[]
    def trade(delta,raw,t,reason):
        nonlocal cash,q,fees,slippage
        price=raw*(1+np.sign(delta)*slip);commission=abs(delta*price)*fee
        cash-=delta*price+commission;q+=delta;fees+=commission;slippage+=abs(delta*raw)*slip
        fills.append({'timestamp':str(times[t]),'decision_bar':t-delay,'units':float(delta),'price':float(price),'reason':reason})
    for t in range(start,end+1):
        j=t-delay
        if q<0 and t>start:
            cost=abs(q)*a['open'][t]*borrow_rate*(times[t]-times[t-1]).total_seconds()/(365.25*86400);cash-=cost;borrow+=cost
        equity_open=cash+q*a['open'][t]
        if equity_open<=0:
            if q:trade(-q,a['open'][t],t,'insolvent_liquidation')
            history.extend([cash]*(end-t+1));break
        if kind=='benchmark':
            if t==start:trade(cash/(a['open'][t]*(1+slip)*(1+fee)),a['open'][t],t,'buy_hold');entries+=1
        else:
            held=q!=0
            exit_now=False
            if held:
                if kind=='pf':exit_now=close_exit_array(d,sig,j,entry,entry_bar,a['close'])
                else:exit_now=bool((q>0 and (sig['peak'][j]>=d['exit_score'] or sig['regime'][j]<=0)) or (q<0 and (sig['valley'][j]>=d['exit_score'] or sig['regime'][j]>=0)))
                if exit_now:trade(-q,a['open'][t],t,'signal_exit');last_exit=t
            elif (j-last_exit>=int(d.get('min_bars_between',1)) if kind=='pf' else j-last_exit>=d['cooldown']):
                direction=0
                if kind=='pf':direction=int(sig['buy'][j])
                elif sig['regime'][j]==1 and sig['valley'][j]>=d['entry_score']:direction=1
                elif d.get('allow_short') and sig['regime'][j]==-1 and sig['peak'][j]>=d['entry_score']:direction=-1
                if direction:
                    size=.95*cash/(a['open'][t]*(1+slip)*(1+fee));trade(direction*size,a['open'][t],t,'entry');entries+=1
                    entry=a['open'][t]*(1+direction*slip);entry_bar=t;highwater=entry
            if q:
                price=reason=None
                if kind=='pf':price,reason=protective_fill(a['open'][t],a['high'][t],a['low'][t],entry,highwater,d)
                else:
                    atr=sig['atr'][j]
                    if q>0:
                        stop=entry-d['stop_atr']*atr;target=entry+d['target_atr']*atr
                        if a['open'][t]<=stop:price,reason=a['open'][t],'stop_gap'
                        elif a['open'][t]>=target:price,reason=a['open'][t],'target_gap'
                        elif a['low'][t]<=stop:price,reason=stop,'stop'
                        elif a['high'][t]>=target:price,reason=target,'target'
                    else:
                        stop=min(entry+d['stop_atr']*atr,cash/(2*abs(q)));target=entry-d['target_atr']*atr
                        if a['open'][t]>=stop:price,reason=a['open'][t],'stop_or_margin_gap'
                        elif a['open'][t]<=target:price,reason=a['open'][t],'target_gap'
                        elif a['high'][t]>=stop:price,reason=stop,'stop_or_margin'
                        elif a['low'][t]<=target:price,reason=target,'target'
                if price is not None:trade(-q,price,t,reason);last_exit=t
                else:highwater=max(highwater,a['high'][t])
        if t==end and q:trade(-q,a['close'][t],t,'terminal_close')
        history.append(cash+q*a['close'][t])
    return metrics(f.iloc[start:end+1],history,10000.,fees,slippage,borrow,entries,fills),np.asarray(history),fills
