"""Direct actual-contract target blends of PF, quant and paper-style daily rules.

Saved fill/state paths are shadow experts; targets are observed only after the
bar closes. New integer positions trade the following opening print, never the
old experts' historical fill prices. No fractional strategy-fund NAV fiction.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
import pandas as pd
from full_catalogue_options import save,sha
from futures_intraday_retest import SPECS,signal_frame
from intraday_common import metrics

HERE=Path(__file__).resolve().parent
OUT=HERE/'ensemble/intraday'
METHODS=['EqualFamilies','InverseVol','PastSharpe','WithoutPatternFindr','WithoutQuant']

def read_shadow(f,name,root):
    folder=HERE/'next_batch';hp=folder/(name+'_equity.parquet');fp=folder/(name+'_fills.json')
    h=pd.read_parquet(hp);fills=json.loads(fp.read_text())
    h=h.set_index('timestamp').reindex(pd.DatetimeIndex(f.bar_end))
    if h.equity.isna().any() or (h.equity<=0).any():raise ValueError('Missing or insolvent shadow '+name)
    trades=pd.Series(0.,index=pd.DatetimeIndex(f.timestamp))
    for row in fills:
        stamp=pd.Timestamp(row['timestamp'])
        if stamp not in trades.index:raise ValueError('Unmatched saved fill')
        trades.loc[stamp]+=row['units']
    q=trades.cumsum().to_numpy()
    w=q*SPECS[root][0]*f.close.to_numpy()/h.equity.to_numpy()
    # Old books can briefly exceed their opening gross cap at the close. A target
    # adapter caps NEW exposure at95%; it never rewrites old positions or returns.
    w=np.clip(w,-.95,.95)
    return w,{'name':name,'curve_sha256':sha(hp),'fills_sha256':sha(fp)}

def paper_daily(f):
    sf=signal_frame(f).set_index('timestamp')
    d=sf.resample('1D').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    c=d.close
    a=(c>c.rolling(200).mean()).astype(float)
    b=(c>c.shift(252)).astype(float)
    high=d.high.rolling(55).max().shift(1);low=d.low.rolling(20).min().shift(1)
    state=0;channel=[];ibs=[];held=0;pos=0
    ratio=(c-d.low)/(d.high-d.low).replace(0,np.nan)
    for i in range(len(d)):
        if c.iloc[i]>high.iloc[i]:state=1
        elif c.iloc[i]<low.iloc[i]:state=0
        channel.append(state)
        if pos:
            held+=1
            if ratio.iloc[i]>.8 or held>=5:pos=0;held=0
        elif ratio.iloc[i]<.2:pos=1;held=0
        ibs.append(pos)
    daily=pd.DataFrame({'known_at':d.index+pd.Timedelta(days=1),
        'PaperTrend':(a+b+np.asarray(channel)).to_numpy()/3*.95,
        'PaperReversal':np.asarray(ibs)*.95,
        'PaperRisk':np.minimum(.95,.12/(c.pct_change().rolling(63).std()*np.sqrt(252))).fillna(0).to_numpy()})
    z=pd.merge_asof(pd.DataFrame({'known_at':pd.DatetimeIndex(f.bar_end)}),daily,on='known_at',direction='backward').fillna(0)
    return z.drop(columns='known_at').to_numpy(),list(daily.columns[1:])

def execute(f,targets,start,end,root,ticks=1,delay=1,benchmark=False):
    mult,tick=SPECS[root];capital=1e6;cash=capital;q=0;mark=0.;fees=slip=0.;fills=[];hist=[]
    a={c:f[c].to_numpy() for c in ['open','close','old_open']};times=pd.DatetimeIndex(f.timestamp)
    def charge(delta,price,t,reason,symbol):
        nonlocal cash,fees,slip
        fee=abs(delta)*2.5;slippage=abs(delta)*mult*tick*ticks
        cash-=fee+slippage;fees+=fee;slip+=slippage
        fills.append({'timestamp':str(times[t]),'symbol':symbol,'units':int(delta),'reference_price':float(price),'modeled_price':float(price+np.sign(delta)*tick*ticks),'fee':fee,'slippage':slippage,'reason':reason})
    for t in range(start,end+1):
        op=a['open'][t];close=a['close'][t]
        if q:
            old=a['old_open'][t] if np.isfinite(a['old_open'][t]) else op
            cash+=q*mult*(old-mark)
            if np.isfinite(a['old_open'][t]):
                charge(-q,old,t,'roll_out',f.old_symbol.iloc[t]);charge(q,op,t,'roll_in',f.symbol.iloc[t])
        if cash<=0:raise ValueError('Insolvent account: keep failure, no wealth reset')
        j=t-delay
        scheduled=(t==start or times[t].normalize()!=times[t-1].normalize())
        want=None
        if benchmark:
            if t==start:want=.95
        elif scheduled:want=float(targets[j])
        if want is not None:
            if abs(want)>.950001 or not np.isfinite(want):raise ValueError('Invalid target')
            desired=int(np.sign(want)*np.floor(abs(want)*cash/(op*mult+2.5+tick*ticks*mult)))
            if desired!=q:charge(desired-q,op,t,'rebalance',f.symbol.iloc[t]);q=desired
        elif abs(q)*op*mult>cash*.99:
            desired=int(np.sign(q)*max(0,np.floor(cash*.95/(op*mult+2.5+tick*ticks*mult))))
            if desired!=q:charge(desired-q,op,t,'gross_cap_reduction',f.symbol.iloc[t]);q=desired
        cash+=q*mult*(close-op);mark=close
        if t==end and q:charge(-q,close,t,'terminal_close',f.symbol.iloc[t]);q=0
        hist.append(cash)
    m=metrics(f.iloc[start:end+1],hist,capital,fees,slip,0,sum(x['reason']=='rebalance' for x in fills),fills)
    return m,np.asarray(hist),fills

def fit_weights(f,returns,names,method):
    # A completed UTC day's final mark is usable only on subsequent bars.
    r=pd.DataFrame(returns,index=pd.DatetimeIndex(f.bar_end)).resample('1D',closed='right',label='right').agg(lambda v:np.prod(1+v)-1)
    r=r.loc[r.index.isin(pd.DatetimeIndex(f.bar_end).normalize().unique())]
    mean=r.rolling(63,min_periods=21).mean();sd=r.rolling(63,min_periods=21).std().clip(lower=.01/np.sqrt(252))
    inv=1/sd;inv=inv.div(inv.sum(axis=1),axis=0).fillna(1/len(names))
    score=(mean/sd*np.sqrt(252)).clip(-3,3).fillna(0)
    v=np.exp(np.column_stack([score.to_numpy(),np.zeros(len(score))]));v/=v.sum(axis=1,keepdims=True)
    if method=='InverseVol':daily=np.column_stack([inv.to_numpy(),np.zeros(len(inv))])
    elif method=='PastSharpe':daily=v
    else:
        w=np.r_[np.ones(len(names)),0.]
        if method=='WithoutPatternFindr':w[names.index('PatternFindr')]=0
        if method=='WithoutQuant' and 'Quant' in names:w[names.index('Quant')]=0
        w/=w.sum();daily=np.tile(w,(len(r),1))
    df=pd.DataFrame(daily,index=r.index,columns=[*names,'Cash']).reset_index(names='known_at')
    # At bar end t, equal timestamps are already complete; all future days excluded.
    out=pd.merge_asof(pd.DataFrame({'known_at':pd.DatetimeIndex(f.bar_end)}),df,on='known_at',direction='backward').drop(columns='known_at')
    return out.fillna(pd.Series(np.r_[np.full(len(names),1/len(names)),0.],index=out.columns)).to_numpy()

def main(root):
    OUT.mkdir(parents=True,exist_ok=True)
    folder=OUT/root;folder.mkdir(exist_ok=True)
    protocol=OUT/'protocol.json'
    if not protocol.exists():save(protocol,{'registered_at':datetime.now(timezone.utc).isoformat(),
      'markets':['ES','GC','CL'],'evaluation':'2023-Jun2026; prior 2021-2022 as shadow-account and feature warmup. Familiar history, retrospective.',
      'methods':METHODS,'paper':'Daily completed-price SMA200,252day momentum,55/20breakout family;5day IBS reversal;12% volatility target. These are declared daily-signal futures adaptations.',
      'experts':'ALL supported saved PF15m books by root, quantC7/C8/C11 forES. Observe close holdings/NAV, average within family. No performance-selected exclusions; no unsafe module imports.',
      'execution':'NEW actual-contract account, daily first observed UTC bucket rebalancing based on prior completed bar. Both raw roll legs charged; integer contracts,95%gross cap. $1m, zero collateral yield, $2.50/side +1tick or3tickstress. No original shadow protective orders are copied; vote blend is a new strategy.',
      'weights':'Equal families; trailing63 observed UTCday inversevol; clipped pastSharpe softmax pluscash; omitPF/omitQuant ablations. Historical book costs included in weight-training returns; new account pays its own fees and slippage.',
      'gate':'All calendar years andfullsample must beat matched95%gross passivelyrolled samefuture return anddailydd, also3tick andextra-bar-delay. No actual bid/ask ormargin schedule, soexecution qualificationfalse.',
      'tensortrade':'NewPPO is trained in separate daily US ETF/BTC target study. This futures branch tests fixed causal ensembles only, not PPO deployment.'})
    f=pd.read_parquet(HERE/'next_batch'/f'{root}_execution_15m.parquet')
    start=int(np.flatnonzero(f.timestamp>=pd.Timestamp('2021-06-01',tz='UTC'))[0])
    f=f.iloc[start:].reset_index(drop=True)
    targets,names=paper_daily(f);audit=[]
    configs=json.loads((HERE/'extension/patternfindr_configs.json').read_text())
    paths=[]
    for row in configs:
        d=row.get('config',{})
        if d.get('ticker')==root+'=F' and d.get('interval')=='15m' and (HERE/'next_batch'/(row['id']+'_equity.parquet')).exists():paths.append((row['id'],'PatternFindr'))
    if root=='ES':paths += [(r['name'],'Quant') for r in json.loads((HERE/'extension/quant_family_protocol.json').read_text())['choices']]+[('C11','Quant')]
    groups={}
    for name,family in paths:
        w,a=read_shadow(f,name,root);groups.setdefault(family,[]).append(w);audit.append(dict(a,family=family))
    for family,values in groups.items():targets=np.column_stack([targets,np.mean(values,axis=0)]);names.append(family)
    save(folder/'sources.json',{'feed_sha256':sha(HERE/'next_batch'/f'{root}_execution_15m.parquet'),'shadow_books':audit,'families':names})
    # Re-execute each family causally before estimating its past performance.
    returns=[]
    for i,name in enumerate(names):
        _,eq,_=execute(f,targets[:,i],1,len(f)-1,root)
        e=np.r_[1e6,eq];returns.append(np.r_[0,e[1:]/e[:-1]-1])
    returns=np.column_stack(returns)
    streams={m:np.sum(fit_weights(f,returns,names,m)[:,:-1]*targets,axis=1) for m in METHODS}
    rows=[];first=int(np.flatnonzero(f.timestamp>=pd.Timestamp('2023-01-01',tz='UTC'))[0])
    periods=[('full',first,len(f)-1)]+[(str(y),int(ix[0]),int(ix[-1])) for y in range(2023,2027) if len(ix:=np.flatnonzero(f.timestamp.dt.year==y))>1]
    for block,a,b in periods:
        for case,ticks,delay in [('base',1,1),('three_ticks',3,1),('extra_bar_delay',1,2)]:
            bm,be,bfills=execute(f,None,a,b,root,ticks,benchmark=True)
            for method,s in streams.items():
                m,eq,ff=execute(f,s,a,b,root,ticks,delay)
                rows.append({'root':root,'method':method,'block':block,'case':case,**m,'benchmark_return':bm['total_return'],'benchmark_drawdown':bm['max_drawdown'],'pass':bool(m['total_return']>bm['total_return'] and m['max_drawdown']<=bm['max_drawdown'])})
                if block=='full':
                    pd.DataFrame({'timestamp':f.bar_end.iloc[a:b+1],'equity':eq}).to_parquet(folder/(method+'_'+case+'.parquet'),index=False);save(folder/(method+'_'+case+'_fills.json'),ff)
            if block=='full':
                pd.DataFrame({'timestamp':f.bar_end.iloc[a:b+1],'equity':be}).to_parquet(folder/('BuyHold_'+case+'.parquet'),index=False);save(folder/('BuyHold_'+case+'_fills.json'),bfills)
    save(folder/'results.json',rows)
    save(folder/'manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),'protocol_sha256':sha(protocol),'source_sha256':sha(folder/'sources.json'),'candidate_cases':len(rows),'shadow_books':len(paths),'all_gates_pass':[m for m in METHODS if all(r['pass'] for r in rows if r['method']==m)],'execution_qualified':False})
    print(root,'done',flush=True)

if __name__=='__main__':main(sys.argv[1])
