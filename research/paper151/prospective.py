"""Append-only prospective SIMULATION. No broker, order API or messaging calls.

Decisions are timestamped before their precommitted future opening-price fill.
Late runs never generate retroactive decisions. Bitcoin deliberately uses a
two-daily-bar signal lag so a daily Chicago-evening recorder can commit in time.
"""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from filelock import FileLock

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tensortrade_lab.market_data import schedule
from tensortrade_lab.sources import download_yahoo
from tensortrade_lab.portfolio import Panel,Ledger

OUT=HERE/'prospective'
SPECS={'Gold80_20':{'symbols':['SPY','GLD'],'rule':'gold','benchmark':'SPY'},
       'TLT_MA50_200':{'symbols':['TLT'],'rule':'dual','benchmark':'TLT'},
       'TLT_MA20_50_200':{'symbols':['TLT'],'rule':'triple','benchmark':'TLT'},
       'BTC_Momentum252_Delay2':{'symbols':['BTC-USD'],'rule':'momentum','benchmark':'BTC-USD'}}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def initialize():
    OUT.mkdir(exist_ok=True)
    f=OUT/'freeze.json'
    if f.exists():return json.loads(f.read_text())
    d={'frozen_at':datetime.now(timezone.utc).isoformat(),'strategies':SPECS,'initial_capital_each':10000,
       'source_hashes':{str(Path(__file__).resolve()):digest(__file__),
                       str(ROOT/'src/tensortrade_lab/portfolio.py'):digest(ROOT/'src/tensortrade_lab/portfolio.py')},
       'execution':'US decisions after completed close, next exchange open; BTC latest completed UTC day, next strictly future UTC midnight (two-bar lag). Fractional units; no leverage. Pending orders may be accounted later only if recorded before their fill timestamp.',
       'costs':'US 5bp commission +5bp adverse execution per side; BTC 10bp commission +5bp adverse execution per side. Explicit distributions; zero cash interest in both strategy and benchmark.',
       'missed_runs':'Never backfill signals. Retain holdings, account precommitted orders and observed distributions, flag gaps.',
       'evaluation':'Collect 12 months through at least October 2027; monthly descriptive comparisons only; no retuning or winner declaration from a short favorable start.',
       'status':'exploratory candidates, not validated winners'}
    with f.open('x') as h:json.dump(d,h,indent=2)
    return d

def load_log(path=None):
    path=path or OUT/'events.jsonl';events=[];previous='0'*64
    if not path.exists():return events
    for line in path.read_text().splitlines():
        e=json.loads(line);claimed=e.pop('event_hash')
        if e['previous_hash']!=previous or hashlib.sha256(encoded(e)).hexdigest()!=claimed:
            raise ValueError('Prospective log integrity failure')
        e['event_hash']=claimed;events.append(e);previous=claimed
    return events

def append_event(event,path=None):
    path=path or OUT/'events.jsonl';events=load_log(path)
    event=deepcopy(event);event['previous_hash']=events[-1]['event_hash'] if events else '0'*64
    event['event_hash']=hashlib.sha256(encoded(event)).hexdigest()
    with path.open('a') as f:f.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n');f.flush()
    return event

def next_execution(now,crypto=False):
    if crypto:return now.normalize()+pd.Timedelta(days=1)
    cal=schedule(str(now.date()),str((now+pd.Timedelta(days=12)).date()))
    future=cal[cal.open>now]
    if future.empty:raise ValueError('No future exchange open')
    return future.open.iloc[0]

def weights_for(spec,frames):
    if spec['rule']=='gold':return [.8,.2]
    f=frames[spec['symbols'][0]];c=f.close
    if len(c)<337:raise ValueError('Insufficient signal history')
    if spec['rule']=='dual':return [float(c.iloc[-50:].mean()>c.iloc[-200:].mean())]
    if spec['rule']=='triple':return [float(c.iloc[-20:].mean()>c.iloc[-50:].mean()>c.iloc[-200:].mean())]
    tr=np.cumprod(np.r_[1.,(c.to_numpy()[1:]+f.dividend.to_numpy()[1:])/c.to_numpy()[:-1]])
    return [float(tr[-1]>tr[-253])]

def new_account(spec):
    return {'cash':10000.,'units':{s:0. for s in spec['symbols']},'benchmark_cash':10000.,'benchmark_units':0.,
            'last_processed':None,'last_signal_end':None,'last_target':None,'last_rebalance_month':None,
            'pending_orders':[],'benchmark_started':False,'equity':10000.,'benchmark_equity':10000.}

def fill_account(account,spec,row_by_symbol,pending):
    symbols=spec['symbols'];prices=np.array([row_by_symbol[s]['open'] for s in symbols])
    crypto=symbols==['BTC-USD'];fee=.001 if crypto else .0005;slip=.0005
    # Reuse exactly the self-financing commission/slippage sizing equation.
    dt=pd.Timestamp(pending['execute_at']);times=pd.DatetimeIndex([dt-pd.Timedelta(days=1),dt,dt+pd.Timedelta(days=1)])
    p=Panel(symbols,times,times+pd.Timedelta(hours=6),np.tile(prices,(3,1)),np.tile(prices,(3,1)),
            np.zeros((3,len(symbols))),np.zeros((3,len(symbols),1)),np.ones((3,len(symbols))),
            np.zeros((3,len(symbols))),np.zeros(3))
    ledger=Ledger(p,1,2,fee,slip);ledger.cash=account['cash'];ledger.shares=np.array([account['units'][s] for s in symbols]);ledger.index=1
    ledger.rebalance(np.array(pending['weights']),prices)
    account['cash']=ledger.cash;account['units']=dict(zip(symbols,ledger.shares.tolist()))
    if not account['benchmark_started']:
        price=row_by_symbol[spec['benchmark']]['open'];q=account['benchmark_cash']/(price*(1+slip)*(1+fee))
        account['benchmark_units']=q;account['benchmark_cash']=0.;account['benchmark_started']=True
    return ledger.fills

def advance(account,spec,frames,now,freeze):
    account=deepcopy(account);events=[]
    ends={s:pd.Timestamp(frames[s].bar_end.iloc[-1]) for s in spec['symbols']}
    if len(set(ends.values()))!=1:raise ValueError('Unaligned latest bars')
    end=next(iter(ends.values()));crypto=spec['symbols']==['BTC-USD']
    if end<=pd.Timestamp(freeze['frozen_at']):return account,[{'type':'waiting_for_post_freeze_close'}]
    aligned=frames[spec['symbols'][0]]
    after=pd.Timestamp(account['last_processed']) if account['last_processed'] else pd.Timestamp(freeze['frozen_at'])
    for _,row in aligned[aligned.bar_end>after].iterrows():
        bars={}
        for symbol in spec['symbols']:
            same=frames[symbol][frames[symbol].timestamp==row.timestamp]
            if len(same)!=1:raise ValueError('Missing common session; no price imputation')
            bars[symbol]=same.iloc[0].to_dict()
        for symbol,b in bars.items():
            if b.get('split',1.)!=1. and (account['units'][symbol] or (symbol==spec['benchmark'] and account['benchmark_units'])):
                raise ValueError('New split requires an explicit versioned share-basis reconciliation')
            account['cash']+=account['units'][symbol]*b.get('dividend',0.)
        b=bars[spec['benchmark']];account['benchmark_cash']+=account['benchmark_units']*b.get('dividend',0.)
        pending=account['pending_orders'][0] if account['pending_orders'] else None
        if pending and pd.Timestamp(pending['execute_at'])<=row.timestamp:
            if pd.Timestamp(pending['execute_at'])!=row.timestamp:raise ValueError('Precommitted execution bar missing')
            if pd.Timestamp(pending['recorded_at'])>=row.timestamp:raise ValueError('Attempted retrospective order')
            fills=fill_account(account,spec,bars,pending)
            events.append({'type':'simulated_fills','order':pending,'fills':fills});account['pending_orders'].pop(0)
        account['last_processed']=row.bar_end.isoformat()
    account['equity']=account['cash']+sum(account['units'][s]*frames[s].close.iloc[-1] for s in spec['symbols'])
    account['benchmark_equity']=account['benchmark_cash']+account['benchmark_units']*frames[spec['benchmark']].close.iloc[-1]
    if account['last_signal_end']==end.isoformat():return account,events
    if account['last_signal_end']:
        previous=pd.Timestamp(account['last_signal_end'])
        missing=aligned[(aligned.bar_end>previous)&(aligned.bar_end<end)]
        if len(missing):events.append({'type':'missed_signal_sessions','from':previous.isoformat(),
                                      'to':end.isoformat(),'count':len(missing)})
    execute=next_execution(now,crypto)
    # US latest decision must be the final scheduled session preceding execution.
    if crypto:
        if end!=now.normalize():raise ValueError('Stale BTC completed day')
    else:
        cal=schedule(str((now-pd.Timedelta(days=12)).date()),str(now.date()))
        known=cal[cal.close<=now]
        if known.empty or end!=known.close.iloc[-1]:raise ValueError('Stale US completed session')
    w=weights_for(spec,frames);month=f'{execute.year}-{execute.month:02}'
    need=account['last_target']!=w or (spec['rule']=='gold' and account['last_rebalance_month']!=month)
    if need:
        if account['pending_orders'] and pd.Timestamp(account['pending_orders'][-1]['execute_at'])>=execute:
            raise ValueError('Already committed this or a later execution; no replacement from hindsight')
        order={'recorded_at':now.isoformat(),'signal_end':end.isoformat(),'execute_at':execute.isoformat(),'weights':w}
        account['pending_orders'].append(order);account['last_target']=w;account['last_rebalance_month']=month
        events.append({'type':'future_simulated_order','order':order})
    account['last_signal_end']=end.isoformat()
    events.append({'type':'signal','signal_end':end.isoformat(),'weights':w,'equity':account['equity'],
                   'benchmark_equity':account['benchmark_equity']})
    return account,events

def run():
    freeze=initialize()
    for p,h in freeze['source_hashes'].items():
        if digest(p)!=h:raise ValueError('Frozen implementation changed; create a new version instead')
    now=pd.Timestamp.now(tz='UTC');snapshot=OUT/'snapshots'/now.strftime('%Y%m%dT%H%M%SZ');snapshot.mkdir(parents=True,exist_ok=True)
    frames={};sources={};errors=[]
    for symbol in ['SPY','GLD','TLT','BTC-USD']:
        p=snapshot/(symbol+'.parquet')
        try:
            download_yahoo(symbol,p,start='2024-01-01',end=str((now+pd.Timedelta(days=1)).date()),calendar='24/7' if symbol=='BTC-USD' else 'XNYS')
            frames[symbol]=pd.read_parquet(p);sources[symbol]={'path':str(p),'sha256':digest(p)}
        except Exception as e:errors.append({'symbol':symbol,'error':str(e)[:400]})
    logs=load_log();state=deepcopy(logs[-1]['state']) if logs else {k:new_account(v) for k,v in SPECS.items()}
    actions={}
    for name,spec in SPECS.items():
        if any(s not in frames for s in spec['symbols']):continue
        try:state[name],actions[name]=advance(state[name],spec,frames,now,freeze)
        except Exception as e:errors.append({'strategy':name,'error':str(e)[:400]})
    event=append_event({'recorded_at':now.isoformat(),'freeze_sha256':digest(OUT/'freeze.json'),
        'sources':sources,'actions':actions,'errors':errors,'state':state})
    (OUT/'latest.json').write_text(json.dumps(event,indent=2,allow_nan=False))
    print(json.dumps({'recorded_at':str(now),'errors':errors,'actions':actions,'accounts':state},indent=2))
    return 1 if errors else 0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initialize',action='store_true');a=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with FileLock(OUT/'run.lock'):
        if a.initialize:print(json.dumps(initialize(),indent=2))
        else:sys.exit(run())
