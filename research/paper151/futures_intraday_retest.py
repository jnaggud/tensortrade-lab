"""Actual-contract futures execution with a causal signal-only linked series."""
from intraday_common import *
from dataclasses import asdict
SPECS={'ES':(50.,.25),'GC':(100.,.1),'CL':(1000.,.01)}

def build_root(root,raw,selections):
    f=raw[raw.root==root].sort_values('timestamp').copy();books={t:z.set_index('symbol').to_dict('index') for t,z in f.groupby('timestamp')}
    rows=[];held=None;scale=1.;deferred=0;unprinted=[]
    for time,book in books.items():
        day=time.strftime('%Y%m%d')
        if day not in selections:continue
        desired=selections[day]['front'][root]
        if held is None:held=desired
        old_open=None;old_symbol=None
        if desired!=held:
            if held in book and desired in book:
                old_open=book[held]['open'];old_symbol=held;scale*=old_open/book[desired]['open'];held=desired
            else:deferred+=1
        if held not in book:
            # No trade printed for this selected contract in this bucket. Do not
            # fabricate OHLC or execute an order at another contract's price.
            # The next actual observation recognizes the whole intervening move.
            unprinted.append({'timestamp':str(time),'symbol':held});continue
        z=book[held];rows.append({'timestamp':time,'bar_end':z['bar_end'],'symbol':held,'root':root,'scale':scale,
            'open':z['open'],'high':z['high'],'low':z['low'],'close':z['close'],'volume':z['volume'],
            'old_symbol':old_symbol,'old_open':old_open,'first_print_minute':z['first_print_minute'],'minute_rows':z['minute_rows']})
    out=pd.DataFrame(rows)
    if out.empty or out.timestamp.iloc[-1]!=max(books):
        raise ValueError(f'Cannot complete {root} feed: held contract absent at final observed root bucket')
    if (out[['open','high','low','close']]<=0).any().any():raise ValueError('Nonpositive price requires different signal transformation')
    out.attrs['audit']={'bars':len(out),'rolls':int(out.old_open.notna().sum()),'deferred_roll_buckets':deferred,'unprinted_selected_buckets':len(unprinted),'unprinted_bucket_details':unprinted,'first':str(out.timestamp.min()),'end':str(out.bar_end.max()),
      'construction':'Forward multiplicative linking only at observed overlap, using old/new first-traded bucket opens. Historical signal values never revised. All accounting uses actual raw prices.',
      'limitation':'First traded minutes of the two contracts may differ within a 15-minute bucket. Missing selected-contract print buckets are omitted, with no fills or synthetic bars; next observed price captures the gap. Equity drawdown is measured at observed prints, not every clock bucket. Modeled execution, not a synchronous executable bid/ask claim.'}
    return out

def signal_frame(f):
    z=f.copy()
    for c in ['open','high','low','close']:z[c]=z[c]*z.scale
    return z

def protective(opening,high,low,side,stop,target=None):
    if side>0:
        if stop is not None and opening<=stop:return opening,'stop_gap'
        if target is not None and opening>=target:return opening,'target_gap'
        if stop is not None and low<=stop:return stop,'stop'
        if target is not None and high>=target:return target,'target'
    else:
        if stop is not None and opening>=stop:return opening,'stop_gap'
        if target is not None and opening<=target:return opening,'target_gap'
        if stop is not None and high>=stop:return stop,'stop'
        if target is not None and low<=target:return target,'target'
    return None,None

def quant_entry(sig,j,p,last,t):
    def ready(prefix):return j-last>=int(p.get(prefix+'cooldown',0))
    flags={1:bool(sig['core_long'][j]),2:bool(sig['cap_long'][j]),3:bool(sig['participation_long'][j]),4:bool(sig['trend_carry_long'][j])}
    ready_map={1:ready(''),2:ready('cap_'),3:ready('participation_'),4:ready('trend_carry_')}
    for k,prefix in [(2,'cap_'),(3,'participation_'),(4,'trend_carry_')]:
        if p.get(prefix+'priority') and ready_map[k] and flags[k]:return 1,k
    for k in [1,3,4]:
        if ready_map[k] and flags[k]:return 1,k
    if ready_map[1] and p.get('allow_short',True) and sig['core_short'][j]:return -1,1
    if ready_map[2] and flags[2]:return 1,2
    return 0,0

def run(f,s,d,start,end,root,kind='pf',ticks=1,delay=1,initial=1_000_000.):
    mult,tick=SPECS[root];a={c:f[c].to_numpy() for c in ['open','high','low','close','old_open','scale']};sg={k:np.asarray(v) for k,v in s.items()} if s is not None else {}
    adjusted_close=a['close']*a['scale'];times=pd.DatetimeIndex(f.timestamp);symbols=f.symbol.to_numpy()
    cash=initial;q=0;mark=0.;entry=0.;hwm=0.;lwm=0.;entry_bar=-1;last_exit=-100000;leg_kind=0
    fees=slippage=0.;entries=0;rolls=0;fills=[];hist=[];reductions=0
    def charge(units,price,t,why,symbol=None):
        nonlocal cash,fees,slippage
        comm=abs(units)*2.5;cost=abs(units)*tick*ticks*mult;cash-=comm+cost;fees+=comm;slippage+=cost
        fills.append({'timestamp':str(times[t]),'decision_bar':t-delay,'symbol':symbol or symbols[t],'units':int(units),'reference_price':float(price),'modeled_price':float(price+np.sign(units)*tick*ticks),'fee':comm,'slippage':cost,'reason':why})
    for t in range(start,end+1):
        j=t-delay;op=a['open'][t];scale=a['scale'][t]
        if q:
            prev_open=a['old_open'][t] if np.isfinite(a['old_open'][t]) else op
            cash+=q*mult*(prev_open-mark)
            if np.isfinite(a['old_open'][t]):
                charge(-q,prev_open,t,'roll_out',f.old_symbol.iloc[t]);charge(q,op,t,'roll_in');rolls+=1
            if abs(q)*op*mult>cash*.99:
                new=int(np.sign(q)*max(0,np.floor(cash*.95/(op*mult))));charge(new-q,op,t,'gross_cap_reduction');q=new;reductions+=1
                if not q:last_exit=t
        if cash<=0:
            if q:charge(-q,op,t,'insolvent_liquidation');q=0
            hist.extend([cash]*(end-t+1));break
        held=q!=0;exit_now=False
        if held and kind!='benchmark':
            if kind=='pf':exit_now=close_exit_array(d,sg,j,entry,entry_bar,adjusted_close)
            else:
                held_bars=j-entry_bar
                if leg_kind==1:exit_now=sg['core_long_close_exit' if q>0 else 'core_short_close_exit'][j]
                else:
                    prefix={2:'cap',3:'participation',4:'trend_carry'}[leg_kind]
                    exit_now=held_bars>=d.get(prefix+'_max_hold',10**9) or (held_bars>=d.get(prefix+'_min_hold',0) and sg[prefix+'_close_exit'][j])
            if exit_now:charge(-q,op,t,'signal_exit');q=0;last_exit=t
        elif not held:
            side=1 if kind=='benchmark' and t==start else 0
            if kind=='pf' and j-last_exit>=int(d.get('min_bars_between',1)):side=int(sg['buy'][j])
            elif kind=='quant':side,leg_kind=quant_entry(sg,j,d,last_exit,t)
            if side:
                n=int(cash*.95/(op*mult+2.5+tick*ticks*mult));q=side*n
                if q:
                    charge(q,op,t,'entry');entry=(op+side*tick*ticks)*scale;hwm=lwm=entry;entry_bar=t;entries+=1
        if q and kind!='benchmark':
            if kind=='pf':
                # The old high-water mark and thresholds live on the linked signal basis.
                raw,why=protective_fill(op*scale,a['high'][t]*scale,a['low'][t]*scale,entry,hwm,d)
                raw=raw/scale if raw is not None else None
            else:
                atr=sg['atr'][j];prefix={1:'',2:'cap_',3:'participation_',4:'trend_carry_'}[leg_kind]
                stop=(max(entry-d[prefix+'stop_atr']*atr,hwm-d[prefix+'trail_atr']*atr) if q>0 else min(entry+d['stop_atr']*atr,lwm+d['trail_atr']*atr))/scale
                target=(entry+d['cap_target_atr']*atr)/scale if leg_kind==2 else None
                raw,why=protective(op,a['high'][t],a['low'][t],np.sign(q),stop,target)
            if raw is not None:
                cash+=q*mult*(raw-op);charge(-q,raw,t,why);q=0;last_exit=t
            else:hwm=max(hwm,a['high'][t]*scale);lwm=min(lwm,a['low'][t]*scale)
        if q:cash+=q*mult*(a['close'][t]-op)
        mark=a['close'][t]
        if t==end and q:charge(-q,mark,t,'terminal_close');q=0
        hist.append(cash)
    m=metrics(f.iloc[start:end+1],hist,initial,fees,slippage,0.,entries,fills);m.update(rolls=rolls,gross_cap_reductions=reductions)
    return m,np.asarray(hist),fills

def main():
    raw=pd.read_parquet(OUT/'actual_contract_15m.parquet');selections={r['date']:r for r in json.loads((OUT/'intraday_selection.json').read_text())}
    frames={r:build_root(r,raw,selections) for r in SPECS}
    for r,f in frames.items():
        save(r+'_feed_audit.json',f.attrs['audit']);f.to_parquet(OUT/(r+'_execution_15m.parquet'),index=False)
    configs=[r for r in json.loads((HERE/'extension/patternfindr_configs.json').read_text()) if r.get('config',{}).get('ticker') in [x+'=F' for x in SPECS] and r['config'].get('interval')=='15m']
    # Copy existing pure parity signal definitions only. The old fill engine is not used.
    spec=importlib.util.spec_from_file_location('causal_quant_signals',HERE/'extension/sources/c11_parity.py');e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
    qrep=json.loads((HERE/'extension/quant_family_protocol.json').read_text())['choices']
    quant=[(x['name'],x['params']) for x in qrep]+[('C11',json.loads((HERE/'extension/c11_frozen_params.json').read_text())['params'])]
    periods=[('full','2021-06-01','2026-06-29'),('2021_2022','2021-06-01','2023-01-01'),('2023_2024','2023-01-01','2025-01-01'),('2025_2026','2025-01-01','2026-06-29')]
    cases=[('base',1,1),('three_ticks',3,1),('extra_bar_delay',1,2)]
    save('futures_intraday_protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),'periods':periods,'cases':cases,'capital':1_000_000,
        'source_sha256':sha(OUT/'actual_contract_15m.parquet'),'selection_sha256':sha(OUT/'intraday_selection.json'),
        'signal':'Forward linked signal series; actual contracts for P&L. Saved thresholds unchanged; regime-only filters use declared causal redesign.',
        'execution':'Next bucket opening print; prior-known stops/ATR/trailing levels; adverse gap fill; stop first on ambiguity; both actual roll legs charged. No legacy same-close fills.',
        'sizing':'Integer contracts targeting <=95% gross equity on entry, unchanged holdings through rolls except opening gross-cap reductions. Zero collateral interest. $2.50 per contract side +1/3 ticks. No dated exchange margin schedule claim.',
        'qualification':'Retrospective modeled-price test. Each rule must beat matched rolled passive return and drawdown in every block/stress; executable quotes and prospective data additionally needed.'})
    rows=[];statuses=[];prefix=[];cached_benchmark={}
    candidates=[(r['id'],'pf',r['config'],r['config']['ticker'].split('=')[0]) for r in configs]+[(name,'quant',params,'ES') for name,params in quant]
    for name,kind,d,root in candidates:
        f=frames[root];sf=signal_frame(f);p=dict(d)
        if kind=='quant':
            p=e.default_c5_params()|p;p.update(use_date_range=False,trend_carry_exit_regime='none',trend_carry_exit_on_macd_roll=False)
        def calculate(frame):
            if kind=='pf':return pf_signals(frame,p)
            return pd.DataFrame([asdict(x) for x in e.build_c5_signal_bars(e.build_features(frame.set_index('timestamp')),p)]).drop(columns=['time'])
        try:
            print('signals',name,flush=True);signals=calculate(sf)
            cut=len(sf)-1237;short=calculate(sf.iloc[:cut].copy())
            ok=np.allclose(signals.iloc[6000:cut].to_numpy(float),short.iloc[6000:].to_numpy(float),equal_nan=True,atol=1e-10,rtol=1e-10)
            prefix.append({'strategy':name,'prefix_bars':cut,'passed':bool(ok)})
            if not ok:raise ValueError('Signal prefix invariance failed')
            for block,begin,finish in periods:
                ix=np.flatnonzero((f.timestamp>=pd.Timestamp(begin,tz='UTC'))&(f.timestamp<pd.Timestamp(finish,tz='UTC')));start,end=int(ix[0]),int(ix[-1])
                for case,ticks,delay in cases:
                    key=(root,block,case)
                    if key not in cached_benchmark:cached_benchmark[key]=run(f,None,{},start,end,root,kind='benchmark',ticks=ticks)[0]
                    b=cached_benchmark[key];m,eq,fills=run(f,signals,p,start,end,root,kind=kind,ticks=ticks,delay=delay)
                    rows.append({'strategy':name,'origin':'Pattern_FindR' if kind=='pf' else 'trading_view_mcp_quant','market':root+' actual contracts / 15m','block':block,'case':case,**m,
                        'benchmark_return':b['total_return'],'benchmark_drawdown':b['max_drawdown'],'pass':bool(m['total_return']>b['total_return'] and m['max_drawdown']<=b['max_drawdown'])})
                    if block=='full' and case=='base':
                        pd.DataFrame({'timestamp':f.bar_end.iloc[start:end+1],'equity':eq}).to_parquet(OUT/(name+'_equity.parquet'),index=False);save(name+'_fills.json',fills)
            statuses.append({'strategy':name,'status':'historically_tested_actual_contract_adaptation'})
        except ValueError as error:statuses.append({'strategy':name,'status':'blocked_causal_or_state_machine','reason':str(error)})
        print('done',name,statuses[-1]['status'],flush=True);save('futures_intraday_results.json',rows);save('futures_intraday_status.json',statuses);save('futures_intraday_prefix.json',prefix)
    save('futures_intraday_benchmarks.json',[{'market':k[0],'block':k[1],'case':k[2],**v} for k,v in cached_benchmark.items()])
    save('futures_intraday_manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'candidates':len(candidates),'completed':sum(r['status'].startswith('historically') for r in statuses),'runs':len(rows),'benchmark_runs':len(cached_benchmark),'code_sha256':sha(__file__),'common_sha256':sha(HERE/'intraday_common.py')})
if __name__=='__main__':main()
