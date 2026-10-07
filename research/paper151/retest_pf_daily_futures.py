"""Five remaining saved daily ES/CL configurations, on actual-contract prices."""
from futures_intraday_retest import *
from futures_extension import prepare, ROOTS

def main():
    source=HERE/'extension/actual_futures_daily.parquet'
    dates,books,front,*_=prepare(pd.read_parquet(source));frames={}
    for root in ['ES','CL']:
        records=[];held=None;scale=1.
        for i,time in enumerate(dates):
            symbol=front[i][root];old_open=np.nan;old_symbol=None
            if symbol not in books[time]:raise ValueError(f'Missing selected {symbol} {time}')
            z=books[time][symbol]
            if held is not None and symbol!=held:
                if held not in books[time]:raise ValueError(f'Missing old roll contract {held} {time}')
                old_open=books[time][held]['open'];old_symbol=held;scale*=old_open/z['open']
            held=symbol
            records.append({'timestamp':time,'bar_end':time+pd.Timedelta(days=1),'root':root,'symbol':symbol,
              'open':z['open'],'high':z['high'],'low':z['low'],'close':z['close'],'volume':z['volume'],
              'old_open':old_open,'old_symbol':old_symbol,'scale':scale})
        frames[root]=pd.DataFrame(records)
        frames[root].to_parquet(OUT/(root+'_execution_daily.parquet'),index=False)
    configs=[r for r in json.loads((HERE/'extension/patternfindr_configs.json').read_text()) if r.get('config',{}).get('ticker') in ['ES=F','CL=F'] and r['config'].get('interval')=='1d']
    periods=[('full','2018-01-01','2026-06-29'),('2018_2019','2018-01-01','2020-01-01'),('2020_2022','2020-01-01','2023-01-01'),('2023_2026','2023-01-01','2026-06-29')]
    cases=[('base',1,1),('three_ticks',3,1),('extra_bar_delay',1,2)]
    save('daily_futures_protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(source),'configs':[r['id'] for r in configs],'periods':periods,'cases':cases,
      'execution':'Same actual-contract ledger as the intraday extension, now daily UTC buckets. Signal-only forward linking, next daily opening print, prior stops, adverse gaps and stop-first ambiguity, actual roll legs. Saved parameters; no optimization.',
      'limits':'Early contract selection, UTC partial Sunday buckets, no executable quotes or dated margins; retrospective adaptation, not Pine parity. $1m capital, <=95% target gross, $2.50/contract/side and 1 or 3 ticks; zero cash yield.'})
    rows=[];prefix=[];bench={}
    for r in configs:
        name=r['id'];d=r['config'];root=d['ticker'].split('=')[0];f=frames[root];sf=signal_frame(f);s=pf_signals(sf,d)
        cut=len(sf)-113;short=pf_signals(sf.iloc[:cut].copy(),d)
        ok=np.allclose(s.iloc[253:cut].to_numpy(float),short.iloc[253:].to_numpy(float),equal_nan=True,atol=1e-10,rtol=1e-10)
        prefix.append({'strategy':name,'passed':bool(ok),'prefix_bars':cut})
        if not ok:raise ValueError('Prefix check failed '+name)
        for block,begin,finish in periods:
            ix=np.flatnonzero((f.timestamp>=pd.Timestamp(begin,tz='UTC'))&(f.timestamp<pd.Timestamp(finish,tz='UTC')));start,end=int(ix[0]),int(ix[-1])
            for case,ticks,delay in cases:
                key=(root,block,case)
                if key not in bench:bench[key]=run(f,None,{},start,end,root,kind='benchmark',ticks=ticks)[0]
                b=bench[key];m,eq,fills=run(f,s,d,start,end,root,ticks=ticks,delay=delay)
                rows.append({'strategy':name,'origin':'Pattern_FindR','market':root+' actual contracts / daily','block':block,'case':case,**m,
                    'benchmark_return':b['total_return'],'benchmark_drawdown':b['max_drawdown'],'pass':bool(m['total_return']>b['total_return'] and m['max_drawdown']<=b['max_drawdown'])})
                if block=='full' and case=='base':
                    pd.DataFrame({'timestamp':f.bar_end.iloc[start:end+1],'equity':eq}).to_parquet(OUT/(name+'_equity.parquet'),index=False);save(name+'_fills.json',fills)
        print('done',name,flush=True)
    save('daily_futures_results.json',rows);save('daily_futures_prefix.json',prefix)
    save('daily_futures_manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'configs':len(configs),'runs':len(rows),'benchmark_runs':len(bench),'code_sha256':sha(__file__),'ledger_sha256':sha(HERE/'futures_intraday_retest.py')})
if __name__=='__main__':main()
