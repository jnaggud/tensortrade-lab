"""Read-only actual-contract 15-minute bars, selected before each UTC day."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json,hashlib,re,time,argparse
import pandas as pd
import numpy as np
from futures_extension import prepare
BASE=Path(__file__).resolve().parent
OUT=BASE/'next_batch'
from cross_project_inventory import ARCHIVE
ROOTS=['ES','GC','CL']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def stage(row):
    import databento as db
    path=Path(row['path']);day=pd.Timestamp(row['date']).date();store=db.DBNStore.from_file(path)
    ids={}
    for symbol in row['symbols']:
        for e in store.symbology['mappings'].get(symbol,[]):
            if e['start_date']<=day<e['end_date']:ids[int(e['symbol'])]=symbol
    chunks=[]
    for f in store.to_df(map_symbols=False,count=500000):
        f=f[f.instrument_id.isin(ids)].reset_index()
        if len(f):chunks.append(f[['ts_event','instrument_id','open','high','low','close','volume']])
    if not chunks:raise ValueError('No selected contract prints '+str(path))
    f=pd.concat(chunks,ignore_index=True).sort_values('ts_event');f['symbol']=f.instrument_id.map(ids)
    if f.duplicated(['ts_event','symbol']).any():raise ValueError('Duplicate bars')
    f['bucket']=f.ts_event.dt.floor('15min')
    g=f.groupby(['symbol','bucket']).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),first_print_minute=('ts_event','min'),last_print_minute=('ts_event','max'),minute_rows=('ts_event','size')).reset_index().rename(columns={'bucket':'timestamp'})
    g['bar_end']=g.timestamp+pd.Timedelta(minutes=15);g['root']=g.symbol.str.extract(r'^(ES|GC|CL)',expand=False)
    actual=sha(path)
    if actual!=row['sha256']:raise ValueError('Source hash mismatch '+str(path))
    return g,{'date':row['date'],'source':str(path),'sha256':actual,'selected_symbols':row['symbols'],'rows':len(g),'source_minute_rows':len(f)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',default='20210101');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    OUT.mkdir(exist_ok=True);cache=OUT/'contract_15m_days';cache.mkdir(exist_ok=True)
    d=prepare(pd.read_parquet(BASE/'extension/actual_futures_daily.parquet'));dates,_,front=d[:3]
    selections={dates[i].strftime('%Y%m%d'):{'front':{r:front[i][r] for r in ROOTS},'symbols':sorted({front[j][r] for j in [i,max(0,i-1)] for r in ROOTS}),
        'known_at':str(dates[i-1]+pd.Timedelta(days=1))} for i in range(1,len(dates)) if dates[i].strftime('%Y%m%d')>=args.start}
    m=json.loads((ARCHIVE/'manifest.json').read_text());chosen={}
    for job in m['jobs'].values():
        if job['schema']!='ohlcv-1m':continue
        for f in job.get('files',[]):
            match=re.search(r'glbx-mdp3-(\d{8})\.ohlcv-1m.dbn.zst$',f['path'])
            if not match or match[1] not in selections:continue
            day=match[1]
            if day not in chosen or f['size']>chosen[day]['size']:chosen[day]={**f,'date':day,**selections[day]}
    rows=[chosen[k] for k in sorted(chosen)]
    (OUT/'intraday_selection.json').write_text(json.dumps(rows,indent=2))
    pending=[r for r in rows if not (cache/(r['date']+'.parquet')).exists()]
    print('Selected source files',len(rows),'pending',len(pending),'compressed GB',sum(r['size'] for r in rows)/1e9,flush=True)
    start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(f,a) in enumerate(pool.map(stage,pending,chunksize=1)):
            f.to_parquet(cache/(a['date']+'.parquet'),index=False);(cache/(a['date']+'.json')).write_text(json.dumps(a,indent=2))
            if i%50==0:print('staged',i+1,'of',len(pending),'seconds',round(time.time()-start),flush=True)
    f=pd.concat([pd.read_parquet(cache/(r['date']+'.parquet')) for r in rows],ignore_index=True).sort_values(['timestamp','symbol'])
    f.to_parquet(OUT/'actual_contract_15m.parquet',index=False)
    report={'at':datetime.now(timezone.utc).isoformat(),'files':len(rows),'rows':len(f),'source_sha256':sha(OUT/'actual_contract_15m.parquet'),
      'first':str(f.timestamp.min()),'last':str(f.bar_end.max()),'roots':ROOTS,'source_selection':'Prior completed UTC-day volume; previous and new selected fronts retained for real roll legs. Inherits declared delivery-month exclusion.',
      'missing_source_dates':sorted(set(selections)-set(chosen)), 'fills':'OHLC-based modeling only; first traded minute and bucket end retained; no synchronous bid/ask assertion.'}
    (OUT/'intraday_manifest.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
