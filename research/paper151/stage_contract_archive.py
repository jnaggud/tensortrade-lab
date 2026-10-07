"""Stage actual daily contracts; map symbols within each file's dated symbology.

No continuous-contract selection, back adjustment, download, or archive mutation.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import re
import time
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'extension'
ROOTS=('ES','GC','CL','ZN','NQ')
RX=re.compile(r'^(ES|GC|CL|ZN|NQ)[FGHJKMNQUVXZ]\d{1,2}$')

def stage_one(row):
    import databento as db
    path=Path(row['path'])
    day=pd.Timestamp(row['date']).date()
    store=db.DBNStore.from_file(path)
    symbols={}
    for raw,entries in store.symbology.get('mappings',{}).items():
        if not RX.fullmatch(raw):continue
        for e in entries:
            if e['start_date']<=day<e['end_date']:
                iid=int(e['symbol'])
                if iid in symbols and symbols[iid]!=raw:raise ValueError('Ambiguous dated symbology')
                symbols[iid]=raw
    f=store.to_df(map_symbols=False).reset_index()
    f=f[f.instrument_id.isin(symbols)].copy()
    f['symbol']=f.instrument_id.map(symbols)
    f['root']=f.symbol.str.extract(r'^(ES|GC|CL|ZN|NQ)',expand=False)
    f['source_file']=str(path)
    f['source_date']=str(day)
    if f.duplicated(['ts_event','instrument_id']).any():raise ValueError('Duplicate contract bar')
    return f,{'date':row['date'],'path':str(path),'expected_sha256':row['sha256'],
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(f)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',default='20210101');ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args();files=json.loads((OUT/'archive_daily_files.json').read_text())
    # Duplicate jobs are recorded, largest complete file selected deterministically.
    chosen={}
    for x in files:
        if x['date']<args.start:continue
        if x['date'] not in chosen or x['size']>chosen[x['date']]['size']:chosen[x['date']]=x
    rows=[chosen[k] for k in sorted(chosen)]
    cache=OUT/'contract_days';cache.mkdir(exist_ok=True)
    pending=[x for x in rows if not (cache/(x['date']+'.parquet')).exists()]
    t=time.time();print('files_to_stage',len(pending),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(f,audit) in enumerate(pool.map(stage_one,pending,chunksize=1)):
            if audit['sha256']!=audit['expected_sha256']:raise ValueError('Archive hash mismatch '+audit['path'])
            f.to_parquet(cache/(audit['date']+'.parquet'),index=False)
            (cache/(audit['date']+'.json')).write_text(json.dumps(audit,indent=2))
            if i%100==0:print('staged',i+1,'/',len(pending),'seconds',round(time.time()-t),flush=True)
    frames=[pd.read_parquet(cache/(x['date']+'.parquet')) for x in rows]
    f=pd.concat(frames,ignore_index=True).sort_values(['ts_event','root','symbol'])
    f.to_parquet(OUT/'actual_futures_daily.parquet',index=False)
    audits=[json.loads((cache/(x['date']+'.json')).read_text()) for x in rows]
    report={'created_at':datetime.now(timezone.utc).isoformat(),'roots':ROOTS,'files':len(rows),
            'rows':len(f),'first':str(f.ts_event.min()),'last':str(f.ts_event.max()),
            'source_files_sha256_verified':len(audits),'symbols':f.groupby('root').symbol.nunique().to_dict(),
            'per_day_cache':'Immutable archive-derived rows, one file per source date',
            'selection':'All five-root outright contracts, per-file dated instrument mapping. No chosen front contract.'}
    (OUT/'actual_futures_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
