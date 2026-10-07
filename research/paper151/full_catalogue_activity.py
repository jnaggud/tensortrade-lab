"""Paper §10.3.1, volume and published open-interest filter, 2018 diagnostic."""
import json,re,sys
from pathlib import Path
import pandas as pd
import numpy as np
import databento as db
import futures_extension as engine
from full_catalogue_options import HERE,OUT,save,sha,checked_source

def stage():
    folder=OUT/'open_interest';folder.mkdir(exist_ok=True)
    sources=json.loads((OUT/'statistics_files.json').read_text());parts=[];audits=[];errors=[]
    for num,row in enumerate(sources):
        ds=re.search(r'glbx-mdp3-(\d{8})',row['path'])[1];dest=folder/(ds+'.parquet');auditpath=folder/(ds+'.audit.json')
        if dest.exists() and auditpath.exists():
            a=json.loads(auditpath.read_text())
            if sha(dest)!=a['extract_sha256']:raise ValueError('Changed OI cache')
            parts.append(pd.read_parquet(dest));audits.append(a);continue
        try:
            s=db.DBNStore.from_file(row['path']);day=pd.Timestamp(ds).date();mapping={}
            for symbol,maps in s.symbology['mappings'].items():
                if re.fullmatch(r'(ES|NQ|GC|CL|ZN)[FGHJKMNQUVXZ]\d{1,2}',symbol):
                    for m in maps:
                        if m['start_date']<=day<m['end_date']:
                            iid=int(m['symbol'])
                            if iid in mapping and mapping[iid]!=symbol:raise ValueError('Ambiguous dated symbol')
                            mapping[iid]=symbol
            kept=[];records=0
            for arr in s.to_ndarray(count=1000000):
                records+=len(arr);mask=(arr['stat_type']==9)&np.isin(arr['instrument_id'],list(mapping))
                if mask.any():kept.append(pd.DataFrame.from_records(arr[mask]))
            if kept:
                f=pd.concat(kept,ignore_index=True)
                for col in ['ts_recv','ts_ref','ts_event']:f[col]=pd.to_datetime(f[col],unit='ns',utc=True,errors='coerce')
                f['symbol']=f.instrument_id.map(mapping)
                f=f[['ts_recv','ts_ref','ts_event','symbol','instrument_id','quantity','update_action']]
            else:f=pd.DataFrame(columns=['ts_recv','ts_ref','ts_event','symbol','instrument_id','quantity','update_action'])
            a=dict(checked_source(row),extract_rows=len(f),decoded_records=records)
            f.to_parquet(dest,index=False);a['extract_sha256']=sha(dest);save(auditpath,a)
            parts.append(f);audits.append(a)
            if num%25==0:print('OI staged',num+1,'of',len(sources),'last',ds,flush=True)
        except (ValueError,FileNotFoundError) as e:errors.append({'date':ds,'reason':str(e)})
    nonempty=[f for f in parts if len(f)];result=pd.concat(nonempty,ignore_index=True).sort_values(['ts_recv','ts_ref'])
    result.to_parquet(OUT/'open_interest.parquet',index=False)
    save(OUT/'open_interest_manifest.json',{'files':len(audits),'rows':len(result),'source_audits':audits,'errors':errors,'sha256':sha(OUT/'open_interest.parquet')})

def oi_at(table,symbol,decision):
    z=table[(table.symbol==symbol)&(table.ts_recv<=decision)&(table.ts_ref<=decision)]
    if z.empty:return np.nan
    # A correction for an older session must not supersede a newer session.
    z=z.sort_values(['ts_ref','ts_recv']);r=z.iloc[-1]
    if int(r.update_action)!=1 or r.quantity<=0 or r.quantity>=2147483647 or decision-r.ts_ref>pd.Timedelta(days=7):return np.nan
    return float(r.quantity)

def build_targets(data,oi):
    dates,books,front,second,tr,ret,carry,_,_=data;targets={};audits=[]
    # Only the range supported by received records is eligible.
    for j in np.flatnonzero((dates>=pd.Timestamp('2017-12-01',tz='UTC'))&(dates<pd.Timestamp('2019-01-01',tz='UTC'))):
        w=np.zeros(5);vol=[];inter=[];reason=None
        for root in engine.ROOTS:
            symbol=front[j][root]
            recent=[books[d].get(symbol,{}).get('volume',np.nan) for d in dates[j-9:j+1]]
            now=oi_at(oi,symbol,dates[j]+pd.Timedelta(days=1));before=oi_at(oi,symbol,dates[j-5]+pd.Timedelta(days=1))
            if not np.isfinite(recent).all() or min(sum(recent[:5]),sum(recent[5:]))<=0 or not np.isfinite([now,before]).all():
                reason='Missing contemporaneously published OI or same-contract ten-bar volume';break
            vol.append(np.log(sum(recent[5:])/sum(recent[:5])));inter.append(np.log(now/before))
        if reason is None:
            active=np.argsort(-np.asarray(vol),kind='stable')[:3]
            subset=active[np.argsort(np.asarray(inter)[active],kind='stable')[:2]]
            returns=tr.iloc[j].to_numpy()/tr.iloc[j-5].to_numpy()-1
            w[subset]=returns[subset].mean()-returns[subset]
            if abs(w).sum()>0:w/=abs(w).sum()
        targets[int(j)]=w
        audits.append({'decision':str(dates[j]+pd.Timedelta(days=1)),'weights':w.tolist(),'volume_growth':vol,'oi_growth':inter,'cash_reason':reason})
    return targets,audits

def score():
    protocol=OUT/'activity_protocol.json'
    if not protocol.exists():save(protocol,{'section':'10.3.1','registered_before_scoring':True,'evaluation':'2018-01-01 through 2018-12-31; halves also reported',
       'universe':engine.ROOTS,'signal':'Top three of five by same-selected-contract five-bar volume growth, bottom two of these by five-bar OI growth; normalized loser-long/winner-short. Rounding adaptation to very small universe. Weekly rebalance.',
       'availability':'Only OI received by completed UTC-day decision; newest reference session among already received observations; <=7-day reference age. Missing any input means cash, never a filled estimate.',
       'costs':'Existing actual-contract engine, $1m capital, <=95% gross, $2.50/side plus 1/3/5 ticks, extra signal-day test. Zero cash yield.',
       'limits':'One year, five-root adaptation and daily modeled fills. Not execution-qualified; no unseen holdout.'})
    f=pd.read_parquet(HERE/'extension/actual_futures_daily.parquet');data=engine.prepare(f);oi=pd.read_parquet(OUT/'open_interest.parquet')
    targets,audits=build_targets(data,oi);save(OUT/'activity_decisions.json',audits)
    original=engine.target_weights
    def wrapper(rule,j,tr,ret,carry):
        return targets.get(j,np.zeros(5)) if rule=='FutContrarianActivity' else original(rule,j,tr,ret,carry)
    engine.target_weights=wrapper
    rows=[];errors=[];dates=data[0]
    for block,left,right in [('full','2018-01-01','2019-01-01'),('H1','2018-01-01','2018-07-01'),('H2','2018-07-01','2019-01-01')]:
        ix=np.flatnonzero((dates>=pd.Timestamp(left,tz='UTC'))&(dates<pd.Timestamp(right,tz='UTC')))
        for case,ticks,delay in [('base',1,1),('three_ticks',3,1),('five_ticks',5,1),('extra_day',1,2)]:
            try:
                m,h,fills=engine.run(data,'FutContrarianActivity',int(ix[0]),int(ix[-1]),ticks=ticks,delay=delay)
                b,_,_=engine.run(data,'PassiveFutures',int(ix[0]),int(ix[-1]),ticks=ticks,delay=delay)
                rows.append({'section':'10.3.1','strategy':'FutContrarianActivity','window':block,'case':case,**m,
                  'benchmark_return':b['total_return'],'benchmark_max_drawdown':b['max_drawdown'],
                  'pass':bool(m['total_return']>b['total_return'] and m['max_drawdown']<=b['max_drawdown']),'qualified':False})
                if case=='base' and block=='full':
                    pd.DataFrame(h).to_parquet(OUT/'activity_equity.parquet',index=False);save(OUT/'activity_fills.json',fills)
            except ValueError as e:errors.append({'window':block,'case':case,'reason':str(e)})
    save(OUT/'activity_results.json',rows);save(OUT/'activity_errors.json',errors)
    save(OUT/'activity_manifest.json',{'runs':len(rows),'benchmark_runs':len(rows),'failed_closed':len(errors),
       'code_sha256':sha(__file__),'oi_sha256':sha(OUT/'open_interest.parquet'),'prices_sha256':sha(HERE/'extension/actual_futures_daily.parquet'),
       'decision_days':len(audits),'missing_input_cash_days':sum(a['cash_reason'] is not None for a in audits),'qualified':0})
if __name__=='__main__':{'stage':stage,'score':score}[sys.argv[1]]()
