"""Paper §2 leg registry and fixed, historical ES weekly-options samples.

No model-generated premiums, tuning, or changes to frozen prospective work.
All options-on-futures rules are adaptations of the paper's stock constructions.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys
import numpy as np
import pandas as pd
import databento as db

HERE = Path(__file__).resolve().parent
OUT = HERE / 'full_catalogue'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)):return [clean(v) for v in x]
        if isinstance(x, (float,np.floating)) and not np.isfinite(x):return None
        if isinstance(x, np.integer):return int(x)
        return x
    path.write_text(json.dumps(clean(obj), indent=2, default=str, allow_nan=False)+'\n')

def registry():
    # Leg tuple = class, strike offset in half-widths, expiry slot, quantity.
    C = lambda k, q=1, far=0: ('C', k, far, q)
    P = lambda k, q=1, far=0: ('P', k, far, q)
    U = lambda q=1: ('F', 0, 0, q)
    r = {
      '2.2':[U(),C(2,-1)], '2.3':[U(-1),P(-2,-1)],
      '2.4':[U(),P(-2)], '2.5':[U(-1),C(2)],
      '2.6':[C(-2),C(2,-1)], '2.7':[P(-2),P(2,-1)],
      '2.8':[C(-2,-1),C(2)], '2.9':[P(-2,-1),P(2)],
      '2.10':[C(0),P(0,-1)], '2.11':[C(0,-1),P(0)],
      '2.12':[C(2),P(-2,-1)], '2.13':[C(2,-1),P(-2)],
      '2.14':[C(0),C(2,-1),C(4,-1)],
      '2.15':[P(-4),P(-2),P(0,-1)],
      '2.16':[C(0,-1),C(2),C(4)],
      '2.17':[P(-4,-1),P(-2,-1),P(0)],
      '2.18':[C(0,1,1),C(0,-1)], '2.19':[P(0,1,1),P(0,-1)],
      '2.20':[C(-4,1,1),C(2,-1)], '2.21':[P(4,1,1),P(-2,-1)],
      '2.22':[C(0),P(0)], '2.23':[C(2),P(-2)], '2.24':[C(-2),P(2)],
      '2.25':[C(0,-1),P(0,-1)], '2.26':[C(2,-1),P(-2,-1)],
      '2.27':[C(-2,-1),P(2,-1)],
      '2.28':[C(0,2),U(-1)], '2.29':[P(0,2),U()],
      '2.30':[C(0,-2),U()], '2.31':[P(0,-2),U(-1)],
      '2.32':[U(),C(0,-1),P(0,-1)], '2.33':[U(),C(2,-1),P(-2,-1)],
      '2.34':[C(0,2),P(0)], '2.35':[C(0),P(0,2)],
      '2.36':[C(0,-1),C(2,2)], '2.37':[P(0,-1),P(-2,2)],
      '2.38':[C(-2),C(0,-2)], '2.39':[P(2),P(0,-2)],
      '2.40':[C(-2),C(0,-2),C(2)],
      '2.40.1':[C(-4),C(0,-2),C(2)],
      '2.41':[P(-2),P(0,-2),P(2)],
      '2.41.1':[P(-4),P(0,-2),P(2)],
      '2.42':[C(-2,-1),C(0,2),C(2,-1)],
      '2.43':[P(-2,-1),P(0,2),P(2,-1)],
      # Preserve the PAPER'S iron-spread naming, which is unconventional.
      '2.44':[P(-2),P(0,-1),C(0,-1),C(2)],
      '2.45':[P(-2,-1),P(0),C(0),C(2,-1)],
      '2.46':[C(-3),C(-1,-1),C(1,-1),C(3)],
      '2.47':[P(-3),P(-1,-1),P(1,-1),P(3)],
      '2.48':[C(-3,-1),C(-1),C(1),C(3,-1)],
      '2.49':[P(-3,-1),P(-1),P(1),P(3,-1)],
      '2.50':[P(-3),P(-1,-1),C(1,-1),C(3)],
      '2.51':[P(-3,-1),P(-1),C(1),C(3,-1)],
      '2.52':[C(-2),C(2,-1),P(2),P(-2,-1)],
      '2.53':[U(),P(-2),C(2,-1)],
      '2.54':[P(-2,-1),C(0),C(2,-1)],
      '2.55':[P(-2),C(0,-1),C(2)],
      '2.56':[P(-2,-1),P(0),C(2,-1)],
      '2.57':[P(-2),P(0,-1),C(2)],
    }
    assert len(r)==58
    return r

def payoff(legs, spot, center=100., halfwidth=5.):
    s=np.asarray(spot, float); out=np.zeros_like(s)
    for kind,k,far,q in legs:
        if far: raise ValueError('Different expiries need time-specific valuations')
        strike=center+k*halfwidth
        out += q*(s-center if kind=='F' else np.maximum(s-strike,0) if kind=='C' else np.maximum(strike-s,0))
    return out

def episodes():
    dates=[]
    for month in pd.period_range('2025-07','2026-06',freq='M'):
        d=month.start_time.tz_localize('UTC');d+=pd.Timedelta(days=(-d.weekday())%7)
        dates.append(d)
    return dates

def register():
    OUT.mkdir(exist_ok=True)
    p=OUT/'options_protocol.json'
    if p.exists(): return
    save(p, {'registered_at':datetime.now(timezone.utc).isoformat(),
      'episodes':[str(d.date()) for d in episodes()], 'entry_utc':'Monday 14:00', 'exit_utc':'Wednesday 19:55',
      'decision_utc':'Monday 13:59; definitions and completed previous daily close only',
      'selection':'Earliest Friday European weekly expiry 9-35 days away. Far expiry is next eligible Friday weekly <=63 days away on exactly the same underlying future. No substitute when missing.',
      'strikes':'Closest listed common call/put near-expiry center to prior close, with a complete nine-strike arithmetic grid. Half-width chosen from positive multiples of 5, nearest 0.5% of prior close, capped at 1.5%; centers limited to 0.5% from prior close. Selection uses ONLY known definitions. Exact registry offsets and equal spacing preserved. Required far-expiry legs must exist at these same strikes, else fail closed.',
      'availability_amendment':'Version 2 corrects five-point rounding that requested unlisted contracts. Recorded before ANY option return scoring. Original protocol and selections/extracts preserved under options_selection_v1; no outcome-dependent selection.',
      'sizes':'One unit per registry, ratio legs two units. One ES future benchmark, $50 per point, $1m initial reference cash in each independent episode; zero cash yield.',
      'costs':'Observed bid/ask plus $2.50 per contract side. Stress one additional adverse 0.25-point tick per fill. Quotes require displayed size >= abs(quantity).',
      'marking':'Every 5 minutes, 14:00-19:55 UTC on each of the three days. Quotes no older than one minute; align within one minute. No fill across missing/invalid quotes. Missing intermediate marks counted; endpoint missing fails case.',
      'qualification':'Always false. Twelve three-day samples are episode research, not continuous-year validation. No dated SPAN margins, latency, simultaneous multi-leg fills or exercised/assigned lifecycle. Some short structures have unbounded loss; $1m does not prove collateral adequacy.',
      'comparison':'After-cost terminal return and sampled maximum drawdown vs long one identical ES future during identical episode and available marks. No annualized returns, no claim of equivalently risky portfolios.',
      'registry':registry(), 'paper':'ssrn-3247865.pdf, sections 2.2-2.57 incl 2.40.1 and 2.41.1',
      'adaptations':'Stock legs replaced by their ES futures underlying. All positions liquidated before expiry; calendars do not roll the short leg. No directional entry filter is implied by a payoff recipe.'})

def source_files():
    return {(r['schema'],r['date']):r for r in json.loads((OUT/'archive_files.json').read_text())}

def checked_source(row):
    if Path(row['path']).stat().st_size!=row['size']: raise ValueError('Source size mismatch '+row['path'])
    actual=sha(row['path'])
    if row.get('sha256') and actual!=row['sha256']: raise ValueError('Source hash mismatch '+row['path'])
    return {'path':row['path'],'sha256':actual,'expected_sha256':row.get('sha256'),
            'hash_status':'matched_prior_hash' if row.get('sha256') else 'new_baseline_original_hash_absent'}

def select(day, row, history):
    folder=OUT/'options'/day.strftime('%Y%m%d');folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'definitions.parquet'
    if not cache.exists():
        parts=[]
        for f in db.DBNStore.from_file(row['path']).to_df(map_symbols=False,count=200000):
            mask=(f.underlying.str.match(r'^ES[HMUZ]\d{1,2}$',na=False)|f.raw_symbol.str.match(r'^ES[HMUZ]\d{1,2}$',na=False))&f.instrument_class.isin(['F','C','P'])
            if mask.any():parts.append(f[mask])
        if not parts:raise ValueError('No ES definitions')
        d=pd.concat(parts).reset_index();audit=checked_source(row)
        d.to_parquet(cache,index=False);save(folder/'definitions_audit.json',dict(audit,extract_sha256=sha(cache)))
    else:
        audit=json.loads((folder/'definitions_audit.json').read_text())
        if sha(cache)!=audit['extract_sha256']:raise ValueError('Definition cache hash mismatch')
        d=pd.read_parquet(cache)
    decision=day+pd.Timedelta(hours=13,minutes=59)
    known=d[d.ts_recv<=decision].sort_values('ts_recv').drop_duplicates('instrument_id',keep='last')
    known=known[(known.security_update_action!='D')&(known.expiration>decision)]
    options=known[known.raw_symbol.str.match(r'^EW[1-5][FGHJKMNQUVXZ]\d{1,2} [CP]\d',na=False)]
    options=options[(options.expiration>=decision+pd.Timedelta(days=9))&(options.expiration<=decision+pd.Timedelta(days=63))]
    near=options[options.expiration<=decision+pd.Timedelta(days=35)].expiration.min()
    if pd.isna(near):raise ValueError('No eligible near weekly expiry')
    n=options[options.expiration==near];under=n.underlying.unique()
    if len(under)!=1:raise ValueError('Ambiguous underlying future')
    options=options[options.underlying==under[0]]
    far=options[options.expiration>near].expiration.min()
    u=known[(known.raw_symbol==under[0])&(known.instrument_class=='F')]
    if len(u)!=1:raise ValueError('Missing unique underlying definition')
    h=history[(history.symbol==under[0])&(history.ts_event+pd.Timedelta(days=1)<=decision)].sort_values('ts_event')
    if h.empty:raise ValueError('No previous completed underlying close')
    prev=h.iloc[-1];spot=float(prev.close)
    near_chain=options[options.expiration==near]
    common=set(near_chain.loc[near_chain.instrument_class=='C','strike_price']) & set(near_chain.loc[near_chain.instrument_class=='P','strike_price'])
    centers=sorted((k for k in common if abs(k/spot-1)<=.005),key=lambda k:(abs(k-spot),k))
    widths=sorted(np.arange(5,spot*.015+1,5),key=lambda w:(abs(w-spot*.005),w))
    grid=next(((k,w) for k in centers for w in widths if all(k+n*w in common for n in range(-4,5))),None)
    if grid is None:raise ValueError('No complete, prior-listed near-expiry arithmetic strike grid within fixed bounds')
    center,width=map(float,grid)
    need=set((a,b,c) for legs in registry().values() for a,b,c,q in legs)
    selected=[];mapping={};missing={}
    cols=['raw_symbol','instrument_id','instrument_class','expiration','underlying','underlying_id','strike_price','unit_of_measure_qty','ts_recv']
    for kind,k,slot in sorted(need):
        z=u if kind=='F' else options[(options.instrument_class==kind)&(options.expiration==(far if slot else near))&(abs(options.strike_price-(center+k*width))<1e-7)]
        key=f'{kind}:{k}:{slot}'
        if len(z)!=1:missing[key]=f'{len(z)} definitions at required strike/expiry';continue
        r=z.iloc[0]
        if not np.isclose(float(r.unit_of_measure_qty),50):raise ValueError('Unverified multiplier')
        selected.append(r[cols].to_dict());mapping[key]=int(r.instrument_id)
    save(folder/'selection.json',{'decision_at':str(decision),'near':str(near),'far':str(far),'underlying':under[0],
        'prior_close':spot,'prior_close_day':str(prev.ts_event),'center':center,'half_width':width,
        'mapping':mapping,'missing_definitions':missing,'contracts':selected})
    return folder,selected

def stage():
    register();files=source_files();history=pd.read_parquet(HERE/'extension/actual_futures_daily.parquet');errors=[]
    for day in episodes():
        try:
            folder,contracts=select(day,files['definition',day.strftime('%Y%m%d')],history)
            ids=[int(r['instrument_id']) for r in contracts]
            for date in pd.date_range(day,periods=3):
                ds=date.strftime('%Y%m%d');dest=folder/f'quotes_{ds}.parquet';ap=folder/f'quotes_{ds}.audit.json'
                if dest.exists() and ap.exists():
                    if sha(dest)!=json.loads(ap.read_text())['extract_sha256']:raise ValueError('Quote cache changed')
                    continue
                row=files['bbo-1m',ds];s=db.DBNStore.from_file(row['path'])
                for c in contracts:
                    mapped=[int(x['symbol']) for x in s.symbology['mappings'].get(c['raw_symbol'],[]) if x['start_date']<=date.date()<x['end_date']]
                    if mapped!=[int(c['instrument_id'])]:raise ValueError('Dated instrument mapping mismatch '+c['raw_symbol'])
                # Reuse only immutable v1 extracts that actually contain every new ID.
                old=OUT/'options_selection_v1'/'options'/folder.name/dest.name
                oldaudit=old.with_suffix('.audit.json')
                if old.exists() and oldaudit.exists():
                    audit=json.loads(oldaudit.read_text())
                    if sha(old)!=audit['extract_sha256']:raise ValueError('Archived v1 quote cache changed')
                    prior=pd.read_parquet(old)
                    if set(ids)<=set(prior.instrument_id.astype(int)):
                        f=prior[prior.instrument_id.isin(ids)].copy();f.to_parquet(dest,index=False)
                        save(ap,dict(audit,extract_sha256=sha(dest),selected_rows=len(f),reused_from=str(old)))
                        print(json.dumps({'reused':ds,'episode':str(day.date()),'rows':len(f),'contracts':len(ids)}),flush=True)
                        continue
                parts=[];count=0
                for a in s.to_ndarray(count=1000000):
                    count+=len(a);mask=np.isin(a['instrument_id'],ids)
                    if mask.any():parts.append(pd.DataFrame.from_records(a[mask]))
                if not parts:raise ValueError('No selected quotes')
                f=pd.concat(parts,ignore_index=True)
                for col in ['ts_recv','ts_event']:f[col]=pd.to_datetime(f[col],unit='ns',utc=True,errors='coerce')
                for col in ['bid_px_00','ask_px_00']:f[col]=f[col].astype(float)/1e9
                f=f[(f.ts_recv.dt.hour>=13)&(f.ts_recv.dt.hour<=20)].sort_values(['ts_recv','instrument_id'])
                if f.duplicated(['ts_recv','instrument_id']).any():raise ValueError('Duplicate quote records')
                audit=checked_source(row);f.to_parquet(dest,index=False)
                save(ap,dict(audit,extract_sha256=sha(dest),decoded_records=count,selected_rows=len(f)))
                print(json.dumps({'staged':ds,'episode':str(day.date()),'rows':len(f),'contracts':len(ids)}),flush=True)
        except (ValueError,KeyError,FileNotFoundError) as e:
            errors.append({'episode':str(day.date()),'reason':str(e)});print('STAGE FAILURE',errors[-1],flush=True)
    save(OUT/'options_stage_errors.json',errors)

def quotes_at(f, ids, quantities, grid):
    out={};valid=np.ones(len(grid),dtype=bool)
    for iid,q in zip(ids,quantities):
        z=f[f.instrument_id==iid].sort_values('ts_recv')
        a=pd.merge_asof(pd.DataFrame({'mark':grid}),z,left_on='mark',right_on='ts_recv',direction='backward',tolerance=pd.Timedelta(minutes=1))
        ok=(a.bid_px_00>=0)&(a.ask_px_00>=a.bid_px_00)&(a.ask_px_00<1e6)&(a.bid_sz_00>=max(1,abs(q)))&(a.ask_sz_00>=max(1,abs(q)))
        valid &= ok.to_numpy();out[iid]=a
    return out,valid

def run_one(f,selection,legs,day,stress):
    mapping=selection['mapping'];ids=[];qty=[]
    for kind,k,slot,q in legs:
        key=f'{kind}:{k}:{slot}'
        if key not in mapping:raise ValueError('Required contract missing: '+key)
        ids.append(mapping[key]);qty.append(q)
    if len(set(ids))!=len(ids):raise ValueError('Duplicate legs must be netted')
    uid=mapping['F:0:0'];all_ids=list(ids);all_q=list(qty)
    if uid not in ids:all_ids.append(uid);all_q.append(1)
    grid=pd.DatetimeIndex([t for d in pd.date_range(day,periods=3) for t in pd.date_range(d+pd.Timedelta(hours=14),d+pd.Timedelta(hours=19,minutes=55),freq='5min')])
    quotes,valid=quotes_at(f,all_ids,all_q,grid)
    if not valid[0] or not valid[-1]:raise ValueError('Missing sized/synchronous entry or exit quote')
    # Entire position is hypothetically liquidated at each valid observed mark.
    tick=.25*stress;fee=2.5;capital=1e6
    pnl=np.zeros(len(grid));fills=[]
    for iid,q in zip(ids,qty):
        z=quotes[iid];entry=float(z.ask_px_00.iloc[0]+tick if q>0 else z.bid_px_00.iloc[0]-tick)
        close=z.bid_px_00.to_numpy()-tick if q>0 else z.ask_px_00.to_numpy()+tick
        if entry<0 or np.any(close[valid]<0):raise ValueError('Negative fill under adverse tick')
        pnl+=q*(close-entry)*50-2*abs(q)*fee
        fills.extend([{'at':str(grid[0]),'instrument_id':iid,'quantity':q,'price':entry,'fee':abs(q)*fee},
                      {'at':str(grid[-1]),'instrument_id':iid,'quantity':-q,'price':float(close[-1]),'fee':abs(q)*fee}])
    b=quotes[uid];bh=(b.bid_px_00.to_numpy()-tick-float(b.ask_px_00.iloc[0])-tick)*50-2*fee
    eq=np.r_[capital,capital+pnl[valid]];base=np.r_[capital,capital+bh[valid]]
    dd=lambda a:float(np.max(1-a/np.maximum.accumulate(a)))
    result={'return':float(eq[-1]/capital-1),'max_sampled_drawdown':dd(eq),'benchmark_return':float(base[-1]/capital-1),
        'benchmark_max_sampled_drawdown':dd(base),'marks':int(valid.sum()),'missing_marks':int((~valid).sum()),
        'fees':sum(x['fee'] for x in fills),'pass_in_episode_only':bool(eq[-1]>base[-1] and dd(eq)<=dd(base)),
        'qualified':False,'ever_insolvent':bool(eq.min()<=0)}
    # Independent terminal cash-flow reconciliation from signed fill records.
    net=sum(-x['quantity']*x['price']*50-x['fee'] for x in fills)
    if not np.isclose(net,pnl[-1],atol=1e-7,rtol=0):raise ValueError('Fill cash-flow reconciliation failed')
    h=pd.DataFrame({'timestamp':grid[valid],'equity':capital+pnl[valid],'benchmark_equity':capital+bh[valid]})
    return result,h,fills

def score():
    rows=[];errors=[]
    for day in episodes():
        folder=OUT/'options'/day.strftime('%Y%m%d')
        try:
            s=json.loads((folder/'selection.json').read_text())
            f=pd.concat([pd.read_parquet(folder/f'quotes_{d.strftime("%Y%m%d")}.parquet') for d in pd.date_range(day,periods=3)],ignore_index=True)
        except (FileNotFoundError,ValueError) as e:
            for section in registry():
                for case in ['base','extra_tick']:errors.append({'section':section,'episode':str(day.date()),'case':case,'reason':str(e)})
            continue
        for section,legs in registry().items():
            for stress in [0,1]:
                case='extra_tick' if stress else 'base';key={'section':section,'episode':str(day.date()),'case':case}
                try:
                    r,h,fills=run_one(f,s,legs,day,stress);rows.append(dict(key,**r))
                    name=section.replace('.','_')+'_'+case
                    h.to_parquet(folder/(name+'_equity.parquet'),index=False);save(folder/(name+'_fills.json'),fills)
                except ValueError as e:errors.append(dict(key,reason=str(e)))
        print('Scored',str(day.date()),len(rows),'completed',len(errors),'failed closed',flush=True)
    save(OUT/'options_results.json',rows);save(OUT/'options_errors.json',errors)
    save(OUT/'options_manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'completed':len(rows),'failed_closed':len(errors),
      'catalogue_entries_with_completed_base_case':len(set(r['section'] for r in rows if r['case']=='base')),
      'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'options_protocol.json'),'qualified':0,
      'data_audits':{str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'options').glob('**/*.audit.json'))},
      'results_sha256':sha(OUT/'options_results.json')})

if __name__=='__main__':
    {'register':register,'stage':stage,'score':score}[sys.argv[1]]()
