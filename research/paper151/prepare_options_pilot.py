"""Small ES weekly-options bid/ask panel, known-before-decision definitions."""
from intraday_common import *
from stage_intraday_contracts import sha as stream_sha
import databento as db
import re,time

def definitions(row):
    s=db.DBNStore.from_file(row['path']);parts=[]
    for f in s.to_df(map_symbols=False,count=200000):
        z=f[(f.underlying.str.match(r'^ES[HMUZ]\d{1,2}$',na=False)|f.raw_symbol.str.match(r'^ES[HMUZ]\d{1,2}$',na=False)) & f.instrument_class.isin(['F','C','P'])]
        if len(z):parts.append(z)
    x=pd.concat(parts).reset_index()
    actual=stream_sha(row['path'])
    if actual!=row['sha256']:raise ValueError('Definitions hash mismatch')
    return x

def main():
    files=json.loads((OUT/'options_pilot_files.json').read_text());by={(x['schema'],x['date']):x for x in files}
    decision=pd.Timestamp('2026-06-01T13:59:00Z');d=definitions(by['definition','20260601']);d.to_parquet(OUT/'es_options_definitions_pilot.parquet',index=False)
    known=d[d.ts_recv<=decision].sort_values('ts_recv').drop_duplicates('instrument_id',keep='last')
    known=known[(known.security_update_action!='D') & (known.expiration>decision)]
    # Explicitly European Friday weeklies, never classify quarterly ESM options by a generic group code.
    options=known[known.raw_symbol.str.match(r'^EW[1-5][FGHJKMNQUVXZ]\d{1,2} [CP]\d',na=False)]
    options=options[(options.expiration>=decision+pd.Timedelta(days=9)) & (options.expiration<=decision+pd.Timedelta(days=28))]
    if options.empty:raise ValueError('No verified eligible European weeklies')
    expiry=options.expiration.min();options=options[options.expiration==expiry]
    underlying=options.underlying.unique()
    if len(underlying)!=1:raise ValueError('Ambiguous underlying future')
    u=known[(known.raw_symbol==underlying[0])&(known.instrument_class=='F')]
    if len(u)!=1:raise ValueError('Underlying definition missing')
    history=pd.read_parquet(HERE/'extension/actual_futures_daily.parquet')
    prev=history[(history.symbol==underlying[0])&(history.ts_event+pd.Timedelta(days=1)<=decision)].sort_values('ts_event').iloc[-1]
    spot=float(prev.close)
    call=options[(options.instrument_class=='C')&(options.strike_price>=spot*1.02)].sort_values('strike_price').iloc[0]
    put=options[(options.instrument_class=='P')&(options.strike_price<=spot*.98)].sort_values('strike_price',ascending=False).iloc[0]
    selected=pd.DataFrame([u.iloc[0],call,put])
    if not np.allclose(selected.unit_of_measure_qty,50):raise ValueError('Multiplier not verified')
    ids=set(selected.instrument_id.astype(int));selected.to_parquet(OUT/'options_pilot_contracts.parquet',index=False)
    save('options_pilot_protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),'decision_at':str(decision),'start_execution':'2026-06-01T14:00:00Z','end_execution':'2026-06-05T19:55:00Z',
      'prior_underlying_price':spot,'prior_price_day':str(prev.ts_event),'expiry':str(expiry),
      'contracts':selected[['raw_symbol','instrument_id','instrument_class','expiration','underlying','underlying_id','strike_price','unit_of_measure_qty','ts_recv']].to_dict('records'),
      'selection':'Earliest eligible European Friday weekly 9-28 days away, 2% OTM call and 2% OTM put from last completed underlying close. Close five-day pilot before expiry.',
      'rules':['one fully collateralized ES future plus one short call','one short put reserved at strike*50'],
      'capital':1_000_000,'execution':'Next-minute observed bid/ask, visible size >=1, quote timestamps aligned within one minute, no midpoint fills. Entry at 14:00 UTC, close Friday 19:55 UTC. Only prior known definitions.',
      'costs':'$2.50 per contract side plus observed spread; stress one extra 0.25 index-point adverse tick per leg, explicitly modeled. Zero collateral yield.',
      'limits':'One-week plumbing pilot, not qualification. Fully funded option-on-futures adaptations, not stock covered calls. European expiry avoided; no American early-assignment assumption. BBO interval may carry forward unchanged quotes; update-age/latency not independently known.',
      'references':['https://www.cmegroup.com/trading/equity-index/weekly-eom-options-faq.html','https://databento.com/docs/schemas-and-data-formats/bbo']})
    print('Selected',selected.raw_symbol.tolist(),'expiry',expiry,flush=True)
    all_frames=[];audits=[];start=time.time()
    for date in ['20260601','20260602','20260603','20260604','20260605']:
        output=OUT/('options_quotes_'+date+'.parquet')
        if output.exists() and (OUT/('options_quotes_'+date+'.audit.json')).exists():all_frames.append(pd.read_parquet(output));continue
        row=by['bbo-1m',date];s=db.DBNStore.from_file(row['path']);kept=[];records=0
        for contract in selected.itertuples():
            mapped=[int(x['symbol']) for x in s.symbology['mappings'].get(contract.raw_symbol,[]) if x['start_date']<=pd.Timestamp(date).date()<x['end_date']]
            if mapped!=[contract.instrument_id]:raise ValueError('Dated option/underlying mapping changed: '+contract.raw_symbol)
        for a in s.to_ndarray(count=1_000_000):
            records+=len(a);mask=np.isin(a['instrument_id'],list(ids));b=a[mask]
            if len(b):kept.append(pd.DataFrame.from_records(b))
        if not kept:raise ValueError('Selected quote IDs absent '+date)
        f=pd.concat(kept,ignore_index=True)
        for col in ['ts_recv','ts_event']:f[col]=pd.to_datetime(f[col],unit='ns',utc=True,errors='coerce')
        for col in ['bid_px_00','ask_px_00']:f[col]=f[col].astype(float)/1e9
        f=f[(f.ts_recv.dt.hour>=13)&(f.ts_recv.dt.hour<=20)]
        f.to_parquet(output,index=False)
        audit={'date':date,'source':row['path'],'sha256':stream_sha(row['path']),'expected_sha256':row['sha256'],'decoded_records':records,'selected_rows':len(f),
               'hash_status':'matches_recorded_source_hash' if row['sha256'] else 'newly_computed_baseline_no_hash_in_original_manifest'}
        if Path(row['path']).stat().st_size!=row['size']:raise ValueError('Source size mismatch')
        if audit['expected_sha256'] and audit['sha256']!=audit['expected_sha256']:raise ValueError('BBO source checksum mismatch')
        save('options_quotes_'+date+'.audit.json',audit);all_frames.append(f);audits.append(audit)
        print('options staged',date,len(f),'seconds',round(time.time()-start),flush=True)
    f=pd.concat(all_frames,ignore_index=True).sort_values(['ts_recv','instrument_id']);f.to_parquet(OUT/'options_pilot_quotes.parquet',index=False)
    save('options_pilot_data_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'files':5,'rows':len(f),'sha256':stream_sha(OUT/'options_pilot_quotes.parquet'),'contract_sha256':stream_sha(OUT/'options_pilot_contracts.parquet'),'source_audits':[str(OUT/('options_quotes_'+d+'.audit.json')) for d in ['20260601','20260602','20260603','20260604','20260605']]})
if __name__=='__main__':main()
