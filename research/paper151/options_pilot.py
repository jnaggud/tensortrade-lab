"""Observed-bid/ask, five-day fully collateralized options-on-futures pilot."""
from intraday_common import *

def valid_quotes(f):
    return (np.isfinite(f.bid_px_00)&np.isfinite(f.ask_px_00)&(f.bid_px_00>=0)&(f.ask_px_00>=f.bid_px_00)&(f.ask_px_00<1e6)&(f.bid_sz_00>=1)&(f.ask_sz_00>=1))

def synchronized(f,ids,time,tolerance=pd.Timedelta(minutes=1)):
    out={}
    for iid in ids:
        z=f[(f.instrument_id==iid)&(f.ts_recv<=time)&(f.ts_recv>=time-tolerance)]
        if z.empty:raise ValueError(f'Missing fresh, sized quote {iid} at {time}')
        r=z.iloc[-1]
        if not valid_quotes(pd.DataFrame([r])).iloc[0]:raise ValueError('Invalid bid/ask or size')
        out[iid]=r
    if max(z.ts_recv for z in out.values())-min(z.ts_recv for z in out.values())>tolerance:raise ValueError('Legs not synchronized')
    return out

def run(f,contracts,kind,stress=0):
    u=contracts[contracts.instrument_class=='F'].iloc[0];option=contracts[contracts.instrument_class==('C' if kind=='covered_call' else 'P')].iloc[0]
    ids=[int(u.instrument_id),int(option.instrument_id)];start=pd.Timestamp('2026-06-01T14:00:00Z');end=pd.Timestamp('2026-06-05T19:55:00Z')
    if pd.Timestamp(option.expiration)<=end:raise ValueError('Expiry would require explicit lifecycle handling')
    first=synchronized(f,ids,start);last=synchronized(f,ids,end)
    # Model one full contract, known $50 per index point, funded by $1m.
    capital=1e6;mult=50.;tick=.25*stress;fee=2.5
    u_entry=float(first[ids[0]].ask_px_00)+tick;premium=float(first[ids[1]].bid_px_00)-tick
    if premium<0:raise ValueError('Negative premium after stress')
    reserve=u_entry*mult if kind=='covered_call' else float(option.strike_price)*mult
    if reserve+2*fee>capital:raise ValueError('Insufficient full notional reserve')
    entry_fee=2*fee if kind=='covered_call' else fee;exit_fee=entry_fee
    history=[];gaps=[]
    timestamps=sorted(set(f.loc[(f.ts_recv>=start)&(f.ts_recv<=end),'ts_recv']))
    for t in timestamps:
        try:q=synchronized(f,ids,t)
        except ValueError:gaps.append(str(t));continue
        # Liquidation-side equity: buy back short option at ask, sell held future at bid.
        option_cost=float(q[ids[1]].ask_px_00)+tick
        net_option=(premium-option_cost)*mult
        fut_pnl=(float(q[ids[0]].bid_px_00)-tick-u_entry)*mult
        pnl=net_option+(fut_pnl if kind=='covered_call' else 0)-entry_fee-exit_fee
        benchmark=fut_pnl-2*fee
        history.append({'timestamp':t,'equity':capital+pnl,'benchmark_equity':capital+benchmark})
    if not history or history[-1]['timestamp']!=end:raise ValueError('Missing terminal synchronized quote')
    h=pd.DataFrame(history);eq=np.r_[capital,h.equity];bh=np.r_[capital,h.benchmark_equity]
    dd=lambda x:float((1-x/np.maximum.accumulate(x)).max())
    final=float(h.equity.iloc[-1]/capital-1);base=float(h.benchmark_equity.iloc[-1]/capital-1)
    result={'strategy':kind,'case':'base' if stress==0 else 'extra_adverse_tick','initial_capital':capital,'total_return':final,'benchmark_return':base,
      'max_sampled_drawdown':dd(eq),'benchmark_max_sampled_drawdown':dd(bh),'quotes_marked':len(h),'missing_synchronized_timestamps':len(gaps),
      'fees':entry_fee+exit_fee,'start':str(start),'end':str(end),'reserve':reserve,'contract':option.raw_symbol,
      'entry_premium_points':premium,'exit_option_ask_points':float(last[ids[1]].ask_px_00)+tick,
      'pass_in_pilot_only':bool(final>base and dd(eq)<=dd(bh)),'qualified':False,
      'limits':'Five-day pilot, first printed quotes assumed fillable for size one; 1m snapshot age/latency unverified. Marks only observed 13:00-20:59 UTC quote windows. Not a long historical validation.'}
    return result,h,gaps

def main():
    f=pd.read_parquet(OUT/'options_pilot_quotes.parquet').sort_values('ts_recv');contracts=pd.read_parquet(OUT/'options_pilot_contracts.parquet')
    save('options_quote_quality.json',{'rows':len(f),'valid_bid_ask_sized_rows':int(valid_quotes(f).sum()),'duplicates':int(f.duplicated(['ts_recv','instrument_id']).sum()),
      'quote_clock':'ts_recv is minute interval end. ts_event is last trade timestamp and NOT the quote timestamp.',
      'contract_identifiers':contracts[['raw_symbol','instrument_id','underlying','underlying_id']].to_dict('records')})
    if f.duplicated(['ts_recv','instrument_id']).any():raise ValueError('Duplicate option quote')
    rows=[];errors=[]
    for kind in ['covered_call','cash_secured_put']:
        for stress in [0,1]:
            try:
                r,h,gaps=run(f,contracts,kind,stress);rows.append(r);h.to_parquet(OUT/(kind+('_base' if not stress else '_stress')+'_equity.parquet'),index=False);save(kind+'_missing_marks.json',gaps)
            except ValueError as e:errors.append({'strategy':kind,'stress':stress,'reason':str(e)})
    save('options_pilot_results.json',rows);save('options_pilot_errors.json',errors)
    save('options_pilot_manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'completed_runs':len(rows),'failed_closed_runs':len(errors),'qualified':0,'code_sha256':sha(__file__),'data_sha256':sha(OUT/'options_pilot_quotes.parquet'),'protocol_sha256':sha(OUT/'options_pilot_protocol.json')})
    print(rows,errors,flush=True)
if __name__=='__main__':main()
