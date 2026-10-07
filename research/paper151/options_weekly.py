"""A separately registered 52-week extension. Original episodes stay immutable."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import sys
import numpy as np
import pandas as pd
import full_catalogue_options as old

HERE = Path(__file__).resolve().parent
OUT = HERE / 'options_weekly'
DAYS = pd.date_range('2025-06-30', '2026-06-22', freq='W-MON', tz='UTC')

def register():
    OUT.mkdir(exist_ok=True)
    protocol = OUT / 'protocol.json'
    if protocol.exists():
        p = json.loads(protocol.read_text())
        if p['engine_sha256'] != old.sha(old.__file__):
            raise ValueError('Registered options engine changed')
        return
    old.save(protocol, {
        'registered_at': datetime.now(timezone.utc).isoformat(),
        'weeks': [str(d.date()) for d in DAYS], 'entries': len(old.registry()),
        'engine_sha256': old.sha(old.__file__),
        'parent_protocol_sha256': old.sha(HERE/'full_catalogue/options_protocol.json'),
        'timing': 'Every Monday 14:00 UTC to Wednesday 19:55 UTC. Flat between episodes. Same prior-listed v2 strike selection; no outcome-based choice.',
        'capital': 'One construction unit each week, $1m starting cash per strategy. Carry ALL realized profits/losses into the next week; no fresh capital. Zero interest. No compounding position size.',
        'missing_data': 'Unknown definitions or absent valid sized entry quote: do not enter; cash that week. Once entered, missing scheduled exit is an accounting failure: HALT the continuous series there; never omit the losing or unknown week and resume. Missing intraperiod marks counted.',
        'benchmark': 'Separate scheduled one-ES-future portfolio, every week with valid entry and exit; same costs and start/end timetable. This is an exposure comparator, not fully invested SPY buy-and-hold. No qualification against the user buy-and-hold gate.',
        'costs': 'Observed bid/ask + $2.50 per contract per side; separate adverse extra 0.25-point tick case. Never fill below zero.',
        'limitations': 'Continuous capital account for an intermittent weekly strategy, NOT continuous risk observation. Daytime five-minute marks only; overnight drawdown, dated SPAN collateral, latency, quote-update age and simultaneous fills unverified. All results exploratory; naked losses unbounded.',
        'qualification': False,
    })

def stage():
    register()
    files = old.source_files()
    # Reuse previously verified exact selection/quote extracts only.
    for day in DAYS:
        source = HERE/'full_catalogue/options'/day.strftime('%Y%m%d')
        dest = OUT/'options'/source.name
        if source.exists():
            dest.mkdir(parents=True, exist_ok=True)
            for p in source.iterdir():
                if p.name.startswith(('definitions', 'quotes_', 'selection')) and not (dest/p.name).exists():
                    shutil.copy2(p, dest/p.name)
    old.OUT = OUT
    old.episodes = lambda: DAYS
    old.register = lambda: None
    old.source_files = lambda: files
    old.stage()

def combine(curves, initial=1e6):
    """Dollar P&L chaining, never arithmetic addition of episode returns."""
    cash = initial
    result = []
    for h in curves:
        adjusted = h.copy()
        adjusted['equity'] = cash + h.equity.to_numpy() - initial
        cash = float(adjusted.equity.iloc[-1])
        result.append(adjusted)
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame(columns=['timestamp','equity'])

def entry_available(f, selection, legs, day):
    ids, qty = [], []
    for kind,k,slot,q in legs:
        key = f'{kind}:{k}:{slot}'
        if key not in selection['mapping']:
            return False, 'missing required definition '+key
        ids.append(selection['mapping'][key]);qty.append(q)
    uid = selection['mapping']['F:0:0']
    if uid not in ids: ids.append(uid);qty.append(1)
    _, valid = old.quotes_at(f, ids, qty, pd.DatetimeIndex([day+pd.Timedelta(hours=14)]))
    return bool(valid[0]), 'missing sized entry quote' if not valid[0] else None

def unresolved_entry(f, selection, legs, day, stress):
    """Preserve known entry exposure when later valuation fails. No invented exit."""
    ids=[selection['mapping'][f'{kind}:{k}:{slot}'] for kind,k,slot,q in legs]
    qty=[q for kind,k,slot,q in legs]
    at=day+pd.Timedelta(hours=14)
    quotes,valid=old.quotes_at(f,ids,qty,pd.DatetimeIndex([at]))
    if not valid[0]:return {'entry_executable':False,'entry_fills':[],'reason':'entry unavailable'}
    fills=[]
    for iid,q in zip(ids,qty):
        z=quotes[iid];price=float(z.ask_px_00.iloc[0]+.25*stress if q>0 else z.bid_px_00.iloc[0]-.25*stress)
        if price<0:return {'entry_executable':False,'entry_fills':[],'reason':'negative modeled entry price'}
        fills.append({'at':str(at),'instrument_id':iid,'quantity':q,'price':price,'fee':2.5*abs(q)})
    return {'entry_executable':True,'entry_fills':fills,'unresolved_positions':{str(i):q for i,q in zip(ids,qty)},
        'terminal_equity':None,'reason':'Entry quotes known; subsequent required valuation/execution failed. Pending exposure is separate from the last fully settled subaccount; do not treat settled cash as total wealth.'}

def score():
    register()
    summaries, events = [], []
    datasets = {}
    for day in DAYS:
        folder = OUT/'options'/day.strftime('%Y%m%d')
        try:
            sel = json.loads((folder/'selection.json').read_text())
            parts = []
            for d in pd.date_range(day,periods=3):
                p = folder/f'quotes_{d.strftime("%Y%m%d")}.parquet'
                a = json.loads(p.with_suffix('.audit.json').read_text())
                if old.sha(p) != a['extract_sha256']: raise ValueError('Quote cache changed')
                parts.append(pd.read_parquet(p))
            datasets[day] = (sel, pd.concat(parts,ignore_index=True), None)
        except (FileNotFoundError, ValueError) as exc:
            # Cannot infer whether we entered when the whole episode is unavailable.
            datasets[day] = (None,None,str(exc))
    for section, legs in list(old.registry().items())+[('ScheduledES',[('F',0,0,1)])]:
        for stress in [0,1]:
            case = 'extra_tick' if stress else 'base'
            curves, fills = [], []
            entered = skipped = gaps = 0
            halted = None
            settled_cash = 1e6
            for day in DAYS:
                key = dict(section=section,case=case,week=str(day.date()))
                if settled_cash<=0:
                    halted=dict(key,reason='Insolvent settled account; no fresh capital or further entries')
                    events.append(dict(halted,status='accounting_halt'));break
                sel, f, err = datasets[day]
                if err:
                    halted = dict(key,reason='Unresolved data availability: '+err)
                    events.append(dict(halted,status='accounting_halt'));break
                available, reason = entry_available(f,sel,legs,day)
                if not available:
                    events.append(dict(key,status='cash_no_entry',reason=reason));skipped+=1
                    curves.append(pd.DataFrame({'timestamp':[day+pd.Timedelta(hours=14),day+pd.Timedelta(days=2,hours=19,minutes=55)],'equity':[1e6,1e6]}))
                    continue
                try:
                    r,h,ff = old.run_one(f,sel,legs,day,stress)
                except ValueError as exc:
                    halted = dict(key,reason=str(exc),unresolved=unresolved_entry(f,sel,legs,day,stress))
                    events.append(dict(halted,status='accounting_halt'));break
                entered+=1;gaps+=r['missing_marks'];curves.append(h[['timestamp','equity']]);fills.extend(ff)
                events.append(dict(key,status='closed',**r))
                observed=settled_cash+h.equity.to_numpy()-1e6
                settled_cash=float(observed[-1])
                if np.min(observed)<=0:
                    halted=dict(key,reason='Observed insolvency; subsequent margin/liquidation model unavailable. Retain losses and stop; no recapitalization.')
                    events.append(dict(halted,status='accounting_halt'));break
            h = combine(curves)
            name = section.replace('.','_')+'_'+case
            h.to_parquet(OUT/(name+'_equity.parquet'),index=False)
            old.save(OUT/(name+'_fills.json'),fills)
            cashflow = sum(-r['quantity']*r['price']*50-r['fee'] for r in fills)
            actual = float(h.equity.iloc[-1]-1e6) if len(h) else 0.
            if not np.isclose(actual,cashflow,atol=1e-6,rtol=0): raise ValueError('Continuous ledger failed reconciliation')
            eq = np.r_[1e6,h.equity.to_numpy()]
            summaries.append(dict(section=section,case=case,complete=halted is None,halted=halted,
                entered_weeks=entered,cash_weeks=skipped,unobserved_marks=gaps,
                full_year_return=float(eq[-1]/1e6-1) if halted is None else None,
                last_fully_settled_equity=float(eq[-1]),settled_subaccount_return=float(eq[-1]/1e6-1),
                max_observed_drawdown=float(np.max(1-eq/np.maximum.accumulate(eq))),
                fees=sum(x['fee'] for x in fills),reconciliation_error=abs(actual-cashflow),qualified=False))
    old.save(OUT/'results.json',summaries);old.save(OUT/'events.json',events)
    old.save(OUT/'manifest.json',{'at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':old.sha(OUT/'protocol.json'),'code_sha256':old.sha(__file__),
        'planned_weeks':len(DAYS),'candidate_series':116,'benchmark_series':2,
        'completed_candidate_series':sum(r['complete'] for r in summaries if r['section']!='ScheduledES'),
        'halted_candidate_series':sum(not r['complete'] for r in summaries if r['section']!='ScheduledES'),
        'qualified':0,'source_audits':{str(p.relative_to(OUT)):old.sha(p) for p in sorted((OUT/'options').glob('**/*.audit.json'))}})
    print(json.dumps({'series':len(summaries),'completed':sum(r['complete'] for r in summaries)}),flush=True)

if __name__ == '__main__':
    {'register':register,'stage':stage,'score':score}[sys.argv[1]]()
