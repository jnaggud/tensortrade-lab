"""Independent fill/position/cash replay plus source and training audits."""
from pathlib import Path
from datetime import datetime,timezone
import json,re
import numpy as np
import pandas as pd
from full_catalogue_options import save,sha
from full_catalogue_validate import FROZEN
from futures_intraday_retest import SPECS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'ensemble'

def share_replay(path,frames):
    h=pd.read_parquet(path);fills=json.loads(path.with_name(path.stem+'_fills.json').read_text())
    symbols=[c[6:] for c in h.columns if c.startswith('units_')];n=len(h)
    prices={s:frames[s].set_index('bar_end').reindex(pd.DatetimeIndex(h.timestamp)) for s in symbols}
    if any(f.close.isna().any() for f in prices.values()):raise AssertionError('Missing mark source '+str(path))
    close=np.column_stack([prices[s].close for s in symbols]);opening=np.column_stack([prices[s].open for s in symbols]);div=np.column_stack([prices[s].dividend if 'dividend' in prices[s] else np.zeros(n) for s in symbols])
    ends={v:i for i,v in enumerate(pd.DatetimeIndex(h.timestamp))};opens={v:i for i,v in enumerate(pd.DatetimeIndex(prices[symbols[0]].timestamp))}
    units=np.zeros((n,len(symbols)));cashflows=np.zeros(n)
    for x in fills:
        at=pd.Timestamp(x['timestamp']);closing=x['reason'] in ['terminal','terminal_liquidation','dividend_reinvestment']
        j=(ends if closing else opens)[at];k=symbols.index(x['symbol']);ref=(close if closing else opening)[j,k]
        if not np.isclose(ref,x['reference_price'],atol=1e-8,rtol=0):raise AssertionError('Wrong execution reference '+str(path))
        units[j,k]+=x['units'];cashflows[j]-=x['units']*x['price']+x['fee']
    units=units.cumsum(axis=0);actual=h[['units_'+s for s in symbols]].to_numpy()
    np.testing.assert_allclose(units,actual,atol=1e-8,rtol=0)
    earned=h.interest.to_numpy();signed='borrow' in h
    distributions=(np.vstack([np.zeros(len(symbols)),units[:-1]])*div).sum(axis=1).cumsum()
    recorded=(h.long_dividends-h.short_dividends).to_numpy() if signed else h.dividends.to_numpy()
    np.testing.assert_allclose(distributions,recorded,atol=1e-7,rtol=0)
    expected=10000+cashflows.cumsum()+earned+distributions-(h.borrow.to_numpy() if signed else 0)
    err=max(float(np.max(np.abs(expected-h.cash.to_numpy()))),float(np.max(np.abs(expected+(units*close).sum(axis=1)-h.equity.to_numpy()))))
    if err>1e-6:raise AssertionError('Share account replay failed '+str(path)+' '+str(err))
    if np.abs(units[-1]).max()>1e-8:raise AssertionError('Unclosed share account')
    # Independently calculate yield/borrow increments from prior balances.
    days=pd.DatetimeIndex(prices[symbols[0]].timestamp).normalize().to_series().diff().dt.days.fillna(0).to_numpy()
    rate=np.zeros(n)
    if not signed and 'BTC' not in symbols and 'no_interest' not in path.name:
        raw=pd.read_parquet(ROOT/'data/rotation/cash_yield.parquet');raw['day']=pd.to_datetime(raw.date,utc=True)
        joined=pd.merge_asof(pd.DataFrame({'day':pd.DatetimeIndex(prices[symbols[0]].timestamp).normalize()}),raw,on='day',allow_exact_matches=False)
        rate=(joined.discount_yield/(1-joined.discount_yield*91/360)*365/360).fillna(0).to_numpy()
    previous=np.r_[10000,expected[:-1]];interest=previous*rate*days/365
    np.testing.assert_allclose(interest.cumsum(),earned,atol=1e-7,rtol=0)
    if signed:
        short=-(np.minimum(units,0)*close).sum(axis=1);borrow=np.r_[0,short[:-1]]*(.10 if 'double_cost' in path.name else .05)*days/365
        np.testing.assert_allclose(borrow.cumsum(),h.borrow.to_numpy(),atol=1e-7,rtol=0)
    return {'ledger':str(path.relative_to(HERE)),'rows':n,'fills':len(fills),'max_error_dollars':err}

def future_replay(path,feed,root):
    h=pd.read_parquet(path);f=feed.set_index('bar_end').reindex(pd.DatetimeIndex(h.timestamp));fills=json.loads(path.with_name(path.stem+'_fills.json').read_text());mult,tick=SPECS[root]
    ix={v:i for i,v in enumerate(pd.DatetimeIndex(f.timestamp))};principal=np.zeros(len(h));changes={}
    for x in fills:
        j=ix[pd.Timestamp(x['timestamp'])];row=f.iloc[j]
        ref=row.close if x['reason']=='terminal_close' else row.old_open if x['reason']=='roll_out' else row.open
        if abs(ref-x['reference_price'])>1e-8:raise AssertionError('Wrong raw future price')
        if int(x['units'])!=x['units']:raise AssertionError('Fractional future')
        principal[j]-=x['units']*x['modeled_price']*mult+x['fee']
        changes.setdefault(j,[]).append(x)
    principal=1e6+principal.cumsum();positions={};equity=[]
    for j,row in enumerate(f.itertuples()):
        for x in changes.get(j,[]):positions[x['symbol']]=positions.get(x['symbol'],0)+x['units']
        if any(q and sym!=row.symbol for sym,q in positions.items()):raise AssertionError('Held expired/old contract')
        equity.append(principal[j]+positions.get(row.symbol,0)*row.close*mult)
    if any(positions.values()):raise AssertionError('Unclosed future')
    err=float(np.max(np.abs(np.asarray(equity)-h.equity.to_numpy())))
    if err>1e-6:raise AssertionError('Future replay failed '+str(path)+' '+str(err))
    return {'ledger':str(path.relative_to(HERE)),'rows':len(h),'fills':len(fills),'max_error_dollars':err}

def main():
    checks=[];hashes=[];models=[];training=[]
    for path in (HERE/'data_expansion').rglob('*.audit.json'):
        a=json.loads(path.read_text());source=path.with_name(path.name.replace('.audit.json',''))
        if 'vintages.audit' in path.name:source=path.with_name(path.name.replace('.audit.json','.zip'))
        assert sha(source)==a['sha256'],'Changed public download '+str(source)
        hashes.append(str(source))
    for market in ['US_ETF_portfolio','BTC_daily']:
        folder=OUT/market;sources=json.loads((folder/'sources.json').read_text());frames={}
        for s in sources:
            if sha(s['path'])!=s['sha256']:raise AssertionError('Ensemble source changed')
            hashes.append(s['path'])
            if s['kind']=='prices':frames[s['symbol']]=pd.read_parquet(s['path'])
        for path in (folder/'ledgers').glob('*.parquet'):checks.append(share_replay(path,frames))
        for path in folder.glob('20*/ppo_*.json'):
            m=json.loads(path.read_text());assert sha(path.with_suffix('.zip'))==m['model_sha256'];assert m['actual_timesteps']==16384;assert m['identity']['source_identity']==sha(folder/'sources.json');models.append(str(path))
        for path in folder.glob('20*/training_audit.json'):
            a=json.loads(path.read_text());assert a['ridge']['max_label_index']<a['ridge']['cut'];assert pd.Timestamp(a['trained_strictly_before'])<pd.Timestamp(a['first_decision'])
            assert all(pd.Timestamp(x['train_end'])<pd.Timestamp(a['first_decision']) for x in a['ppo']);training.append(str(path))
    for root in ['ES','GC','CL','BTC']:
        folder=OUT/'intraday'/root;sources=json.loads((folder/'sources.json').read_text());feed=HERE/'next_batch'/('BTC_BITSTAMP_15m.parquet' if root=='BTC' else root+'_execution_15m.parquet')
        assert sha(feed)==sources['feed_sha256'];f=pd.read_parquet(feed);hashes.append(str(feed))
        for s in sources['shadow_books']:
            for suffix,key in [('_equity.parquet','curve_sha256'),('_fills.json','fills_sha256')]:
                path=HERE/'next_batch'/(s['name']+suffix);assert sha(path)==s[key];hashes.append(str(path))
        for path in folder.glob('*.parquet'):
            checks.append(share_replay(path,{'BTC':f}) if root=='BTC' else future_replay(path,f,root))
    for root in ['ES','CL']:
        folder=OUT/'all_clocks'/root;s=json.loads((folder/'sources.json').read_text())
        feed=HERE/'next_batch'/f'{root}_execution_15m.parquet';daily=HERE/'next_batch'/f'{root}_execution_daily.parquet'
        assert sha(feed)==s['intraday_feed_sha256'];assert sha(daily)==s['daily_feed_sha256'];hashes.extend([str(feed),str(daily)])
        for row in s['shadow_books']:
            for suffix,key in [('_equity.parquet','curve_sha256'),('_fills.json','fills_sha256')]:
                path=HERE/'next_batch'/(row['name']+suffix);assert sha(path)==row[key];hashes.append(str(path))
        f=pd.read_parquet(feed)
        for path in folder.glob('*.parquet'):checks.append(future_replay(path,f,root))
    frames={'AAPL':pd.read_parquet(ROOT/'data/yahoo/AAPL_1d.parquet'),'SPY':pd.read_parquet(HERE/'full_catalogue/data/SPY.parquet'),'DBC':pd.read_parquet(HERE/'data_expansion/DBC.parquet')}
    for path in [ROOT/'data/yahoo/AAPL_1d.parquet',HERE/'full_catalogue/data/SPY.parquet',HERE/'data_expansion/DBC.parquet']:
        assert sha(path)==json.loads(path.with_suffix('.metadata.json').read_text())['sha256'];hashes.append(str(path))
    for prefix in ['EarningsSUE','BookToMarket','InflationAllocation','InflationBuyHold']:
        for path in (HERE/'data_expansion').glob(prefix+'_*.parquet'):checks.append(share_replay(path,frames))
    optionchecks=[];extracts=[];ow=HERE/'options_weekly'
    assert (ow/'manifest.json').exists(),'52-week scoring must finish before final validation'
    unresolved_entries=0;events=json.loads((ow/'events.json').read_text())
    for r in json.loads((ow/'results.json').read_text()):
        if r['complete']:
            assert r['entered_weeks']+r['cash_weeks']==52;assert r['full_year_return'] is not None
            continue
        assert r['full_year_return'] is None
        halt=r['halted'];subset=[e for e in events if e['section']==r['section'] and e['case']==r['case']]
        assert all(e['week']<=halt['week'] for e in subset),'Resumed after unresolved account'
        pending=halt.get('unresolved',{})
        if not pending.get('entry_executable'):continue
        assert pending['terminal_equity'] is None
        day=pd.Timestamp(halt['week'],tz='UTC');quotes=pd.read_parquet(ow/'options'/day.strftime('%Y%m%d')/('quotes_'+day.strftime('%Y%m%d')+'.parquet'))
        for x in pending['entry_fills']:
            at=pd.Timestamp(x['at']);z=quotes[(quotes.instrument_id==x['instrument_id'])&(quotes.ts_recv<=at)&(quotes.ts_recv>=at-pd.Timedelta(minutes=1))].sort_values('ts_recv')
            assert len(z);q=x['quantity'];last=z.iloc[-1];tick=.25 if r['case']=='extra_tick' else 0.
            expected=float(last.ask_px_00+tick if q>0 else last.bid_px_00-tick)
            assert min(last.bid_sz_00,last.ask_sz_00)>=abs(q)
            assert expected>=0 and abs(expected-x['price'])<1e-8
            assert pending['unresolved_positions'][str(x['instrument_id'])]==q
            unresolved_entries+=1
    for path in ow.glob('*_equity.parquet'):
        h=pd.read_parquet(path);fills=json.loads(path.with_name(path.name.replace('_equity.parquet','_fills.json')).read_text());net=sum(-x['quantity']*x['price']*50-x['fee'] for x in fills)
        err=abs(1e6+net-float(h.equity.iloc[-1])) if len(h) else abs(net)
        if err>1e-6:raise AssertionError('Weekly option cashflow mismatch')
        positions={}
        for x in fills:positions[x['instrument_id']]=positions.get(x['instrument_id'],0)+x['quantity']
        assert not any(positions.values());optionchecks.append({'ledger':str(path.relative_to(HERE)),'max_error_dollars':err})
    for path in (ow/'options').glob('**/*audit.json'):
        a=json.loads(path.read_text());extract=path.with_name('definitions.parquet') if path.name=='definitions_audit.json' else path.with_name(path.name.replace('.audit.json','.parquet'))
        assert sha(extract)==a['extract_sha256']
        if a.get('expected_sha256'):assert a['expected_sha256']==a['sha256']
        extracts.append(str(path.relative_to(HERE)))
    for manifest,code,protocol in [
        (OUT/'US_ETF_portfolio/manifest.json',HERE/'ensemble_study.py',OUT/'protocol.json'),
        (OUT/'BTC_daily/manifest.json',HERE/'ensemble_study.py',OUT/'protocol.json'),
        *[(OUT/f'intraday/{s}/manifest.json',HERE/'ensemble_intraday.py',OUT/'intraday/protocol.json') for s in ['ES','GC','CL']],
        (OUT/'intraday/BTC/manifest.json',HERE/'ensemble_btc_intraday.py',OUT/'intraday/BTC/protocol.json'),
        *[(OUT/f'all_clocks/{s}/manifest.json',HERE/'ensemble_all_clocks.py',OUT/'all_clocks/protocol.json') for s in ['ES','CL']],
        (HERE/'data_expansion/filing_manifest.json',HERE/'filing_study.py',HERE/'data_expansion/filing_protocol.json'),
        (HERE/'data_expansion/macro_manifest.json',HERE/'macro_study.py',HERE/'data_expansion/macro_protocol.json'),
        (ow/'manifest.json',HERE/'options_weekly.py',ow/'protocol.json')]:
        m=json.loads(manifest.read_text());assert sha(code)==m['code_sha256'];assert sha(protocol)==m['protocol_sha256']
    frozen={str(p):sha(p) for p in FROZEN};assert all(frozen[str(p)]==v for p,v in FROZEN.items())
    import prospective
    events=prospective.load_log()
    log=(OUT/'test_suite.log').read_text();match=re.search(r'(\d+) passed',log);assert match and not re.search(r'\d+ failed',log)
    save(OUT/'ledger_reconciliation.json',{'actual_price_accounts':checks,'weekly_options':optionchecks})
    save(OUT/'validation.json',{'at':datetime.now(timezone.utc).isoformat(),'tests_passed':int(match[1]),'actual_price_accounts_replayed':len(checks),'account_rows_replayed':sum(r['rows'] for r in checks),'weekly_option_accounts_reconciled':len(optionchecks),'unresolved_option_entry_legs_verified':unresolved_entries,'source_hash_checks':len(hashes),'weekly_extract_hash_checks':len(extracts),'ppo_models_verified':len(models),'training_folds_verified':len(training),'frozen_hashes':frozen,'frozen_chain_valid':True,'frozen_events':len(events),'max_accounting_error_dollars':max(x['max_error_dollars'] for x in checks+optionchecks),'scope':'Replay verifies every saved share/futures account mark from signed fills and raw prices, including dividend/interest/borrow increments. Weekly option comparison is terminal signed cashflow of settled portions, plus original-quote checks for unresolved entry legs. Missing intraperiod marks remain unverified. No guarantee of historical fill availability. No live/prospective mutation.'})
    print('Expansion verified',len(checks),'price accounts',len(optionchecks),'weekly option accounts',flush=True)

if __name__=='__main__':main()
