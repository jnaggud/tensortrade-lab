"""Read-only verification of research ledgers, extracts and frozen forward state."""
from pathlib import Path
import json,re
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from full_catalogue_options import HERE,OUT,sha,save
from futures_extension import SPECS

FROZEN={HERE/'prospective.py':'e55d65a68c24267524cad72054c11ba03443d162d1fce3ef19ca236c2342a128',
 HERE.parents[1]/'src/tensortrade_lab/portfolio.py':'73d377fdeb3e32087f70ee8f760450e7ca764d14d1dc0fdd5dbf763445afaef0',
 HERE/'prospective/freeze.json':'82237cdc26e5442be6b58c84ccfbacee1c61679abd8dae8af99fb4a59a348af3'}

def main():
    prices=[];optionchecks=[];futurechecks=[]
    for path in sorted((OUT/'price_ledgers').glob('*.parquet')):
        h=pd.read_parquet(path);last=h.iloc[-1];fills=json.loads(path.with_name(path.stem+'_fills.json').read_text())
        cash=10000-sum(x['units']*x['price']+x['fee'] for x in fills)
        cash+=float(last['interest'])
        cash+=float(last['dividends']) if 'dividends' in last else float(last['long_dividends']-last['short_dividends']-last['borrow'])
        diff=float(cash-last.equity)
        if abs(diff)>1e-6:raise AssertionError('Cash flow mismatch '+path.stem+' '+str(diff))
        if any(abs(last[c])>1e-8 for c in h.columns if c.startswith('units_')):raise AssertionError('Unclosed holdings '+path.stem)
        prices.append({'ledger':str(path.relative_to(OUT)),'difference_dollars':diff,'fills':len(fills)})
    for path in sorted((OUT/'options').glob('*/*_equity.parquet')):
        fills=json.loads(path.with_name(path.name.replace('_equity.parquet','_fills.json')).read_text());h=pd.read_parquet(path)
        net=sum(-x.get('quantity',x.get('units'))*x['price']*50-x['fee'] for x in fills)
        difference=float(1e6+net-h.equity.iloc[-1])
        if abs(difference)>1e-6:raise AssertionError('Option/futures signed fills mismatch '+str(path))
        by={}
        for x in fills:by[x['instrument_id']]=by.get(x['instrument_id'],0)+x.get('quantity',x.get('units'))
        if any(abs(v)>1e-8 for v in by.values()):raise AssertionError('Nonflat terminal derivative position')
        optionchecks.append({'ledger':str(path.relative_to(OUT)),'difference_dollars':difference,'fills':len(fills)})
    for name in ['activity','value','calendar']:
        path=OUT/(name+'_equity.parquet')
        if not path.exists():continue
        h=pd.read_parquet(path);fills=json.loads((OUT/(name+'_fills.json')).read_text());cash=1e6
        positions={}
        for x in fills:
            root=next(r for r in SPECS if x['symbol'].startswith(r));reference=x.get('reference_open',x.get('reference_close'))
            cash-=x['units']*reference*SPECS[root][0]+x['fee']+x['slippage']
            positions[x['symbol']]=positions.get(x['symbol'],0)+x['units']
        if any(v for v in positions.values()):raise AssertionError('Nonflat future '+name)
        difference=float(cash-h.equity.iloc[-1])
        if abs(difference)>1e-6:raise AssertionError('Futures fill cashflow mismatch '+name+' '+str(difference))
        futurechecks.append({'ledger':name,'difference_dollars':difference,'fills':len(fills)})
    audits=[]
    for path in sorted((OUT/'options').glob('*/*audit.json'))+sorted((OUT/'open_interest').glob('*.audit.json')):
        audit=json.loads(path.read_text())
        extract=path.with_name('definitions.parquet') if path.name=='definitions_audit.json' else path.with_name(path.name.replace('.audit.json','.parquet'))
        if sha(extract)!=audit['extract_sha256']:raise AssertionError('Changed extract '+str(extract))
        if audit.get('expected_sha256') and audit['sha256']!=audit['expected_sha256']:raise AssertionError('Mismatched original source hash')
        audits.append({'audit':str(path.relative_to(OUT)),'extract_sha256':audit['extract_sha256']})
    frozen={str(p):sha(p) for p in FROZEN}
    if any(frozen[str(p)]!=expected for p,expected in FROZEN.items()):raise AssertionError('Frozen prospective source changed')
    import prospective
    log=prospective.load_log()
    testlog=(OUT/'test_suite.log').read_text()
    matched=re.search(r'(\d+) passed',testlog)
    if matched is None or re.search(r'\d+ failed',testlog):raise AssertionError('Full suite has not passed')
    # Registry-specific tests re-run after availability amendment and extra modules.
    focused=OUT/'focused_tests.log'
    if not focused.exists() or re.search(r'\d+ failed',focused.read_text()) or 'passed' not in focused.read_text():raise AssertionError('Post-change focused tests missing')
    save(OUT/'validation.json',{'at':datetime.now(timezone.utc).isoformat(),'full_suite_passed':int(matched[1]),
      'full_suite_warning':'One existing stable-baselines3 Box action-space recommendation.',
      'post_change_focused_test_log':str(focused),'price_ledgers_reconciled':len(prices),'option_and_volatility_ledgers_reconciled':len(optionchecks),
      'futures_ledgers_reconciled':len(futurechecks),'extract_hashes_verified':len(audits),'frozen_sources':frozen,
      'frozen_event_chain_valid':True,'frozen_event_count':len(log),'maximum_fill_reconciliation_error_dollars':max([abs(x['difference_dollars']) for x in prices+optionchecks+futurechecks],default=0.),
      'scope':'Extract hashes and saved source verification records checked; raw source streams were hashed during staging, not needlessly reread here. Cash-flow reconciliation does not validate actual exchange fills.'})
    save(OUT/'ledger_reconciliation.json',{'prices':prices,'options_and_volatility':optionchecks,'futures':futurechecks})
    save(OUT/'extract_validation.json',audits)
    print('Verified',len(prices),'price,',len(optionchecks),'option/volatility,',len(futurechecks),'future ledgers and',len(audits),'extracts; frozen chain valid',flush=True)
if __name__=='__main__':main()
