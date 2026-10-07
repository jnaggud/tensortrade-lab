from pathlib import Path
import json,hashlib
import pandas as pd
from collections import Counter
HERE=Path(__file__).parent
recon=json.loads((HERE/'tradingview/reconciliation/summary.json').read_text())
manifest=json.loads((HERE/'tradingview/promoted/manifest.json').read_text())
pct=lambda x:f'{100*x:,.2f}%'
parts=['# TradingView reconciliation — TLT daily\n\n',
'**All four native Pine strategies reproduce their TradingView trade events, integer quantities and cent-rounded fill prices on the same TradingView price history.** The comparisons cover 3 January 2017–23 September 2026. The saved reports and 6,083 chart bars were exported through the working local TradingView connection. This verifies these scripts on TLT with these settings, not every instrument, interval or data source.\n\n',
'## Matched-period results\n\n',
'Price-only accounts: USD10,000, 0.10% each-side commission, no cash interest or dividend credit, no tick slippage. These diagnostic figures do not replace the economic study.\n\n',
'| Pine rule | Execution events | Price-only return | Daily equity drawdown | Immediate margin exits |\n|---|---:|---:|---:|---:|\n']
for rule,r in recon.items():
 m=r['native_fill_replay'];parts.append(f"| {rule} | {r['comparison']['tradingview']['native_fill_count']} | {pct(m['return'])} | {pct(m['daily_peak_drawdown'])} | {r['native_margin_exits_in_matched_sample']} |\n")
parts+=['\nAn execution event is an aggregated entry, ordinary exit or margin exit. Partial trade rows share one original entry; counting every report row as a separate new order would double-count its quantity.\n\n',
'## Why reports originally disagreed\n\n',
'1. **Cash flows:** the local economic model includes cash distributions and idle-cash interest. Native Pine does not credit those cash flows. For example, the main TLT MA50/200 study returned +30.31%, while the matched price-only emulator returned −0.50%. That gap cannot be called a coding error without first aligning accounting conventions.\n',
'2. **Order sizing and rejection:** these Pine scripts size integer orders from the preceding close, reserve commission, and request 99.8% exposure. A price gap can make the order unaffordable at the next open, so the emulator rejects it. AlphaCombo had four rejected opening attempts. A fee can also create a tiny cash shortfall after an otherwise affordable fill; its 100% margin setting then produces a small forced sale. AlphaCombo had two and IBS twelve such exits. This behavior is reproduced here, not added to the original local strategy rules.\n',
'3. **Quote precision and source:** TLT executions round to a $0.01 tick. Some chart bars contain half-cent values. Yahoo and TradingView reproduce the same events for the two moving-average rules and AlphaCombo after emulator accounting is aligned. IBS differs because its threshold reacts to small OHLC differences. On TradingView prices, all 600 IBS execution events match. Yahoo gives a −29.01% price-only return versus −29.72% using TradingView prices; its 597 events do not match. IBS remains sensitive to data source.\n',
'4. **Risk definition:** native maximum intrabar drawdown uses closed-trade equity peaks. Our comparison measures each day’s marked account value against its earlier daily peak. For MA50/200, those are 17.72% and 25.60%, respectively. We reproduce the native drawdown exactly for both moving-average rules and AlphaCombo once entry and exit fees are included. See [TradingView’s documented formula](https://www.tradingview.com/support/solutions/43000681690-max-drawdown-intrabar/).\n',
'5. **End dates:** IBS opened another trade on 24 September, after the local cutoff. Its current native report includes that open trade and shows 35.18% intrabar drawdown. The matched sample ends on 23 September, flat. Those numbers are not evidence of a same-period discrepancy. Open-position marks are excluded from execution events.\n\n',
'Five regression checks passed: the four independent same-feed strategy reconstructions plus partial-trade aggregation/open-mark handling. There are 702 matched execution events across the four strategies. Price-only final cash differs by less than $0.000001 between independent reconstruction and native-fill replay.\n\n',
'## Scope of the ten ports\n\n',
'All ten private scripts were corrected for dividend/share-basis consistency, recompiled with zero errors/warnings, and read back to verify the updated sources. Original versions are retained in pre_distribution_fix/. Four are native single-chart strategies; six display multiasset allocation targets. Native Strategy Tester does not provide the local multiasset account ledger for those six. After the correction, all six portfolio indicators match all 117 monthly target decisions and all 2,442 compared daily target vectors each: 702 monthly decisions and 14,652 daily vectors in total. These are target-weight checks, not a native multiasset account backtest. See [the correction report](DISTRIBUTION_CORRECTION.md).\n\n',
'The private research layout (not distributed) has MA50/200 on daily TLT and the sector target table. Temporary audit studies were removed from the layout; all saved scripts remain available in My Scripts. Distribution accounting and a display-only compiler warning were corrected. No signal threshold was tuned to returns, no live order was sent and no script was publicly published.\n\n',
'## Reproduce\n\n```sh\n.venv/bin/python research/paper151/reconcile_tradingview.py\n.venv/bin/pytest -q tests/test_tradingview_reconciliation_paper151.py\n```\n\n',
'`reconciliation/` stores raw reports, chart prices, audit details and reconstructed daily equity. The emulator-sizing/rejection implementation is an empirical match for these saved scripts and settings; it is not a general replacement for TradingView’s broker emulator. The original economic ranking is unchanged.\n']
(HERE/'tradingview/RECONCILIATION.md').write_text(''.join(parts))
for row in manifest:
 rule=row['strategy']
 if rule in recon:
  row['runtime_reconciliation']={'market':'BATS:TLT','interval':'1D','from':'2017-01-03','through':'2026-09-23','same_feed_events_units_prices_match':True,'yahoo_events_units_match':recon[rule]['comparison']['yahoo']['quantities_match'],'report':'../RECONCILIATION.md'}
 if row['shortlist_rank']<=6:
  row['runtime_reconciliation']={'kind':'portfolio_targets','from':'2017-01-03','through':'2026-09-21','monthly_targets_matched':117,'daily_targets_matched':2442,'portfolio_pnl_in_native_tester':False,'report':'../RECONCILIATION.md'}
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'],'Saved Pine source changed'
(HERE/'tradingview/promoted/manifest.json').write_text(json.dumps(manifest,indent=2))
cat=json.loads((HERE/'catalogue.json').read_text())
combined={'asof':'2026-09-29','recipes':34,'runs_including_benchmarks_and_stresses':925,'catalogue_rows':175,'catalogue_status':dict(Counter(x['status'] for x in cat)),
 'broadly_qualified_families':0,'private_pine_scripts':10,'native_pine_strategies':4,'multiasset_target_indicators':6,
 'test_checks_passed':75,'derivative_unit_checks':19,'derivative_historical_backtests':0,
 'shortlist_decision':'Retain the existing ten as exploratory visualization candidates. No new candidate passed its declared broad qualification gate. No final whole-catalogue winner claim.',
 'blocked_dependency':'Expansion archive directory reads time out; usable contract quote history not verified.',
 'remaining_work':['Historical options/futures ingestion, execution/lifecycle coverage and fixed strategy recipes','Point-in-time stock fundamentals, borrow and event histories','Specialized non-OHLC cash-flow studies','Independent prospective validation'],
 'portfolio_monthly_targets_matched':702,'portfolio_daily_targets_matched':14652,'sources':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'factor_results/manifest.json',HERE/'derivatives.py',HERE/'derivative_cache_schema.json',HERE/'reconcile_tradingview.py',HERE/'tradingview/reconciliation/TLT_native_reports.json',HERE/'tradingview/reconciliation/TLT_tradingview_prices.json',HERE/'tradingview/reconciliation/portfolio_summary.json',HERE/'tradingview/promoted/distribution_fix_checks.json']}}
(HERE/'current_manifest.json').write_text(json.dumps(combined,indent=2))
print(json.dumps({k:v for k,v in combined.items() if k!='sources'},indent=2))
