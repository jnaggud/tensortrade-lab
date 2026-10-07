from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from expanded_study import load_jobs
from run_study import panel_from
from catalogue import build

HERE=Path(__file__).parent;OUT=HERE/'expanded_results'
rows=build()
r=pd.read_json(OUT/'metrics.json');ranking=pd.read_json(OUT/'ranking.json')
promoted=json.loads((HERE/'tradingview/promoted/manifest.json').read_text())
full=r[(r.window=='full')&(r['case']=='base')&~r.strategy.isin(['BuyHold','PassiveBasket'])]
block=r[(r.window!='full')&(r['case']=='base')&~r.strategy.isin(['BuyHold','PassiveBasket'])]
by_market=block.groupby(['market','strategy'])['pass'].agg(['sum','count'])
consistent=by_market[by_market['sum']==by_market['count']]
cash=[]
for market,frames,_,crypto in load_jobs()[0]:
    if market not in ('TLT','BTC','MultiAsset'):continue
    p=panel_from(frames,not crypto,365 if crypto else 252)
    ix=np.flatnonzero(p.timestamp>=pd.Timestamp('2017-01-01',tz='UTC'));value=10000.
    for t in ix:
        days=(p.timestamp[t].normalize()-p.timestamp[t-1].normalize()).days
        value*=1+p.cash_rate[t]*days/365
    cash.append({'market':market,'return':value/10000-1})
(OUT/'cash_reference.json').write_text(json.dumps(cash,indent=2))

def pct(v):return f'{100*v:,.2f}%'
lines=['# 151 Trading Strategies: local screen and provisional Pine shortlist\n\n',
       '**The complete paper was read. The complete catalogue has not yet been historically backtested.** ',
       f'All 175 catalogue entries are tracked; {sum(x["status"]=="historically_tested_adaptation" for x in rows)} currently have tested adaptations. ',
       'The remaining entries have specific data, accounting, non-trading or exclusion dispositions. No untested entry was assigned a performance score.\n\n',
       f'This expanded run evaluated **{full.strategy.nunique()} rule families across {len(full)} market/rule combinations**, producing {len(r)} runs including matched benchmarks, chronological blocks and stress cases. ',
       'These are not independent statistical trials. No hyperparameters were optimized after viewing the results. This is retrospective screening on previously inspected history, not a fresh holdout or a live track record.\n\n',
       '## What the results say\n\n',
       f'{int(full["pass"].sum())} market/rule combinations met the user\'s return-and-drawdown requirement over their full sample. ',
       'None of the 24 families met the stronger broad-robustness screen across every tested market/block and doubled-cost full-period case. ',
       'One asset-specific combination, TLT/Vol12, passed all three chronological base-cost blocks. Its full-period gain was small and below the cash-yield proxy. ',
       'Passing against a losing buy-and-hold benchmark can mean losing less; it does not necessarily mean an attractive investment.\n\n',
       '## Full-period leads (not selected by changing parameters)\n\n',
       '| Market | Fixed rule | Strategy return | Buy-and-hold return | Strategy max drawdown | Buy-and-hold max drawdown |\n|---|---|---:|---:|---:|---:|\n']
for market,name in [('MultiAsset','GoldDiversification'),('TLT','MA50_200'),('TLT','MA20_50_200'),('TLT','Vol12'),('BTC','Momentum252')]:
    x=full[(full.market==market)&(full.strategy==name)].iloc[0]
    lines.append(f'| {market} | {name} | {pct(x["return"])} | {pct(x.return_bh)} | {pct(x.max_drawdown)} | {pct(x.max_drawdown_bh)} |\n')
lines+=['\nReturns are cumulative, after modeled fees and slippage. Drawdowns are close-to-close peak-to-trough losses. ',
        'US/crypto single-instrument samples run from the first eligible January 2017 bar through September 23, 2026; EFA and aligned portfolios stop September 21 because of a missing session. ',
        'Each row uses its own exactly matched benchmark dates. MultiAsset here means the 80% SPY/20% GLD rule, compared with SPY.\n\n',
        '**Cash reference, added for interpretation without changing the declared ranking:** ',
        ', '.join(f'{c["market"]}: {pct(c["return"])}' for c in cash)+'. This uses the same lagged Treasury-yield proxy for US idle cash; it is not a guaranteed broker sweep yield.\n\n',
        '## Ten promoted research families\n\n',
        'These are the ten highest-ranked Pine-compatible families under the predeclared cross-market ranking. **All ten are exploratory, not broadly qualified winners.** ',
        'Rank uses block pass fraction, median excess CAGR and drawdown deterioration after the broad qualification flag. ',
        'Single-instrument families are scored on seven markets (21 blocks); portfolio families on one declared universe (3 blocks). ',
        'Those different denominators and different benchmarks limit comparisons. Several closely related ETF recipes also share signals; the shortlist is not ten independent sources of return.\n\n',
        '| Rank | Rule | Passing market-period blocks | Median excess CAGR (percentage points/year) | Pine form |\n|---:|---|---:|---:|---|\n']
for m in promoted:
    rr=ranking[ranking.strategy==m['strategy']].iloc[0]
    hits=round(rr.block_pass_fraction*rr.scored_blocks)
    kind='Portfolio allocation indicator' if m['pine_kind']=='portfolio_signal_indicator' else 'Native strategy'
    lines.append(f'| {m["shortlist_rank"]} | [{m["strategy"]}]({m["path"]}) | {hits}/{int(rr.scored_blocks)} | {pct(rr.median_excess_cagr)} | {kind} |\n')
lines+=['\nMedian excess CAGR is the strategy\'s annualized growth minus the benchmark\'s annualized growth, in percentage points. ',
        'A negative number is underperformance. The full ranking, including failed recipes, KNN and ANN, is in `ranking.json`; all 749 run metrics are in `metrics.json`. ',
        'Asset-specific Bitcoin momentum and TLT volatility targeting remain useful research leads even though they did not make the broader family shortlist.\n\n',
        '## TradingView verification\n\n',
        'All ten scripts are saved privately. The research layout displays TLT with #07 MA50_200 and the #03 sector allocation indicator. The sector target snapshot matches the local September allocation: one third each in XLE, XLK and XLV. ',
        'The native TLT report returned approximately −0.50%; a local diagnostic excluding dividend cash and idle-cash interest returned −0.44%, versus +30.31% in the declared total-return study. This explains most of the return gap, but drawdown, share rounding, allocation, dates and feeds remain unreconciled. See [the Pine guide](../tradingview/promoted/README.md).\n\n',
        '## How to reproduce and inspect\n\n',
        '```sh\n.venv/bin/pytest -q tests/test_paper151.py tests/test_portfolio.py\n.venv/bin/python research/paper151/expanded_study.py\n.venv/bin/python research/paper151/promote.py\n.venv/bin/python research/paper151/make_report.py\n```\n\n',
        '27 accounting/causality tests passed. They include future-price mutation, mature machine-learning labels, exact fee reconciliation, dividend entitlement, no leverage and conservative portfolio weights. ',
        'All ten Pine sources compiled with zero errors and zero warnings and were saved privately in TradingView. Saved sources were read back and matched after line-ending normalization. Runtime verification is recorded in the promotion directory. ',
        'Source hashes, protocol hash, model fit dates and warnings are in `manifest.json`; full-sample equity, fills and decisions are in `ledgers/`.\n\n',
        '## Material limits and remaining work\n\n',
        '- The catalogue-wide historical test is incomplete. The Expansion Databento archive did not respond, and many strategies need data not verified in the working local sources. See `../CATALOGUE.md` and `../DATA_INVENTORY.md`.\n',
        '- Price-based recipes are explicit adaptations. Long-only ETF rankings do not reproduce the paper\'s stock long/short factors. Physical property, tax, project and control-investment activities require different analyses.\n',
        '- The old Bitcoin archive differs from the fresh Yahoo history on some dates. This run uses the new single-source snapshot. Yahoo data can be revised and aggregate crypto prices are not venue quotes.\n',
        '- Fixed ETF universes, common market exposures and prior research introduce selection concerns. Chronological diagnostics do not erase prior exposure to the data or correct all multiple-testing bias. No statistical claim of future outperformance is made.\n',
        '- The top six Pine files display portfolio allocation signals, not native multiasset Strategy Tester PnL. Four are native single-chart strategies. Costs, interest, dividends, start/end dates, allocation rounding and price feeds differ between local and TradingView accounting; reconciliation is outstanding.\n',
        '- Short intraday samples were tested in the earlier `../results/` study and failed the return screen. They are not eligible for this daily shortlist. Futures contract-roll execution, historical option bid/ask fills, market making, fundamentals and point-in-time sentiment need separate engines/data validation.\n',
        '- The next evidence step is a frozen prospective paper-trading evaluation plus independent source/execution reconciliation. Neither is claimed completed.\n']
(OUT/'REPORT.md').write_text(''.join(lines))

# Compact research visualization: illustrative leads plus all-pair risk/return.
fig=make_subplots(rows=3,cols=1,subplot_titles=['Stocks / gold versus SPY','Treasury rules versus TLT','Bitcoin momentum versus BTC'],vertical_spacing=0.08)
groups=[('MultiAsset',['BuyHold','GoldDiversification']),('TLT',['BuyHold','MA50_200','MA20_50_200','Vol12']),('BTC',['BuyHold','Momentum252'])]
for row,(market,names) in enumerate(groups,1):
    for name in names:
        h=pd.read_parquet(OUT/'ledgers'/f'{market}__{name}.parquet')
        fig.add_trace(go.Scatter(x=h.timestamp,y=h.equity/10000,name=market+' '+name),row=row,col=1)
    fig.update_yaxes(type='log',title_text='Growth of $1 (log)',row=row,col=1)
fig.update_layout(height=1100,template='plotly_white',title='Historical leads — retrospective, after modeled costs')
scatter=go.Figure()
for passed,color in [(False,'#b65d4e'),(True,'#16817a')]:
    g=full[full['pass']==passed]
    scatter.add_trace(go.Scatter(x=100*g.max_drawdown,y=100*g.cagr,mode='markers',name='Full-sample pass' if passed else 'Full-sample fail',marker=dict(color=color,size=9),text=g.market+' / '+g.strategy,customdata=np.column_stack([100*g.cagr_bh,100*g.max_drawdown_bh]),hovertemplate='%{text}<br>CAGR %{y:.2f}%<br>Drawdown %{x:.2f}%<br>Benchmark CAGR %{customdata[0]:.2f}%<br>Benchmark drawdown %{customdata[1]:.2f}%<extra></extra>'))
scatter.update_layout(template='plotly_white',height=570,title='All 96 combinations; full-sample pass is not broad qualification',xaxis_title='Maximum drawdown % (lower is better)',yaxis_title='Net CAGR % (higher is better)')
html='<!doctype html><html><head><meta charset="utf-8"><title>151 strategies — local research</title><style>body{font:17px system-ui;max-width:1200px;margin:35px auto;color:#183037;padding:0 25px}p{line-height:1.6}.notice{padding:20px;background:#fff5dd;border-left:5px solid #cc8f27}h1{font-size:32px}</style></head><body><h1>151 Trading Strategies: local research</h1><p class="notice">24 rule families · 96 market/rule combinations · 749 total runs. 11 full-sample passes. No family passed the broad cross-market requirement. TLT/Vol12 passed all three individual periods but lagged the cash-yield reference. The complete 175-entry catalogue is not yet historically tested. These charts are retrospective diagnostics.</p>'
html+=scatter.to_html(full_html=False,include_plotlyjs=True)+fig.to_html(full_html=False,include_plotlyjs=False)
html+='<p>Cash, dividends, fees and adverse execution are modeled locally. Sources and assumptions are in REPORT.md and manifest.json. The six portfolio Pine ports show signals only. No future performance is guaranteed.</p></body></html>'
(OUT/'dashboard.html').write_text(html)
print('Wrote REPORT.md, dashboard.html, cash_reference.json and updated full catalogue status.')
