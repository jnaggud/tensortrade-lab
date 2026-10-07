"""Summarize the second expansion without altering the original shortlist."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).parent;OUT=HERE/'signed_results'
r=pd.read_json(OUT/'metrics.json')
full=r[(r.window=='full')&(r['case']=='base')]
rank=json.loads((OUT/'ranking.json').read_text())
diagnostics={x['strategy']:x for x in json.loads((OUT/'frictionless_diagnostic.json').read_text())}
names={'LSMomentum':'Long/short sector momentum','LSLowVol':'Long/short low volatility','LSIBS':'Long/short internal bar strength','LSCluster':'Single-cluster mean reversion','LSMultiCluster':'Three-cluster mean reversion','LSWeightedResidual':'Weighted-regression mean reversion','LSPair':'XLP/XLY relative-return pair','BuyHold':'SPY buy-and-hold','PassiveBasket':'Passive nine-sector basket','Cash':'Cash-yield proxy'}
def pct(n):return f'{100*n:,.2f}%'
parts=['# Second expansion: seven long/short ETF rules\n\n',
       '**None of the seven rules met the required return/drawdown criterion.** This batch adds 80 runs, including matched benchmarks, chronological blocks and stress cases. ',
       'All seven also failed against SPY in the full-period base case. The existing ten provisional Pine scripts were left unchanged.\n\n',
       '## What changed\n\n',
       'The earlier study mostly bought assets or held cash. This expansion can hold positive and negative share positions at the same time. ',
       'It tracks short-sale cash alongside the matching liability, pays distributions owed on borrowed shares, charges borrow fees across calendar days and reserves collateral. ',
       'Only free cash earns the lagged Treasury-yield proxy. Trading decisions use completed bars and fills use the next open.\n\n',
       'These are fixed ETF adaptations of paper sections 3.1, 3.4, 3.8, 3.9, 3.9.1, 3.10 and 4.4. Three previously untested catalogue entries now have tested adaptations: pairs, multiple clusters and weighted regression. ',
       'Total catalogue coverage is now 22 of 175 rows with tested adaptations, covering 31 recipe families across the two expanded batches. The two batches contain 829 runs, plus seven separate frictionless diagnostics. These counts are not independent statistical trials.\n\n',
       '## Full-period results\n\n',
       '3 January 2017 through 21 September 2026. USD10,000 initial capital; returns compound after modeled costs. Drawdown is the largest close-to-close decline from an earlier equity peak.\n\n',
       '| Rule | Net cumulative return | Maximum drawdown | Frictionless diagnostic return |\n|---|---:|---:|---:|\n']
for row in full.to_dict('records'):
    gross=pct(diagnostics[row['strategy']]['return']) if row['strategy'] in diagnostics else '—'
    parts.append(f'| {names[row["strategy"]]} | {pct(row["return"])} | {pct(row["max_drawdown"])} | {gross} |\n')
parts+=['\nThe frictionless column was added **after** the fixed screen solely to investigate failures. It removes trading costs, borrow charges and cash interest; it is not ranked or presented as executable performance. ',
        'The daily cluster rules had some positive frictionless returns, but frequent rebalancing overwhelmed those returns at the declared costs. Pair trading and low-volatility long/short were negative even in this diagnostic. ',
        'This rejects these particular recipes for the stated objective; it does not disprove every implementation of the paper\'s broad ideas.\n\n',
        'The momentum rule\'s +1.68% was below both SPY (+299.55%) and the cash-yield proxy (+27.75%), despite its lower drawdown. A smoother ride alone does not satisfy the user\'s objective.\n\n',
        '## Test design and limitations\n\n',
        '- Gross exposure at each rebalance is at most 100% of equity; the signed rules target equal long/short dollars. Positions can drift between rebalances.\n',
        '- Base costs: 5bps commission plus 5bps adverse execution on each traded side, 2% annualized assumed borrow and terminal liquidation costs. Stress cases use doubled trading costs, 10% borrow, an extra signal delay and no cash yield. Three chronological blocks are also scored.\n',
        '- Borrow availability and fees are assumptions, not historical loan quotes. ETF universe selection, fixed group assignments, daily bars, ex-date cash treatment, fractional shares and lack of intraday liquidation/recall modeling limit realism.\n',
        '- Reserve 150% of short market value as cash collateral. A completed-close breach liquidates at the next open; an opening breach also liquidates then. This is a simplified research convention, not a broker/regulatory margin implementation. No collateral liquidations or insolvencies occurred in the historical runs; both paths are covered by adversarial unit tests.\n',
        '- The weighted regression uses an intercept and previously estimated SPY betas, with inverse-variance weights. Target dollar neutrality, beta neutrality and cluster neutrality were checked. Future-price mutations could not alter earlier signals or trades.\n',
        '- All data and dates were already available for retrospective research; there is no untouched holdout, prospective track record or proof of future outperformance.\n\n',
        '14 new accounting/causality tests passed. Ten stored full-period histories had valid finite equity and fully closed terminal positions. All input and protocol hashes matched.\n\n',
        'The [SEC short-sales bulletin](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-51) supports the borrow/dividend accounting mechanics. ',
        'Actual account requirements depend on the applicable rules and broker; see [FINRA Rule 4210](https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210).\n\n',
        '## Reproduce\n\n```sh\n.venv/bin/pytest -q tests/test_signed_paper151.py\n.venv/bin/python research/paper151/signed_study.py\n.venv/bin/python research/paper151/signed_study.py --diagnostics-only\n.venv/bin/python research/paper151/catalogue.py\n.venv/bin/python research/paper151/make_signed_report.py\n```\n\n',
        'The fixed protocol is in `../SIGNED_PROTOCOL.md`; source hashes are in `manifest.json`; metrics are in `metrics.json`; full histories, fills and decisions are in `ledgers/`. ',
        '`frictionless_diagnostic.json` preserves the additional non-ranked diagnostic separately.\n\n',
        '## Next research sequence\n\n',
        '1. Public factor data: implement residual momentum and alpha rotation with factors available before each decision. These can extend coverage without waiting for the external drive.\n',
        '2. Options/futures: verify the raw Expansion archive when accessible, then build actual-contract roll accounting and a multi-leg options engine. A few snapshots or option-derived features are insufficient.\n',
        '3. Combine eligible evidence, reject unstable or cost-sensitive candidates, reconcile local/Pine signals and accounting, and then revise the provisional top ten. Keep a prospective paper-trading test separate from the historical screen.\n']
(OUT/'REPORT.md').write_text(''.join(parts))
print('Wrote signed_results/REPORT.md')
