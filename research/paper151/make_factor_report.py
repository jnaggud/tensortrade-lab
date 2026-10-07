from pathlib import Path
import json
import pandas as pd
HERE=Path(__file__).parent;OUT=HERE/'factor_results'
r=pd.read_json(OUT/'metrics.json');full=r[(r.window=='full')&(r['case']=='base')];rank=json.loads((OUT/'ranking.json').read_text())
pct=lambda x:f'{x*100:,.2f}%'
parts=['# Factor strategies: third research batch\n\n',
'**None qualified for promotion.** Three fixed recipes for two paper ideas were tested in three factor-data modes, with 96 benchmark, block and stress runs. The three expanded batches now contain 34 recipe families and 925 runs; repeated data/cost modes are not independent statistical trials. Twenty-four of the 175 catalogue rows have historically tested adaptations.\n\n',
'## What the rules mean\n\n',
'Residual momentum asks which sectors have done well beyond what broad market, size and value exposures explain. The long-only version buys the top three; the long/short version also shorts the bottom three. Alpha rotation instead ranks the average return unexplained by those exposures. Neither is evidence that a regression can identify future winners.\n\n',
'Technically, each ETF uses 36 consecutive monthly excess returns, regressed on an intercept and Mkt−RF, SMB and HML. Residual momentum retains the intercept in the paper’s equation 279 and ranks the last 12 residuals by mean/sample standard deviation. Alpha rotation ranks the fitted intercept. These ETF rules and monthly windows are declared adaptations.\n\n',
'## Results after modeled costs\n\n',
'3 January 2017–21 September 2026; USD10,000 initial capital. Drawdown uses the daily account-value peak.\n\n',
'| Rule | Factor history | Net compounded return | Maximum drawdown |\n|---|---|---:|---:|\n']
for row in full.to_dict('records'):parts.append(f"| {row['strategy']} | {row['factor_mode']} | {pct(row['return'])} | {pct(row['max_drawdown'])} |\n")
parts+=['\nArchived alpha rotation returned **207.66%**, versus **299.55%** for SPY, with a larger maximum drawdown. It passed one individual block, but failed the overall requirement. Archived residual momentum returned **154.35%** long-only and **−21.10%** long/short.\n\n',
'## Why historical releases matter\n\n',
'The French library reconstructs historical returns as its inputs change. Its annual archive contains July cuts released during August. We downloaded the actual linked 2016–2025 releases and the current file through August 2026, preserving hashes. No 2026 archive was listed. See the [official archive](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors_archive.html) and [data methodology](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).\n\n',
'- **Archived:** each annual release becomes eligible on September 1, checked at the prior-close decision time. Signals can be stale for more than a year. This is factor-vintage aware; ETF prices themselves are current revised history.\n',
'- **Revised annual:** current factor values, with the same month cutoffs. Alpha rotation rises from 207.66% to 227.58% solely through changed rankings from revised inputs. This roughly 19.92 percentage-point difference demonstrates why revised factors cannot be casually called point-in-time data. Residual rankings happened to remain unchanged.\n',
'- **Revised monthly:** more frequent updates with a one-full-month publication allowance, but using today’s factor history. This is a sensitivity diagnostic, ineligible for promotion.\n\n',
'The archived mode is deliberately conservative and does not replicate access to every historical monthly factor release. Its failure does not establish that all timely residual-momentum implementations fail. No result is an untouched holdout or a prospective trading record.\n\n',
'## Verification and reproduction\n\n',
'Seven dedicated tests passed: publication boundaries, unchanged earlier signals when future/unpublished factor values are changed, intercept treatment, missing-month rejection and percent-data parsing. All 12 stored base-case histories have finite equity, valid decision-before-fill timestamps and matching input/protocol hashes. Signed borrow availability and collateral remain modeled assumptions.\n\n',
'```sh\n.venv/bin/pytest -q tests/test_factor_paper151.py\n.venv/bin/python research/paper151/factor_study.py\n.venv/bin/python research/paper151/make_factor_report.py\n.venv/bin/python research/paper151/catalogue.py\n```\n\n',
'`../FACTOR_PROTOCOL.md` was frozen before results. `manifest.json` records sources and code hashes; `metrics.json` contains all cases; `ledgers/` stores daily equity, fills, weights, coefficients and archive cutoffs. Public factor files are downloaded by `../download_factors.py`.\n']
(OUT/'REPORT.md').write_text(''.join(parts))
