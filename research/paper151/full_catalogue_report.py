"""Reconcile every paper row with real evidence, without upgrading scenarios to history."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json,html
import numpy as np
import pandas as pd
from full_catalogue_options import HERE,OUT,save,sha
from full_catalogue_prices import RULES

NEEDS={
 '3.2':'As-released earnings, estimates/surprises and publication times; investable stock universe including delistings and corporate actions.',
 '3.3':'As-filed book values with availability dates, historical stock universe/delistings and borrow; current fundamentals are not historical inputs.',
 '3.5':'Historical single-stock call/put implied-volatility panels and actual underlying/option quotes; ES index options cannot supply a cross-section of stocks.',
 '3.16':'Merger announcement/termination dates, dated cash/stock deal terms, target/acquirer prices and stock borrow.',
 '3.19':'Calibrated order-queue, latency, cancellation and adverse-selection execution model. Local 2026 MBO/depth archives exist, but have not been validated for a market-maker fill simulation.',
 '5.9':'Dated individual-bond ratings, maturities, quotes, defaults and cash flows for the low-risk sorts.',
 '5.10':'Dated bond spreads, ratings and maturities for cross-sectional value regressions, including failed issuers.',
 '6.2':'Synchronous index-basket and futures quotes, historical index weights, expected dividends, funding, borrow and latency.',
 '6.3':'Matched single-stock/index option chains, contemporaneous constituent weights and actual option lifecycle/hedging costs.',
 '6.3.1':'The same dispersion inputs plus validated constituent subset/risk model; no single-stock option chain panel has been staged.',
 '6.4':'Synchronous quotes and sizes for two ETFs on the same index, settlement/borrow and a spread-fill latency model.',
 '7.2':'Individual CFE VIX futures settlements/quotes and expiry calendars matched with contemporaneous VIX. The staged GLBX futures are different instruments.',
 '7.3.1':'Dated medium-term VIX futures and rolls, ETN quotes/actions and historical short availability.',
 '7.6':'Actual variance-swap strike quotes, notional/settlement terms, funding and counterparty costs. Realized returns alone cannot price a historical entry.',
 '8.5':'Simultaneous three-currency executable bid/ask quotes, sizes, venue timestamps and latency; daily currency ETF bars are insufficient.',
 '9.2':'CFTC positions with original public-release dates and revisions, mapped to individual commodity futures. Public COT data have not yet been staged/validated.',
 '9.6':'Physical spot and actual term-structure quotes plus rolling out-of-sample calibration; a mathematical model identity is not a tested forecast.',
 '10.1':'Specified physical/portfolio exposure and dated matching futures prices, quantities, delivery and basis; hedge success must be measured against that exposure.',
 '10.1.1':'The asset being hedged, both assets\u2019 historical prices and a causal hedge-ratio fit, plus futures execution/roll data.',
 '10.1.2':'Bond cash flows and changing dollar durations, deliverable basket/conversion factors and dated interest-rate futures.',
 '10.2':'Complete observed held/deferred prices. Original and liquid-deferred retry both fail on concrete missing daily bars; no prices have been imputed.',
 '11.7':'MBS pool prices, actual prepayments, cash flows and contemporaneous swap quotes; calibrated prepayment/duration model.',
 '14.1':'Historical inflation-swap quotes and contractual CPI reference vintages/lags, settlement and collateral funding.',
 '14.2':'Matched TIPS/Treasury CUSIP quotes/cash flows, inflation-swap prices, accrued interest and financing; ETF prices cannot establish this arbitrage.',
 '14.3':'Location-specific weather settlements, traded weather contract quotes, and the actual business demand/revenue exposure being hedged.',
 '14.4':'Region/delivery-matched power and gas futures, units and venue contract specifications, plus plant heat-rate and operating exposure. No validated joint panel is staged.',
 '15.1':'Dated distressed-debt purchase quotes, defaults, recovery cash flows, legal expenses and an investable historical universe.',
 '15.3':'As-filed balance sheets, point-in-time bankruptcy/default labels and delisting outcomes; no survivorship-safe panel is staged.',
 '15.3.1':'The same distress features/labels plus position limits and exposure history for a specified portfolio.',
 '16.4':'Regional property total-return series with original publication vintages and actual investable implementation/transaction costs.',
 '16.5':'Property rents, valuations, financing and vintage inflation data; stock-market REIT returns do not establish direct-property inflation hedging.',
 '16.6':'Property-level acquisition, renovation, financing, taxes/carrying, resale proceeds and time-to-sale histories.',
 '17.3':'The portfolio\u2019s historical liquidity demands, realizable asset-sale costs and permissible reserve policy.',
 '17.4':'Historical repo rates, collateral valuations/haircuts, margin calls, counterparty defaults and settlement conventions.',
 '17.5':'Lawful loan-level origination/default/repayment data, collateral appraisals and realized resale/operating costs.',
 '18.3':'Timestamped original news/social text and point-in-time training labels/vocabulary. Backfilled local sentiment is not eligible.',
 '19.2':'Original-release GDP/inflation forecasts, trade-weighted FX, rates and risk-sentiment inputs across countries; revised series would change the signal.',
 '19.3':'Original-release headline/core inflation and commodity basket histories. Public vintage series have not yet been staged/validated.',
 '19.4':'Dated multi-country bond panels and release-vintage macro/sovereign-risk predictors, financing and currency hedging.',
 '19.5':'Announcement calendars known before trade entry, separating scheduled from surprise/rescheduled events, and matched tradable prices. A reconstructed final calendar would introduce look-ahead.',
}
for s in ['5.2','5.3','5.4','5.5','5.6','5.7','5.8','5.8.1','5.11','5.12','5.13']:
    NEEDS[s]='Individual-bond cash flows, dated bid/ask prices, yields/durations and financing through time. Synthetic yield shocks verify construction only; constant-maturity bond ETFs do not reproduce a fixed-maturity ladder/bullet.'
NEEDS['5.14']='Matched bond/CDS historical quotes, default/recovery settlements and financing; negative quoted basis is not risk-free realized profit.'
NEEDS['5.15']='Dated swap, Treasury and floating-rate/repo quotes with actual funding/collateral conventions and costs.'
for s in ['8.2','8.2.1','8.3','8.4']:
    NEEDS[s]='Dated multi-currency spot/forward bid/ask, short-term funding/rollover rates and currency conversions; equity-style currency ETF bars do not reproduce financed FX carry.'
for s in ['11.2','11.3','11.4','11.5','11.6']:
    NEEDS[s]='Historical tranche/index/single-name CDS quotes, attachment/detachment terms, defaults/recoveries, risky durations, cash flows and funding.'
for s in ['12.1','12.2']:
    NEEDS[s]='Historical convertible issue terms and bond bid/ask, call/conversion/default events, credit/volatility inputs, stock borrow and actual hedges.'
for s in ['13.1','13.2','13.2.1']:
    NEEDS[s]='Investor-specific tax eligibility and dated applicable rules plus actual security/financing cash flows. Only the paper\u2019s conditional algebra is checked; no current tax-law assertion or recommendation.'
for s in ['15.2','15.2.1','15.2.2','15.2.3']:
    NEEDS[s]='Case-level control rights, court/reorganization outcomes, financing, intervention decisions and all legal/operating cash flows. This is an investment/business process, not a complete chart-trading rule.'
for s in ['16.3','16.3.1','16.3.2']:
    NEEDS[s]='Property-level type/geography/economic exposures, rents, transaction/appraisal vintages, financing and operating costs; investable universe and allocation policy.'

LABELS={'historical_adaptation':'Historical adaptation','historical_episode_only':'Historical quote episodes only',
 'synthetic_construction_only':'Synthetic construction only','blocked_data_or_model':'Blocked: data/model','excluded':'Excluded'}

def read(name):return json.loads((OUT/name).read_text())
def pct(x):return f'{100*x:.2f}%'

def build():
    required=['prices_manifest.json','additional_manifest.json','value_manifest.json','activity_manifest.json','calendar_manifest.json','options_manifest.json','volatility_manifest.json','scenario_manifest.json','validation.json']
    for name in required:
        if not (OUT/name).exists():raise ValueError('Study still incomplete: '+name)
    previous=HERE/'history/2026-10-02-final';catalogue=json.loads((previous/'catalogue.json').read_text())
    prices=read('prices_results.json')+read('additional_results.json')+read('value_results.json')+read('activity_results.json')+read('calendar_results.json')
    for x in prices:
        market=x.get('market','')
        x['benchmark_label']=market if market.startswith('FX') else 'VNQ' if market=='REITGeo' else 'GC/CL passive futures' if x['section'] in ['9.4','10.2'] else 'Five-market passive futures' if x['section']=='10.3.1' else 'SPY'
    options=read('options_results.json');vol=read('volatility_results.json');scenarios=read('scenario_results.json')
    errors=read('options_errors.json')+read('volatility_errors.json')+read('calendar_errors.json')
    entries=[]
    for original in catalogue:
        r=dict(original);s=r['section'];new=[x for x in prices if x['section']==s]
        episodes=[x for x in options+vol if x['section']==s]
        if s=='7.5':episodes=[x for x in options if x['section']=='2.12']
        checks=[x for x in scenarios if x['section']==s]
        failures=[x for x in errors if x.get('section')==('2.12' if s=='7.5' else s)]
        old=original['status']=='historically_tested_adaptation'
        status='excluded' if s in ['17.2','17.6'] else 'historical_adaptation' if old or new else 'historical_episode_only' if episodes else 'synthetic_construction_only' if checks else 'blocked_data_or_model'
        r.update(status=status,prior_status=original['status'],historical_completed_cases=len(new),episode_completed_cases=len(episodes),failed_closed_cases=len(failures),synthetic_checks=len(checks),qualified=False)
        evidence=list(original.get('evidence',[]))
        if new:
            evidence += ['full_catalogue/'+f for f in ['prices_results.json','additional_results.json','value_results.json','activity_results.json','calendar_results.json'] if any(x['section']==s for x in read(f))]
            r['recipes']=sorted(set(r.get('recipes',[])+[x['strategy'] for x in new]))
            desc=[RULES[name]['description'] for name in r['recipes'] if name in RULES]
            if s=='7.3':desc=['Monthly short VXX/long VXZ regression hedge; 2019-2026 series only, 5% assumed borrow and 20% stress. Short availability absent; collateral liquidations recorded.']
            if s=='9.4':desc=['Five-calendar-year raw near-futures price ratio on GC/CL, a physical-spot proxy. 2022-2026 actual-contract P&L and costs; no full-sample pass.']
            if s=='10.3.1':desc=['2018 only: volume/open-interest filtered five-future contrarian adaptation, received-time OI eligibility, weekly rebalance. Missing input means cash; not a multi-year validation.']
            if s=='20':desc=['80/20 SPY/IGF public-fund diversification, monthly rebalance. Direct project cash flows are separately synthetic.']
            r['notes']=' '.join(desc) or r.get('notes','')
        if episodes:
            evidence+=['full_catalogue/'+('volatility_results.json' if s in ['7.4','7.4.1'] else 'options_results.json')]
            r['episode_months']=sorted(set(x['episode'][:7] for x in episodes if x['case']=='base'))
            r['episode_base_passes']=sum(x.get('pass_in_episode_only',False) for x in episodes if x['case']=='base')
            r['notes']='Twelve scheduled three-day ES weekly-option samples, July 2025-June 2026. Only completed cases counted; bid/ask, visible size and commissions; no full-year/expiry/margin/latency qualification.'
            base=[x for x in episodes if x['case']=='base']
            r['notes']+=f' Completed {len(base)} base episodes; median three-day return {pct(float(np.median([x["return"] for x in base])))}; largest sampled drawdown {pct(max(x["max_sampled_drawdown"] for x in base))}; {r["episode_base_passes"]}/{len(base)} episode-only benchmark passes. Not annualized.'
            if s=='7.5':r['notes']+=' Same risk-reversal construction as §2.12; linked evidence, not another independent experiment.'
            if s in ['7.4','7.4.1']:r['notes']+=' Model IV/realized-volatility entry gate; §7.4.1 has discrete hourly integer-futures hedges.'
        if checks:evidence.append('full_catalogue/scenario_results.json')
        if failures:evidence.append('full_catalogue/'+('calendar_errors.json' if s=='10.2' else 'volatility_errors.json' if s.startswith('7.') else 'options_errors.json'))
        r['evidence']=sorted(set(evidence))
        r['historical_gap']=NEEDS.get(s,'' if old or new else original['requirements'])
        if status=='excluded':r['historical_gap']='Descriptive noncandidate; no trading implementation.'
        if status=='synthetic_construction_only':r['notes']='Construction/cash-flow checks passed using expressly hypothetical inputs. Historical performance has NOT been tested. '+r.get('notes','')
        entries.append(r)
    if len(entries)!=175 or len({x['section'] for x in entries})!=175:raise AssertionError('Incomplete catalogue')
    counts=dict(Counter(r['status'] for r in entries));save(OUT/'coverage.json',entries)
    save(OUT/'readiness_gates.json',[{'section':r['section'],'status':r['status'],'historical_test_complete':r['status'] in ['historical_adaptation','historical_episode_only'],'scope':LABELS[r['status']],'missing_for_original_or_full_execution':r['historical_gap']} for r in entries])
    # Distinct counts: one case per candidate/window/stress; matched benchmarks separate.
    masks=[x for x in prices if x['window']=='full' and x['case']=='base']
    rankings=[]
    for name in sorted(set(x['strategy'] for x in prices)):
        group=[x for x in prices if x['strategy']==name]
        rankings.append({'strategy':name,'section':group[0]['section'],'completed_cases':len(group),'passes':sum(x['pass'] for x in group),
          'all_case_gate_pass':all(x['pass'] for x in group),'execution_qualified':False,
          'full_base':next((x for x in group if x['window']=='full' and x['case']=='base'),None)})
    save(OUT/'rankings.json',rankings)
    episode_summary=[]
    for section in sorted({x['section'] for x in options+vol},key=lambda x:tuple(map(int,x.split('.')))):
        base=[x for x in options+vol if x['section']==section and x['case']=='base']
        episode_summary.append({'section':section,'base_episodes':len(base),'episode_passes':sum(x['pass_in_episode_only'] for x in base),
          'median_three_day_return':float(np.median([x['return'] for x in base])),
          'worst_three_day_return':min(x['return'] for x in base),'best_three_day_return':max(x['return'] for x in base),
          'largest_sampled_drawdown':max(x['max_sampled_drawdown'] for x in base),'qualified':False})
    save(OUT/'option_episode_summary.json',episode_summary)
    audits=list((OUT/'options').glob('*/quotes_*.audit.json'));oa=[json.loads(p.read_text()) for p in audits]
    summary={'at':datetime.now(timezone.utc).isoformat(),'catalogue_rows':175,'all_rows_accounted_for':True,
      'full_catalogue_historical_backtest_complete':False,'coverage_status_counts':counts,
      'catalogue_entries_with_some_historical_test':counts.get('historical_adaptation',0)+counts.get('historical_episode_only',0),
      'new_price_candidate_cases':len(prices),'new_price_benchmark_executions':len(prices),'new_price_recipes':len(rankings),
      'new_option_episode_cases':len(options),'new_volatility_episode_cases':len(vol),'failed_option_cases':len(read('options_errors.json')),
      'failed_volatility_cases':len(read('volatility_errors.json')),'failed_calendar_retry_cases':len(read('calendar_errors.json')),
      'synthetic_checks':len(scenarios),'synthetic_sections':len({x['section'] for x in scenarios}),
      'full_sample_price_passes':sum(x['pass'] for x in masks),'all_case_price_gate_passes':sum(x['all_case_gate_pass'] for x in rankings),
      'execution_qualified':0,'option_quote_files':len(audits),'option_quote_rows':sum(x['selected_rows'] for x in oa),
      'option_raw_source_hashes_matched':sum(x['hash_status']=='matched_prior_hash' for x in oa),
      'option_raw_source_hashes_new_baseline':sum(x['hash_status']!='matched_prior_hash' for x in oa),
      'open_interest_source_files':read('open_interest_manifest.json')['files'],'open_interest_rows':read('open_interest_manifest.json')['rows'],
      'validation':read('validation.json'),'previous_checkpoint':'history/2026-10-02-final/current_manifest.json',
      'notes':['Prior 27 historically adapted entries retained; prior Pattern_FindR/quant tests and ten exploratory Pine files unchanged.',
        'Option/volatility episodes are not continuous annual strategy backtests. §7.5 shares §2.12 evidence and is not double-counted in experiment totals.',
        'Scenarios test algebra and accounting, never historical profits. Blocked entries are not counted as tested.',
        'One-year activity-filter sample is under historical_adaptation; its limited date range remains explicit.',
        'No optimization/search for profitable parameters, no live orders, no new frozen prospective settings.']}
    save(OUT/'manifest.json',summary)
    report=['# Full-catalogue research pass\n\n',f'Updated {summary["at"]}. Every one of the paper\u2019s 175 catalogue rows now has an explicit disposition. **This is not 175 completed historical backtests.**\n\n',
      '| Evidence level | Entries | Meaning |\n|---|---:|---|\n']
    meaning={'historical_adaptation':'Historical prices tested with declared adaptations; original-universe and execution limits remain.',
      'historical_episode_only':'Short observed-quote samples only; insufficient for full-history qualification.',
      'synthetic_construction_only':'Hypothetical formula/cash-flow checks only; no historical return claim.',
      'blocked_data_or_model':'Concrete required data or execution/business model has not been validated.',
      'excluded':'Two descriptive unlawful activities are not strategy candidates.'}
    for s,label in LABELS.items():report.append(f'| {label} | {counts.get(s,0)} | {meaning[s]} |\n')
    report.append(f'\nThe new work completed **{len(prices)} price-based candidate cases** and {len(prices)} matched benchmark executions, **{len(options)} option-construction episodes**, **{len(vol)} additional volatility episodes**, and **{len(scenarios)} synthetic construction checks**. A case is a market/period/cost setting, not a new independent strategy.\n\n')
    report.append('## What the new historical tests found\n\n| Strategy | Period start | Benchmark | Net return | Matched buy-and-hold | Maximum drawdown | Benchmark drawdown | Full-period pass | All cases |\n|---|---|---|---:|---:|---:|---:|---|---|\n')
    for rank in rankings:
        x=rank['full_base']
        if x is None:continue
        ret=x.get('return',x.get('total_return'))
        report.append(f'| {x["strategy"]} | {x["start"][:10]} | {x["benchmark_label"]} | {pct(ret)} | {pct(x["benchmark_return"])} | {pct(x["max_drawdown"])} | {pct(x["benchmark_max_drawdown"])} | {"Yes" if x["pass"] else "No"} | {rank["passes"]}/{rank["completed_cases"]} |\n')
    report.append('\nPass means higher after-cost return and no worse sampled maximum drawdown. Different rows use different markets and dates; returns must be compared with the matched benchmark in the same row. Price studies retain daily-close drawdown limitations. None is execution-qualified, and a short favorable sample is not a winner.\n\n')
    report.append('## Option and volatility tests\n\nAll 58 §2 constructions have a leg registry; §7.4/7.4.1 add the IV/RV gate and hourly integer-futures delta hedge. §7.5 shares the §2.12 risk reversal. Twelve fixed first-Monday-to-Wednesday samples cover July 2025 through June 2026. These are ES options-on-futures adaptations of stock-option constructions. Entry is Monday 14:00 UTC; exit Wednesday 19:55 UTC, before expiry. Each independent sample uses $1m reference capital, one construction unit, $50/point, observed bid/ask, displayed size, $2.50 per contract side, and an extra adverse-tick stress.\n\n')
    report.append('The original five-point strike-rounding rule requested some unlisted contracts. Before any option return scoring, version 2 selected the nearest complete strike grid from definitions already known at the decision. The original protocol, code, selections and extracts are retained under `options_selection_v1/`. Dates, costs, holding periods and payoff structures were unchanged. Far expiries on the same underlying were never invented.\n\n')
    report.append(f'{len(read("options_errors.json"))} option cases and {len(read("volatility_errors.json"))} volatility cases failed closed. The detailed JSON records the required missing contract or quote. Intermediate missing marks are counted, and drawdown is labelled sampled. There is no historical margin, simultaneous multi-leg execution, quote-update-age, assignment or overnight-tick-risk validation. $1m does not cap unbounded short-option losses. No annual return is inferred from these sparse episodes.\n\n')
    report.append('## Data and verification\n\n')
    report.append(f'- Frozen Yahoo daily prices and distributions: earlier ETFs plus 15 supplemental ETFs/ETNs, ending September 23, 2026; sector panel ends September 21. Revised research prices, not exchange fill records. VXX/VXZ are only the retrieved post-2018 series.\n- Existing actual-contract ES/NQ/GC/CL/ZN daily prices, 2017-June 2026, with prior-volume selection and actual roll costs. Commodity value uses GC/CL near-futures proxies for physical spot. The activity filter evaluates 2018 only.\n- {summary["open_interest_source_files"]} statistics source files, {summary["open_interest_rows"]:,} extracted OI observations; decisions respect receipt time, reference session, corrections/deletes and freshness.\n- {len(audits)} option quote-day extracts, {summary["option_quote_rows"]:,} rows; {summary["option_raw_source_hashes_matched"]} raw files matched a saved hash and {summary["option_raw_source_hashes_new_baseline"]} establish new baselines where the archive had no prior hash. Definitions were checked separately.\n- Annual archived Fama-French factors use conservative publication availability; no future factor revision is used as an earlier signal.\n- Frozen prospective source files and freeze hash remain unchanged. Tests, fill reconciliations and source checks are in `validation.json`.\n\n')
    report.append('## What remains untested\n\n`coverage.json`, `readiness_gates.json` and the searchable dashboard retain a row for every entry, including the exact blocker. Gaps include as-released company fundamentals/earnings, single-stock option chains, merger/default/recovery histories, actual bonds/CDS/swaps/convertibles, CFE VIX futures, financed FX quotes, original-release macro/COT inputs, validated market-making execution, property and business cash flows, and investor-specific tax eligibility. Some public inputs may be obtainable; they have not yet been acquired and validated. A synthetic formula check does not close these gaps.\n\nThe calendar-spread retry also stopped in every one of its 16 cases because a held/deferred contract lacked a required daily price. Both generations of failures remain visible. Neither futures curve ratios nor missing-price interpolation are booked as trading profit.\n\n')
    report.append('## Reproduce and inspect\n\nRun the scripts with the repository `.venv/bin/python`. `full_catalogue_options.py stage` extracts the fixed contracts; `score` evaluates them. `full_catalogue_volatility.py` evaluates the additional volatility recipes. `full_catalogue_prices.py`, `full_catalogue_additional.py`, `full_catalogue_value.py`, and `full_catalogue_activity.py score` reproduce price tests. `full_catalogue_calendar.py` preserves the retry failures. `full_catalogue_scenarios.py` runs only artificial construction checks. Then run `full_catalogue_validate.py` and `full_catalogue_report.py`. No script submits orders. Do not alter `prospective.py`, `portfolio.py` or the frozen forward record.\n')
    (OUT/'REPORT.md').write_text(''.join(report))
    # Catalogue is the complete durable manifest, with earlier evidence retained.
    save(HERE/'catalogue.json',entries)
    md=['# Complete 175-entry catalogue\n\nA row is accounted for, not necessarily historically tested. See [full research report](full_catalogue/REPORT.md) and [searchable dashboard](full_catalogue/dashboard.html).\n\n| Section / PDF page | Strategy | Evidence level | Historical gap / limitation |\n|---|---|---|---|\n']
    for r in entries:md.append(f'| {r["section"]} / {r["pdf_page"]} | {r["title"]} | {LABELS[r["status"]]} | {(r["historical_gap"] or r["notes"]).replace("|","/")} |\n')
    (HERE/'CATALOGUE.md').write_text(''.join(md))
    make_dashboard(entries,summary,rankings)
    old=json.loads((previous/'current_manifest.json').read_text());current=dict(old)
    current.update(asof=summary['at'],catalogue_status=counts,historically_tested_catalogue_entries=counts.get('historical_adaptation',0),
      historical_any_scope_catalogue_entries=summary['catalogue_entries_with_some_historical_test'],full_catalogue=summary,
      previous_checkpoint='history/2026-10-02-final/current_manifest.json',unit_and_regression_checks_passed=summary['validation']['full_suite_passed'],
      remaining_work=['Historical data/model blockers in full_catalogue/readiness_gates.json; scenarios and short episodes do not complete full-catalogue historical validation.',
        'No strategy is execution-qualified. Preserve the frozen twelve-month prospective observation through at least October 2027.'])
    current['legacy_count_note']='Original recipe/run/PF/Pine fields outside full_catalogue retain the Oct 2 convention; new experiments and coverage are counted explicitly under full_catalogue.'
    save(HERE/'current_manifest.json',current)
    (HERE/'CURRENT_RESULTS.md').write_text('# Current paper research results\n\n'+f'Updated {summary["at"]}. Every catalogue row is accounted for, but full historical testing is incomplete.\n\n'+
        '\n'.join(f'- {counts.get(s,0)}: {label}.' for s,label in LABELS.items())+
        f'\n\nThe new work completed {len(prices)} price-based candidate cases plus matched benchmarks, {len(options)} option-construction cases, {len(vol)} additional volatility cases, and {len(scenarios)} synthetic checks. No broadly qualified new winner.\n\nThe full results, definitions and limitations are in [full_catalogue/REPORT.md](full_catalogue/REPORT.md). Open the [searchable 175-entry dashboard](full_catalogue/dashboard.html).\n\nNo new execution-qualified winner or Pine promotion. The prior ten exploratory Pine scripts and four frozen prospective candidates are unchanged. Earlier Pattern_FindR/quant results remain in [next_batch/REPORT.md](next_batch/REPORT.md); the prior checkpoint is preserved in `history/2026-10-02-final/`.\n')
    print(json.dumps({k:summary[k] for k in ['coverage_status_counts','new_price_candidate_cases','new_option_episode_cases','new_volatility_episode_cases','execution_qualified']},indent=2))

def make_dashboard(entries,summary,rankings):
    payload=json.dumps({'entries':entries,'summary':summary,'ranks':rankings,'episodes':read('option_episode_summary.json'),'labels':LABELS},ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>151 Trading Strategies · Full catalogue</title>
<style>body{font:16px/1.5 system-ui;margin:0;background:#f6f7fa;color:#142034}main{max-width:1450px;margin:auto;padding:32px}h1{font-size:32px;margin:0 0 8px}h2{margin-top:32px}p{max-width:1000px}nav{display:flex;gap:10px;flex-wrap:wrap}input,select{font:inherit;padding:10px;border:1px solid #a9b5c6;border-radius:6px}input{min-width:300px}.cards{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}.card{background:white;padding:16px;border:1px solid #d9e1ea;border-radius:8px;min-width:145px}.card b{display:block;font-size:30px;color:#244fc0}.warning{padding:16px;background:#fff2d6;border-left:4px solid #c48813}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce3ec;vertical-align:top}th{background:#edf1f7;position:sticky;top:0}td:nth-child(2){min-width:180px}small{color:#536377}details summary{cursor:pointer;color:#244fc0}a{color:#1745ac}.scroll{overflow:auto;max-height:680px}.badge{white-space:nowrap;font-size:12px;padding:4px 7px;border-radius:4px;background:#e8eef6}.historical_episode_only{background:#fff0ce}.synthetic_construction_only{background:#e9e5ff}.blocked_data_or_model{background:#fce3e3}.excluded{background:#ddd}#shown{margin:12px 0}button{font:inherit;cursor:pointer;padding:8px 12px;border-radius:6px;border:1px solid #99a8b8;background:white}</style>
<main><h1>151 Trading Strategies · Full catalogue</h1><p>All 175 entries are accounted for. Historical adaptations, short quote samples, synthetic construction checks and untested entries are shown separately.</p><p class="warning"><b>This is not 175 completed historical backtests.</b> No strategy is execution-qualified. A short favorable episode does not establish a winner. Four frozen prospective candidates and the prior ten exploratory Pine scripts remain unchanged.</p>
<div class="cards" id="cards"></div><p><a href="REPORT.md">Detailed report</a> · <a href="coverage.json">Full coverage JSON</a> · <a href="validation.json">Verification</a> · <a href="readiness_gates.json">Exact data/model gaps</a></p>
<h2>Catalogue evidence</h2><nav><input id="search" placeholder="Search strategy, section or missing data" aria-label="Search catalogue"><select id="status" aria-label="Evidence level"><option value="">Every evidence level</option></select><select id="chapter" aria-label="Chapter"><option value="">Every chapter</option></select><button id="reset">Reset</button></nav><div id="shown"></div><div class="scroll"><table><thead><tr><th>Paper</th><th>Strategy</th><th>Evidence level</th><th>Details and remaining gap</th></tr></thead><tbody id="body"></tbody></table></div>
<h2>New price-based tests</h2><p>Each return is after modeled costs and compared with its matched benchmark. Different rows cover different markets and periods. Options are deliberately excluded from this long-period comparison.</p><div class="scroll"><table><thead><tr><th>Strategy</th><th>Period</th><th>Return</th><th>Buy & hold</th><th>Max drawdown</th><th>Benchmark drawdown</th><th>Checks passed</th></tr></thead><tbody id="prices"></tbody></table></div><p><small>Past research is not an untouched holdout. Drawdown is sampled, short availability and quote execution are not proven, and hypothetical checks never count as historical performance.</small></p></main>
<script>const DATA=PAYLOAD;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const pct=n=>(100*n).toFixed(2)+'%';
for(const [key,label] of Object.entries(DATA.labels)){document.getElementById('cards').innerHTML+=`<div class="card"><b>${DATA.summary.coverage_status_counts[key]||0}</b>${esc(label)}</div>`;document.getElementById('status').innerHTML+=`<option value="${key}">${esc(label)}</option>`;}
for(const c of [...new Set(DATA.entries.map(x=>x.chapter))])document.getElementById('chapter').innerHTML+=`<option>${esc(c)}</option>`;
function draw(){const s=document.getElementById('search').value.toLowerCase(),type=document.getElementById('status').value,ch=document.getElementById('chapter').value;const rows=DATA.entries.filter(x=>(!type||x.status===type)&&(!ch||x.chapter===ch)&&JSON.stringify(x).toLowerCase().includes(s));document.getElementById('shown').textContent=`${rows.length} of 175 entries`;
document.getElementById('body').innerHTML=rows.map(x=>`<tr><td>§${esc(x.section)}<br><small>PDF p.${x.pdf_page}</small></td><td>${esc(x.title)}<br><small>${esc(x.chapter)}</small></td><td><span class="badge ${x.status}">${esc(DATA.labels[x.status])}</span><br><small>${x.historical_completed_cases} new price cases · ${x.episode_completed_cases} episode cases · ${x.synthetic_checks} synthetic checks</small></td><td>${esc(x.notes)}<details><summary>Evidence and required next input</summary><p>${esc(x.historical_gap||'See study-specific execution limitations.')}</p><p>${x.evidence.map(e=>`<a href="../${encodeURI(e)}">${esc(e)}</a>`).join('<br>')}</p><small>Failed closed cases: ${x.failed_closed_cases}. Prior status: ${esc(x.prior_status)}.</small></details></td></tr>`).join('');}
for(const id of ['search','status','chapter'])document.getElementById(id).addEventListener('input',draw);document.getElementById('reset').onclick=()=>{for(const id of ['search','status','chapter'])document.getElementById(id).value='';draw()};draw();
document.getElementById('prices').innerHTML=DATA.ranks.filter(r=>r.full_base).map(r=>{const x=r.full_base;return `<tr><td>${esc(r.strategy)}</td><td>${esc(x.start.slice(0,10))}–${esc(x.end.slice(0,10))}</td><td>${pct(x.return??x.total_return)}</td><td>${pct(x.benchmark_return)}</td><td>${pct(x.max_drawdown)}</td><td>${pct(x.benchmark_max_drawdown)}</td><td>${r.passes}/${r.completed_cases}</td></tr>`}).join('');</script></html>'''
    (OUT/'dashboard.html').write_text(page.replace('PAYLOAD',payload))

if __name__=='__main__':build()
