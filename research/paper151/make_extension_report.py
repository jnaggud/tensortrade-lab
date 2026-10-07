"""One evidence scorecard; preserve previous frozen studies and separate test populations."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import json,hashlib,html
import pandas as pd
import numpy as np
from cross_project_inventory import OUT,save,sha
HERE=OUT.parent
PCT=lambda x:'—' if x is None else f'{100*x:,.2f}%'
BENCH={'BuyHold','Cash','PassiveBasket','PassiveFutures','PassiveCommodity'}

def read(p):return json.loads((HERE/p).read_text())
def main():
    rows=[]
    for folder,origin in [('expanded_results','Paper / daily'),('signed_results','Paper / signed ETF'),('factor_results','Paper / factors')]:
        data=read(folder+'/metrics.json')
        if folder=='factor_results':data=[x for x in data if x['factor_mode']=='archived']
        for r in data:
            if r['window']!='full' or r['case']!='base' or r['strategy'] in BENCH:continue
            market=r.get('market','Sectors');same=[x for x in data if x['strategy']==r['strategy'] and x.get('market','Sectors')==market]
            blocks=[x for x in same if x['window']!='full' and x['case']=='base']
            rows.append({'origin':origin,'strategy':r['strategy'],'market':market,'period':r.get('start','2017-01-03')[:10]+' → '+r.get('end','2026-09-21')[:10],
              'return':r['return'],'benchmark_return':r['return_bh'],'drawdown':r['max_drawdown'],'benchmark_drawdown':r['max_drawdown_bh'],
              'full_pass':r['pass'],'block_pass':f"{sum(x['pass'] for x in blocks)}/{len(blocks)}",
              'costs':r.get('fees',0)+r.get('slippage',0)+r.get('borrow',0),'capital':10000,
              'status':'Historical lead; failed broad gate' if r['pass'] else 'Failed historical comparison',
              'notes':'Frozen September 29 study. Matched buy/hold; explicit distributions and cash-rate convention. Historical sample already explored.',
              'evidence':'../'+folder+'/metrics.json'})
    v=read('extension/velocity_results.json')
    for r in v:
        if r['origin']!='PatternFindr_adaptation' or r['case']!='full':continue
        same=[x for x in v if x['strategy']==r['strategy'] and x['origin']=='PatternFindr_adaptation']
        blocks=[x for x in same if x['case'] in ['2017_2019','2020_2022','2023_latest']]
        rows.append({'origin':'Pattern_FindR','strategy':r['strategy'],'market':r['market'],
          'period':r['start'][:10]+' → '+r['end'][:10],'return':r['total_return'],'benchmark_return':r['benchmark_return'],
          'drawdown':r['max_drawdown'],'benchmark_drawdown':r['benchmark_drawdown'],'full_pass':r['pass'],
          'block_pass':f"{sum(x['pass'] for x in blocks)}/{len(blocks)}",'costs':r['fees']+r['slippage'],'capital':10000,
          'status':'Failed historical comparison','notes':'Saved daily configuration; causal adaptation, next-open execution, gap-aware stops, prior-bar trailing levels. No new optimization. Includes duplicate configurations.',
          'evidence':'velocity_results.json'})
    f=read('extension/futures_results.json')
    def fut_pass(r):
        name='PassiveCommodity' if r['strategy'].startswith('Commodity') else 'PassiveFutures'
        b=next(x for x in f if x['strategy']==name and x['block']==r['block'] and x['case']==r['case'])
        return r['total_return']>b['total_return'] and r['max_drawdown']<=b['max_drawdown'],b
    scored=[]
    for r in f:
        if r['strategy'] in BENCH:continue
        ok,b=fut_pass(r);scored.append({**r,'benchmark':b['strategy'],'benchmark_return':b['total_return'],
          'benchmark_drawdown':b['max_drawdown'],'pass':ok})
        if r['block']!='full' or r['case']!='base':continue
        blocks=[x for x in f if x['strategy']==r['strategy'] and x['case']=='base' and x['block']!='full']
        rows.append({'origin':'Paper / actual futures','strategy':r['strategy'],
          'market':'GC / CL' if r['strategy'].startswith('Commodity') else 'ES / NQ / GC / CL / ZN',
          'period':r['start'][:10]+' → '+r['end'][:10],'return':r['total_return'],'benchmark_return':b['total_return'],
          'drawdown':r['max_drawdown'],'benchmark_drawdown':b['max_drawdown'],'full_pass':ok,
          'block_pass':f"{sum(fut_pass(x)[0] for x in blocks)}/{len(blocks)}",'costs':r['fees']+r['slippage'],'capital':1000000,
          'status':'Failed historical comparison; modeled daily fills',
          'notes':'Actual contract prices and two-leg roll costs; unlevered 95% gross target. 252 UTC buckets are not 252 exchange sessions. No dated margin or executable quote-size validation; zero collateral interest.',
          'evidence':'futures_results.json'})
    save('futures_scored.json',scored)
    quant=read('extension/c11_retest.json')+read('extension/quant_family_results.json')
    for r in quant:
        if r['case']!='1_ticks_per_market_side':continue
        rows.append({'origin':'trading_view_mcp_quant','strategy':r['strategy'],'market':'ES continuous / 15m',
          'period':r['start'][:10]+' → '+r['end'][:10],'return':r['total_return'],'benchmark_return':None,
          'drawdown':r['max_drawdown'],'benchmark_drawdown':None,'full_pass':False,'block_pass':'Not eligible',
          'costs':None,'capital':50000,'status':'Quarantined diagnostic',
          'notes':'Adjusted continuous feed with same-day-volume contract selection, same-close fills and no actual roll ledger. Simulated recovery after insolvency is not investable. '+('Account crossed zero at bar close.' if r['ever_insolvent'] else 'Two-tick stress crossed zero.'),
          'evidence':'c11_retest.json' if r['strategy']=='C11' else 'quant_family_results.json'})
    save('scorecard.json',rows)
    status=read('extension/velocity_status.json');inv=read('extension/inventory_manifest.json');archive=read('extension/actual_futures_manifest.json')
    leads=read('extension/lead_sensitivity.json');counts=defaultdict(lambda:[0,0])
    for r in leads:
        if r['cost_multiple']==1:
            x=counts[(r['spec']['family'],r['block'])];x[0]+=int(r['pass']);x[1]+=1
    cat=read('catalogue.json');fman=read('extension/futures_manifest.json');tv=read('tradingview/promoted/manifest.json')
    decision={'asof':'2026-10-02','retained_scripts':[x['strategy'] for x in tv],
      'newly_promoted':[],'qualified_new_candidates':0,'reason':'Daily Pattern_FindR adaptations and eight completed futures recipes fail full-sample comparisons. C7/C8/C11 are quarantined. Existing ten remain provisional visualization candidates.',
      'mcp_health':'2026-10-02: CDP and chart API connected; BATS:TLT, 1D. Read-only health check.',
      'limitations':'Four native strategies; six multiasset target indicators. No native multiasset P&L account and no all-catalogue winner claim.'}
    save('tradingview_decision.json',decision)
    manifest={'asof':'2026-10-02','paper_recipes_completed':42,'paper_runs_including_benchmarks_and_stresses':925+len(f),
      'historically_tested_catalogue_entries':sum(x['status']=='historically_tested_adaptation' for x in cat),
      'catalogue_rows':len(cat),'catalogue_status':dict(Counter(x['status'] for x in cat)),
      'patternfindr_configs_inventoried':93,'patternfindr_daily_configs_tested':39,'patternfindr_runs':len(v),
      'patternfindr_status':dict(Counter(x['status'] for x in status)),
      'quant_pine_files_inventoried':176,'quant_families_diagnostically_retested':3,'quant_diagnostic_runs':len(quant),
      'lead_nearby_settings':21,'lead_sensitivity_runs':len(leads),'futures_completed_runs':len(f),'futures_failed_closed_runs':16,
      'new_completed_runs_including_diagnostics_and_benchmarks':len(f)+len(v)+len(quant)+len(leads),
      'broadly_qualified_families':0,'private_pine_scripts':10,'native_pine_strategies':4,'multiasset_target_indicators':6,
      'unit_and_regression_checks_passed':87,'signal_prefix_checks_passed':40,'historical_option_strategy_runs':0,
      'forward_frozen_at':read('prospective/freeze.json')['frozen_at'],'forward_version':2,'forward_completed_trades_at_report':0,
      'automation_id':'paper-151-prospective-strategy-recorder','automation_schedule':'Daily 17:15 America/Chicago',
      'shortlist_decision':decision['reason'],'original_checkpoint':'history/2026-09-29/current_manifest.json',
      'remaining_work':['52 saved Pattern_FindR intraday/futures configurations need separate faithful data/execution implementations; one auxiliary-input configuration and one malformed file also remain untested.',
        'Calendar spread missing deferred contract bars; retain failed runs rather than fabricate prices.',
        'C7/C8/C11 need actual-contract intraday reconstruction, quote-level execution and margin history.',
        'Archive options definitions/quote schema sampled only; full contract/underlying joins, lifecycle, margin and quote-quality histories not implemented.',
        'Other paper chapters still require point-in-time fundamentals, borrow/events, macro release vintages and instrument-specific cash flows.',
        'Forward observation requires future time; no prospective performance yet.'],
      'sources':{str(p.relative_to(HERE)):sha(p) for p in [OUT/'actual_futures_manifest.json',OUT/'velocity_manifest.json',OUT/'c11_manifest.json',OUT/'quant_family_manifest.json',OUT/'futures_manifest.json',OUT/'lead_manifest.json',OUT/'scorecard.json',HERE/'prospective/freeze.json']}}
    (HERE/'current_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save('manifest.json',manifest)
    verification={'date':'2026-10-02','pytest_passed':87,'pytest_failed':0,'dependency_warnings':2,
      'pattern_signal_prefix_checks':39,'c11_prefix_comparison_mismatches':0,
      'previous_native_same_feed_execution_events_matched':702,'previous_portfolio_monthly_targets_matched':702,
      'previous_portfolio_daily_target_vectors_matched':14652,'live_orders':0,
      'historical_futures_completed_runs':176,'historical_futures_candidate_runs':128,'historical_options_runs':0}
    (HERE/'verification_latest.json').write_text(json.dumps(verification,indent=2)+'\n')
    # Static figure: unlike a leaderboard it makes failure modes visible.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    fr=[x for x in rows if x['origin']=='Paper / actual futures']
    labels=['Trend: sign','Trend: smooth','Trend: neutral','Reversion','Reversion: vol','Reversion: variance','Carry: gold/oil','Skewness: gold/oil']
    fr.sort(key=lambda r:['FutTrendSign252','FutTrendSmooth252','FutTrendNeutral252','FutContrarian5','FutContrarianVol5','FutContrarianVar5','CommodityCarryRank2','CommoditySkew252Rank2'].index(r['strategy']))
    ax.barh(labels,[r['return']*100 for r in fr],color=['#b84143' if r['return']<0 else '#37789c' for r in fr])
    ax.scatter([r['benchmark_return']*100 for r in fr],labels,color='#23866f',marker='D',s=35,label='Matched passive futures')
    ax.axvline(0,c='#9aa8b5',lw=.7);ax.invert_yaxis();ax.set_xlabel('Compounded return after modeled costs (%)')
    ax.set_title('Eight additional paper recipes\nJan 2018–Jun 2026 • $1m reference account');ax.legend(loc='lower right',fontsize=8);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for name,color in [('C7 Constrained Refine C1','#5089a7'),('C8 Successor 200k C5','#bf8a39'),('C11','#b84143')]:
        series=[x for x in quant if x['strategy']==name]
        bx.plot([int(x['case'].split('_')[0]) for x in series],[x['max_drawdown']*100 for x in series],'o-',label=name.split()[0],color=color)
    bx.axhline(100,c='#333333',ls='--',lw=1,label='100% daily-equity drawdown')
    bx.set_xlabel('Adverse slippage per market side (ticks)');bx.set_ylabel('Largest daily-equity decline (%)')
    bx.set_title('Saved quant candidates: slippage stress\nJun 2021–Jun 2026 • quarantined continuous feed');bx.set_xticks([1,2,3,5]);bx.grid(alpha=.15);bx.legend(fontsize=8)
    fig.suptitle('No new candidate qualifies for promotion',fontsize=17,fontweight='bold')
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    # All JSON is embedded; no network or external JS libraries needed.
    payload=json.dumps(rows,allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trading research evidence scorecard</title>
<style>body{margin:0;background:#f3f5f7;color:#172a3a;font:15px system-ui}main{max-width:1450px;margin:36px auto;padding:0 24px}h1{font-size:32px;letter-spacing:-1px;margin-bottom:8px}p{line-height:1.6;max-width:1050px}.muted{color:#516477}.cards{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}.card{background:white;border:1px solid #d8e1e8;padding:18px;border-radius:12px;min-width:180px;flex:1}.card b{font-size:27px;display:block}.alert{padding:15px 20px;background:#fff1dc;border-left:4px solid #bc812f}img{width:100%;border-radius:12px;background:#fff}.tools{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}input,select{padding:10px 12px;font:inherit;border:1px solid #b9c7d1;border-radius:6px;background:white}input{min-width:290px}.scroll{overflow:auto;max-height:650px;border:1px solid #d5dfe6;border-radius:10px;background:#fff}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:12px;text-align:left;border-bottom:1px solid #e1e7ec;vertical-align:top}th{position:sticky;top:0;background:#e7eef4;cursor:pointer;white-space:nowrap}td.num{white-space:nowrap;text-align:right}td small{display:block;color:#52687a;max-width:330px;line-height:1.4;margin-top:5px}.pass{color:#187359}.fail{color:#a7373c}.foot{font-size:13px}a{color:#176c9c}details{margin:20px 0;background:#fff;padding:18px;border-radius:10px}summary{cursor:pointer;font-weight:650}li{margin:9px 0;line-height:1.5}</style>
<main><div class="muted">RESEARCH CHECKPOINT · 2 OCTOBER 2026</div><h1>What survives a fair test?</h1>
<p>Our target is a higher return after costs, with no larger decline from a previous account peak, than a matched buy-and-hold investment. Some original ideas pass one historical sample. <strong>No strategy has met the broad qualification gate.</strong></p>
<div class="cards"><div class="card"><b>42</b>paper recipes tested<br><span class="muted">27 of 175 catalogue entries adapted</span></div><div class="card"><b>39</b>Pattern_FindR daily configs<br><span class="muted">21 SPY + 18 Bitcoin, including duplicates</span></div><div class="card"><b>3</b>quant candidates audited<br><span class="muted">C7, corrected C8 C5, C11</span></div><div class="card"><b>0</b>new qualifying winners<br><span class="muted">10 Pine scripts remain provisional</span></div></div>
<p class="alert">Return percentages cover different markets, periods and account models. Compare each row with its own benchmark. A high return in one row is not a valid ranking across all rows. Quant results are quarantined and cannot qualify.</p>
<img src="comparison.png" alt="Eight futures recipes trail their matched passive benchmarks; quant candidates suffer severe slippage-sensitive drawdowns.">
<div class="tools"><input id="search" placeholder="Search strategy, market or notes" aria-label="Search strategies"><select id="origin" aria-label="Filter source"><option value="">All sources</option></select><select id="status" aria-label="Filter status"><option value="">All outcomes</option><option value="lead">Original full-sample leads</option><option value="fail">Failed comparison</option><option value="quarantine">Quarantined</option></select><span id="count" class="muted"></span></div>
<div class="scroll"><table><thead><tr><th data-key="strategy">Strategy / source ↕</th><th data-key="market">Market / period ↕</th><th data-key="return">Return ↕</th><th data-key="benchmark_return">Buy/hold ↕</th><th data-key="drawdown">Drawdown ↕</th><th data-key="benchmark_drawdown">Buy/hold DD ↕</th><th>Period passes</th><th>Modeled costs</th><th>Status / evidence</th></tr></thead><tbody id="body"></tbody></table></div>
<details open><summary>What these tests mean in plain English</summary><ul><li><strong>Pattern_FindR:</strong> keep the saved settings, but trade after the signal exists, charge costs, allow stops to slip through gaps and count losses while a trade remains open. None of the 39 completed daily configurations beat buy-and-hold over the full matched history.</li><li><strong>Futures:</strong> use actual contracts and pay to leave the old contract and enter the new one. Eight recipes completed; the calendar-spread recipe stopped when a preselected deferred price was missing. This remains a daily-price execution model.</li><li><strong>C7/C8/C11:</strong> the older feed uses a contract choice that depends on that day's final volume. Its prices are adjusted and the engine assumes same-close fills. The profitable-looking final balances also hide severe intervening losses. These runs diagnose weaknesses; they do not validate investable profit.</li><li><strong>Future test:</strong> four rules frozen before any simulated order; decisions are timestamped before a future fill. Daily recorder scheduled for 5:15 p.m. America/Chicago. No forward trades or performance exist at this checkpoint.</li></ul></details>
<details><summary>What remains untested</summary><p>52 Pattern_FindR configurations need intraday or contract-specific implementations; one needs auxiliary inputs and one is malformed. Most of the 176 Pine files were inventoried, not independently backtested. Historical option chains, contract/underlying joins, delivery and assignment, margin and liquidity rules need further implementation. Other paper chapters require fundamentals, events, macro releases or nonmarket cash flows. The full paper catalogue is not fully tested.</p></details>
<p class="foot">Costs shown are cumulative fees + slippage (+ modeled borrow when relevant), in dollars, on each row's stated initial capital. Cost totals omit opportunity costs. Drawdown is the largest decline of daily marked equity; intraday losses may be worse. Period passes count chronological blocks under base costs, not independent holdouts. Original histories and settings were already explored. <a href="REPORT.md">Read the detailed report</a> · <a href="scorecard.json">Machine-readable scorecard</a> · <a href="../CATALOGUE.md">Full catalogue status</a> · <a href="tradingview_decision.json">Pine shortlist decision</a></p></main>
<script>const rows=__DATA__;const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const pct=n=>n===null?'—':(n*100).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})+'%';let sort='origin',direction=1;const search=document.getElementById('search'),origin=document.getElementById('origin'),status=document.getElementById('status');for(const x of [...new Set(rows.map(r=>r.origin))]){const op=document.createElement('option');op.value=x;op.textContent=x;origin.appendChild(op)}function draw(){const term=search.value.toLowerCase();const data=rows.filter(r=>(!origin.value||r.origin===origin.value)&&(!term||JSON.stringify(r).toLowerCase().includes(term))&&(!status.value||(status.value==='lead'&&r.full_pass)||(status.value==='quarantine'&&r.status.startsWith('Quarantined'))||(status.value==='fail'&&r.status.startsWith('Failed')))).sort((a,b)=>{const x=a[sort],y=b[sort];return direction*(typeof x==='number'&&typeof y==='number'?x-y:String(x??'').localeCompare(String(y??'')))});document.getElementById('count').textContent=data.length+' of '+rows.length+' comparisons';document.getElementById('body').innerHTML=data.map(r=>`<tr><td><strong>${esc(r.strategy)}</strong><small>${esc(r.origin)}</small></td><td>${esc(r.market)}<small>${esc(r.period)}</small></td><td class="num">${pct(r.return)}</td><td class="num">${pct(r.benchmark_return)}</td><td class="num">${pct(r.drawdown)}</td><td class="num">${pct(r.benchmark_drawdown)}</td><td>${esc(r.block_pass)}</td><td class="num">${r.costs===null?'See audit':'$'+r.costs.toLocaleString(undefined,{maximumFractionDigits:0})}<small>on $${r.capital.toLocaleString()}</small></td><td><span class="${r.full_pass?'pass':'fail'}">${esc(r.status)}</span><small>${esc(r.notes)}</small><a href="${esc(r.evidence)}">Data</a></td></tr>`).join('')}[search,origin,status].forEach(x=>x.addEventListener('input',draw));document.querySelectorAll('th[data-key]').forEach(x=>x.onclick=()=>{direction=sort===x.dataset.key?-direction:1;sort=x.dataset.key;draw()});draw();</script></html>'''
    (OUT/'scorecard.html').write_text(page.replace('__DATA__',payload))
    # Detailed companion in the established repository report format.
    best_spy=max((x for x in rows if x['origin']=='Pattern_FindR' and x['market']=='SPY'),key=lambda x:x['return'])
    best_btc=max((x for x in rows if x['origin']=='Pattern_FindR' and x['market']=='BTC-USD'),key=lambda x:x['return'])
    report=f'''# Paper 151 + Pattern_FindR + trading_view_mcp_quant

2 October 2026. **No new strategy meets the requested return-and-drawdown standard.** The new work extends the earlier research; it does not erase unsuccessful tests or turn an already explored sample into an untouched holdout. [Interactive scorecard](scorecard.html) contains {len(rows)} full-period strategy/market comparisons, including quarantined diagnostics. [Original checkpoint](../history/2026-09-29/CURRENT_RESULTS.md) remains unchanged.

## What changed

- External drive reads now work. All 2,972 selected compressed daily files passed their recorded SHA-256 checks. The staged panel contains 109,738 individual-contract rows from 2 January 2017 through 28 June 2026 for ES, NQ, GC, CL and ZN. Hash verification confirms file identity, not market-data correctness.
- Eight additional paper recipes completed 128 strategy cases plus 48 benchmark cases. CalendarSlope2 stopped in all 16 attempted cases because a required preselected deferred contract had no observed price. Failed attempts are retained and excluded from profit rankings.
- Pattern_FindR: 93 saved configurations inventoried, 92 valid JSON. Tested 39 supported daily configurations: 21 SPY and 18 BTC. The 344 runs include repeated periods/stresses and 32 benchmark runs. Several configurations are duplicates; this is not 39 independent ideas. No configuration passed the full-sample comparison.
- trading_view_mcp_quant: 176 Pine files inventoried. Retested saved C7, corrected C8 C5 and C11 over the cached June 2021–June 2026 ES history at four slippage levels each. These 12 runs are quarantined diagnostics.
- Tested 21 nearby settings for the original gold, TLT and BTC leads across four periods and two cost levels: 168 additional diagnostic runs, not a new optimization.
- 87 unit/regression checks passed; two existing dependency warnings remain. All 39 Pattern_FindR prefix recalculations matched; C11 also matched 107,840 prefix bars with no signal mismatch. Signal causality on a supplied series does not cure a noncausal construction of that series.

In total, the paper track now has 42 declared completed recipes and 1,101 repeated runs including benchmarks and stresses. Tested adaptations cover 27 of the 175 catalogue rows. This update adds 700 completed runs when Pattern_FindR, quant diagnostics and sensitivity cases are also counted. Those categories should not be confused with independent profitable strategies.

## Pattern_FindR: preserve ideas, correct the experiment

The saved strategies combine oscillators (signals of price position or momentum), their velocity (how quickly those signals move), acceleration, reversals and protective exits. We retained eligible saved parameters without another search. The source contains both a research composite and an older production composite, so the chosen formula is explicitly recorded in [the protocol](velocity_protocol.json). These are causal adaptations, not exact reproductions of the old percentage claims.

The retest enters at the next opening price. Stops that are crossed by a gap fill at the worse opening price. Trailing stops depend only on previously known highs, and an ambiguous daily bar touching both stop and target takes the stop first. Account value is marked every day. SPY pays 5 basis points commission plus 5 slippage per side; BTC pays 10 plus 5. A basis point is 0.01%. Dividends and the declared cash-interest convention apply to both strategy and benchmark. Public Yahoo snapshots remain frozen through 23 September 2026.

The best full-period SPY configuration, `{best_spy['strategy']}`, returned {PCT(best_spy['return'])}, versus {PCT(best_spy['benchmark_return'])} for buy-and-hold. Its drawdown was {PCT(best_spy['drawdown'])}: less risk, but much less return. The best BTC configuration returned {PCT(best_btc['return'])}, versus {PCT(best_btc['benchmark_return'])} for holding BTC. A large positive number by itself does not establish outperformance.

Older engines allow same-close actions, sometimes omit costs, use closed-trade equity for drawdown, use the current bar's high before testing its low for trailing stops, and contain whole-sample initialization/statistics in some paths. Those mechanisms can make results optimistic. The audit does not prove how much of each earlier report arose from each issue. Data feed, period and formula differences also matter.

52 configurations still need separate intraday or actual-contract implementations. One additional daily rule requires unsupported auxiliary inputs; one configuration is malformed. These are explicitly recorded in [the 93-entry status ledger](velocity_status.json). They are not silently counted as tested. Whole-series wavelet variants and source formulas that revise their past require causal redesign before qualification.

## Quant project: why the attractive short sample does not settle it

The saved C8 family search used 200,000 trials, and the BTC range-reversal search used 64,000. Repeated searches on the same market history increase the chance of finding an accidental fit. Those saved headline reports are evidence to investigate, not a fresh holdout.

The following are frozen-parameter transports to the five-year cached ES feed, with $50,000 starting capital and one ES contract. They use $5 round-trip commission and one tick of adverse slippage on market fills in the baseline. The borrowed parity engine remains a same-close fill model. No actual roll or dated margin ledger is present.

| Candidate | Diagnostic final return | Largest daily-equity decline | Account ever below zero at a 15-minute close? |
|---|---:|---:|---|
'''
    for r in quant:
        if r['case']=='1_ticks_per_market_side':report+=f"| {r['strategy']} | {PCT(r['total_return'])} | {PCT(r['max_drawdown'])} | {'Yes' if r['ever_insolvent'] else 'No'} |\n"
    report+='''
C11 reproduced $142,472.50 net profit, but about 87% daily-equity drawdown; two ticks of slippage per market side caused the account to cross zero. C7 and C8 crossed zero even at the one-tick baseline (C8 recovered before the daily snapshot). Their positive final balances therefore do not represent an investable recovery. Simulations that continue after insolvency are shown only to diagnose the model.

The existing continuous-data builder chooses the front contract using that same day's final volume. That selection was unavailable earlier intraday. Back-adjusted prices also are not the prices paid to roll actual contracts. Prefix-invariant indicators do not repair either issue. All three candidates are excluded from promotion pending actual-contract intraday reconstruction. [C11 audit](c11_retest.json), [C7/C8 audit](quant_family_results.json), [source inventory](quant_pine_inventory.json).

Other Pine files, including BTC 15-minute reversal, are inventoried but not independently retested in this batch. Cached TradingView BTC 15-minute exports were found, including a 2023–2026 history; faithful higher-timeframe signal alignment, short financing and execution remain separate implementation work. The 60,137-minute BTCUSDT archive is only about six weeks of observations, and is a different venue. It must not be silently substituted for the original BITSTAMP feed.

## Actual-contract futures: what was tried from the paper

- §10.4 trend: sign of trailing return, a smooth trend score, and a demeaned version. Exposure is scaled by prior volatility.
- §10.3 mean reversion: buy relative losers and sell relative winners, with plain, volatility-scaled and variance-scaled allocations.
- §9.1 roll yield: rank gold and crude oil by the near/deferred price ratio, buying the higher and selling the lower.
- §9.5 skewness: buy the commodity with lower return skewness and sell the higher. Only two commodities are available in this declared universe; this does not reproduce a broad quintile portfolio.
- §10.2 calendar structure: a near/deferred spread with an explicitly adapted slope signal was implemented, but missing preselected deferred prices prevented any complete test.

Prices are actual, unadjusted contracts. Selection for a day uses the preceding completed UTC day's volume, excludes delivery months within seven days (45 for crude oil, which expires before its delivery month), and charges both legs of a roll. The early roll cutoff is a conservative research rule, not a reconstructed historical last-trade calendar. The first traded price in the next UTC bucket is a modeled opening fill; its exact timestamp, order size and simultaneous executable bid/ask are not established.

The $1m reference account targets 95% gross notional, uses integer contracts and no leverage. Opening exposure is reduced when it exceeds 99% of equity. Commission is $2.50 per contract side, plus one tick baseline and three/five-tick stress. Cash earns zero for both strategy and benchmark. Passive benchmarks start equally allocated within the applicable universe, hold contract quantities through rolls and obey the same opening gross cap; they are rolled futures exposures, not literal perpetual holdings or SPY.

Lookbacks use common UTC daily buckets, including short Sunday buckets: 252 bars is not precisely 252 exchange sessions or twelve months. The five-bar contrarian formation is therefore an explicit daily-bucket adaptation. Monthly/weekly decisions use known closed buckets. Missing prices are not forward-filled.

| Recipe | Net return | Passive comparison | Drawdown | Passive drawdown |
|---|---:|---:|---:|---:|
'''
    for r in fr:report+=f"| {r['strategy']} | {PCT(r['return'])} | {PCT(r['benchmark_return'])} | {PCT(r['drawdown'])} | {PCT(r['benchmark_drawdown'])} |\n"
    report+='''
All eight fail the full-period return requirement. These results do not establish that all possible implementations of the paper's ideas fail. Contract specifications were checked against CME's [equity-index contract reference](https://www.cmegroup.com/content/dam/cmegroup/education/modules/files/EQ240_EQ_for_AIT.pdf), [gold product guide](https://www.cmegroup.com/education/courses/event-contracts-underlying-markets/product-gold), [Treasury futures delivery guide](https://www.cmegroup.com/content/dam/cmegroup/trading/interest-rates/files/us-treasury-futures-delivery-process.pdf) and [crude-oil contract specifications](https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html). [Protocol](futures_protocol.json), [all cases](futures_scored.json), [failures](futures_errors.json).

## Nearby settings and market regimes

Each cell is the number of nearby settings passing both return and drawdown tests at base costs. We did not replace the frozen rules with whichever variant won this diagnostic.

| Family | Full history | 2017–2019 | 2020–2022 | 2023–latest |
|---|---:|---:|---:|---:|
'''
    for family in ['GoldDiversification','TLT_MA','BTC_Momentum']:
        report+='| '+family+' | '+' | '.join('/'.join(map(str,counts[(family,b)])) for b in ['full','2017_2019','2020_2022','2023_latest'])+' |\n'
    report+='''
The gold allocations and TLT rules are less sensitive to nearby parameters in some periods, but all tested neighbors fail 2017–2019. BTC is particularly sensitive to the lookback, and none of its neighbors pass the latest block. This is evidence of regime dependence, not robust dominance. [All 168 diagnostics](lead_sensitivity.json).

## Frozen prospective observation

Four separate $10,000 simulated accounts are frozen: monthly 80/20 SPY/GLD, TLT MA50/200, TLT MA20/50/200, and BTC positive 252-day momentum. US decisions use completed daily bars and the next exchange open. BTC uses the most recent completed UTC day and the next strictly future UTC midnight, deliberately a two-bar lag that permits a daily evening recorder. That BTC execution variant must be compared with its own future benchmark, not represented as identical to the original one-bar historical rule.

The recorder is scheduled daily at 17:15 America/Chicago. It appends timestamped, hash-chained events and saves source snapshots. Orders are recorded before the proposed fill. Missed decisions are never invented later; existing positions and already committed orders are accounted when data arrives. A pending BTC order can remain unmarked until its full daily bar closes while the next future order is queued. Only completed-bar equity is reported. New stock splits fail closed until a versioned share-basis reconciliation is supplied.

US costs are 5+5 basis points per side; BTC 10+5. Both strategy and benchmark receive cash distributions, hold those distributions as cash, and earn zero cash interest. Fractional units are permitted. Thus the forward benchmark's dividend policy differs from the original historical automatic reinvestment model. The comparison is internally matched within the forward test.

Version 1 was superseded during validation, before any simulated order, to fix queued BTC decisions and detect a single missed session. Its source, freeze and original event remain preserved. Version 2's exact freeze timestamp and hashes are in [freeze.json](../prospective/freeze.json). At this checkpoint there are zero prospective orders or fills, so no forward return can yet be claimed. Collect twelve months through at least October 2027, with descriptive monthly reviews and no automatic retuning. Scheduled runs depend on this local environment and fresh data being available.

## TradingView and remaining scope

The MCP health check succeeded on daily TLT. The existing ten private scripts are retained: six portfolio-target indicators and four native strategies, previously compiler/source checked and reconciled. No newly tested rule merits replacing them under the requested gate. These are provisional visualization tools, not ten proven winners. [Selection record](tradingview_decision.json) and [Pine documentation](https://www.tradingview.com/pine-script-docs/).

The drive also contains quote and definition files. A bounded first-20,000-row sample read BBO, definition and statistics schemas successfully; the sampled definitions include calls and puts, and the BBO metadata includes option-like symbols. This does not establish a complete, synchronized, executable option-chain history. Only the selected daily OHLC files were fully hash-verified. Option lifecycle, underlying joins, expiry/delivery, dated margin and quote-size rules still require implementation. The generic derivatives accounting module's 19 synthetic checks remain separate from the daily futures historical ledger.

The entire 361-page paper was read earlier; its whole catalogue has not been backtested. 58 option entries, 68 other data/engine entries, one incomplete calendar-spread entry and 19 nonstandard cash-flow topics remain outside completed historical coverage. Two unlawful-activity descriptions remain excluded. No live trade or external message was sent, and no source project was modified.

## Reproduce

Run from the TensorTrade root using `.venv/bin/python`. The extension scripts are `cross_project_inventory.py`, `stage_contract_archive.py`, `velocity_retest.py`, `retest_c11.py`, `retest_quant_family.py`, `lead_diagnostics.py`, `futures_extension.py` and `make_extension_report.py`. Historical scripts use frozen inputs; do not casually replace them with newly revised data. Source/protocol hashes live in each manifest. The prospective recorder is separate and refuses changed frozen source code.

The scorecard uses the individual studies' matched benchmark periods and economic conventions. Do not average or sort raw returns across incompatible markets, account sizes or holding intervals to select a winner.
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps({'scorecard_rows':len(rows),'paper_recipes':42,'paper_runs':925+len(f),'new_runs':manifest['new_completed_runs_including_diagnostics_and_benchmarks'],'qualified':0}))

if __name__=='__main__':main()
