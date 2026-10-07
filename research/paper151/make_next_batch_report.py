"""Combine completed intraday screens with the preserved earlier checkpoint."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, html, json, shutil
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'next_batch'
def read(p):return json.loads((HERE/p).read_text())
def save(p,x):(HERE/p).write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n')
def pct(x):return '—' if x is None else f'{100*x:+.2f}%'
def risk(x):return '—' if x is None else f'{100*x:.2f}%'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    btc=read('next_batch/btc_results.json');fut=read('next_batch/futures_intraday_results.json')
    daily=read('next_batch/daily_futures_results.json');dm=read('next_batch/daily_futures_manifest.json')
    statuses=read('next_batch/futures_intraday_status.json');fm=read('next_batch/futures_intraday_manifest.json')
    bm=read('next_batch/btc_manifest.json');im=read('next_batch/intraday_manifest.json')
    options=read('next_batch/options_pilot_results.json');forward=read('prospective/latest.json')
    assert len(statuses)==fm['candidates'] and len(fut)==fm['runs']
    archive=HERE/'history/2026-10-02-first-pass';archive.mkdir(parents=True,exist_ok=True)
    for name in ['CURRENT_RESULTS.md','STATE.md','current_manifest.json','DATA_INVENTORY.md','DERIVATIVES_READINESS.md','catalogue.json','CATALOGUE.md']:
        if not (archive/name).exists():shutil.copy2(HERE/name,archive/name)
    previous=json.loads((archive/'current_manifest.json').read_text())
    new=btc+fut+daily;full=[r for r in new if r['block']=='full' and r['case']=='base']
    grouped={r['strategy']:[z for z in new if z['strategy']==r['strategy']] for r in full}
    all_cases=[n for n,z in grouped.items() if all(r['pass'] for r in z)]
    tested_pf={r['strategy'] for r in full if r['origin']=='Pattern_FindR'}
    old_status=read('extension/velocity_status.json')
    fut_status={r['strategy']:r for r in statuses}
    current_status=[]
    for r in old_status:
        name=r['strategy'];z=dict(r)
        if name in tested_pf:z.update(status='historically_tested_adaptation',evidence='next_batch/'+('btc_results.json' if 'BTC' in name else ('daily_futures_results.json' if name in {x['strategy'] for x in daily} else 'futures_intraday_results.json')))
        elif name in fut_status:z.update(fut_status[name])
        current_status.append(z)
    save('next_batch/patternfindr_status.json',current_status)
    counts=Counter(r['status'] for r in current_status)
    save('next_batch/qualification.json',{'full_sample_passes':[r['strategy'] for r in full if r['pass']],
         'all_period_and_stress_passes':all_cases,'new_execution_qualified':[],
         'decision':'Retain existing ten exploratory Pine scripts. No new candidate promoted as a proven winner.',
         'why':'Retrospective histories have been explored; futures fills are print-based models with sparse buckets, not synchronous quotes; no prospective edge established. Options pilot is only five days.'})

    # Keep pilot evidence distinct from completed historical paper recipes.
    cat=read('catalogue.json')
    for r in cat:
        if r['section']=='2.2':
            r.update(status='historical_quote_pilot_only',recipes=['ES_covered_call_five_day_pilot'],
              notes='June 1-5 2026 observed-quote ES futures plus short European weekly call pilot. Four base/stress runs across this and a cash-secured short-put building-block variant; not a long historical validation. Cash-secured put is not the covered-put construction in §2.3.')
    save('catalogue.json',cat)
    coverage=(archive/'CATALOGUE.md').read_text()
    coverage=coverage.replace('| 2.2 / 18 | Covered call | needs_historical_option_quotes | Historical contract-level option quotes, strikes, expiries, underlying prices, multipliers, exercise/assignment and collateral. Daily closes alone do not prove executable multi-leg prices. |',
      '| 2.2 / 18 | Covered call | historical_quote_pilot_only | Five-day ES futures/European weekly-call bid/ask pilot, June 1–5 2026. Closed before expiry. Long-history, liquidity, lifecycle and prospective qualification remain incomplete. See next_batch/REPORT.md. |')
    (HERE/'CATALOGUE.md').write_text(coverage)

    # Machine-readable and searchable combined scorecard. Earlier links are rebased.
    score=read('extension/scorecard.json')
    for r in score:
        p=r.get('evidence','')
        if p and not p.startswith('../'):r['evidence']='../extension/'+p
    for r in full:
        peers=grouped[r['strategy']]
        score.append({'origin':r['origin']+' / actual futures' if 'actual' in r['market'] else r['origin']+' / BTC intraday',
          'strategy':r['strategy'],'market':r['market'],'period':r['start'][:10]+' → '+r['end'][:10],
          'return':r['total_return'],'benchmark_return':r['benchmark_return'],'drawdown':r['max_drawdown'],
          'benchmark_drawdown':r['benchmark_drawdown'],'bar_close_drawdown':r['bar_close_drawdown'],
          'full_pass':r['pass'],'block_pass':f"{sum(z['pass'] for z in peers)}/{len(peers)} cases",
          'costs':r['fees']+r['slippage']+r.get('borrow',0),'capital':r['initial_capital'],
          'status':'Historical screen only; not qualified' if r['pass'] else 'Failed historical comparison',
          'notes':'Next observed open; prior-known stops. Saved-rule adaptation, no new parameter search. '+('Actual raw contracts; both roll legs; early contract selection and sparse observations limit execution claims.' if 'actual' in r['market'] else 'Venue-specific BTC bars; assumed fees/slippage and unverified historical short availability.'),
          'evidence':('daily_futures_results.json' if '/ daily' in r['market'] else 'futures_intraday_results.json') if 'actual' in r['market'] else 'btc_results.json'})
    save('next_batch/scorecard.json',score)
    trs=[]
    for r in score:
        vals=[r['origin'],r['strategy'],r['market'],r['period'],pct(r['return']),pct(r['benchmark_return']),risk(r['drawdown']),risk(r['benchmark_drawdown']),r['block_pass'],r['status']]
        trs.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>')
    (OUT/'scorecard.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trading research results</title><style>
    body{font:16px system-ui;background:#f5f7fa;color:#172a3a;margin:0;padding:28px}main{max-width:1550px;margin:auto}h1{font-size:30px}p{max-width:950px;line-height:1.55}input{font:inherit;padding:12px;width:min(600px,90%);margin:16px 0}.scroll{overflow:auto;max-height:70vh;background:white;border:1px solid #ccd6df}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd;vertical-align:top}th{position:sticky;top:0;background:#183e51;color:white}td:nth-child(2){min-width:190px;overflow-wrap:anywhere}td:nth-child(n+5):nth-child(-n+9){white-space:nowrap}a{color:#005f85}.note{background:#fff0d6;padding:14px;border-left:4px solid #ba7400}
    </style><main><h1>Trading research results</h1><p>No strategy is established as a reliable winner after costs with no worse maximum drawdown. These are historical screens, including earlier quarantined diagnostics. Passing one full sample is not qualification.</p>
    <p class="note">Options are a separate five-day pilot and are excluded from this ranking. Drawdowns below use daily marked equity; intraday losses and unobserved-price intervals can be worse. Results cover different periods, instruments and starting capital; compare each row with its own benchmark. A dash means no trustworthy matched comparison was established.</p>
    <p><a href="REPORT.md">Detailed findings</a> · <a href="scorecard.json">All metrics and evidence links</a> · <a href="qualification.json">Promotion decision</a></p><input id="q" aria-label="Filter strategies" placeholder="Filter by strategy, market, source or status"><span id="count"></span><div class="scroll"><table><thead><tr>'''+''.join('<th>'+x+'</th>' for x in ['Source','Strategy','Market','Period','Return','Buy and hold','Drawdown','Benchmark drawdown','Period/stress passes','Status'])+'</tr></thead><tbody>'+''.join(trs)+'''</tbody></table></div></main><script>
    const q=document.getElementById('q'), rows=[...document.querySelectorAll('tbody tr')],count=document.getElementById('count');function filter(){let n=0;for(const r of rows){const show=r.textContent.toLowerCase().includes(q.value.toLowerCase());r.hidden=!show;n+=show;}count.textContent=` ${n} of ${rows.length} comparisons`;}q.addEventListener('input',filter);filter();</script></html>''')

    bfull=[r for r in full if 'BTC' in r['market']];ffull=[r for r in full if '/ 15m' in r['market'] and 'actual' in r['market']]
    dfull=[r for r in full if '/ daily' in r['market']]
    blocked=[r for r in statuses if not r['status'].startswith('historically')]
    def table(rows):
        return '\n'.join(['| Saved rule | Return after costs | Matched buy and hold | Drawdown | Benchmark drawdown |','|---|---:|---:|---:|---:|']+
          [f"| {r['strategy']} | {pct(r['total_return'])} | {pct(r['benchmark_return'])} | {risk(r['max_drawdown'])} | {risk(r['benchmark_drawdown'])} |" for r in rows])
    option_table='\n'.join(['| Five-day base pilot | Return after costs | Passive future | Maximum sampled loss |','|---|---:|---:|---:|']+
      [f"| {r['strategy'].replace('_',' ')} | {pct(r['total_return'])} | {pct(r['benchmark_return'])} | {risk(r['max_sampled_drawdown'])} |" for r in options if r['case']=='base'])
    raw_audits={root:read('next_batch/'+root+'_feed_audit.json') for root in ['ES','GC','CL']}
    audit_table='\n'.join(['| Futures market | Observed 15 minute bars | Rolls | Missing selected-contract print buckets |','|---|---:|---:|---:|']+
      [f"| {root} | {z['bars']:,} | {z['rolls']} | {z['unprinted_selected_buckets']} |" for root,z in raw_audits.items()])
    pending=sum(len(x['pending_orders']) for x in forward['state'].values())
    qfull=[r for r in ffull if r['origin']=='trading_view_mcp_quant']
    best=sorted([r for r in ffull if r['origin']=='Pattern_FindR'],key=lambda r:r['total_return']-r['benchmark_return'],reverse=True)[:6]
    prefix_count=sum(r['passed'] for r in read('next_batch/btc_causality.json'))+sum(r['passed'] for r in read('next_batch/futures_intraday_prefix.json'))+sum(r['passed'] for r in read('next_batch/daily_futures_prefix.json'))
    batch_runs=len(new)+bm['benchmark_runs']+fm['benchmark_runs']+dm['benchmark_runs']+len(options)+8
    report=f'''# Intraday strategy and options pilot results

This update tests the remaining saved Bitcoin intraday rules, reconstructs actual-contract futures tests, and runs the first historical options quote pilot. **No new execution-qualified strategy is ready for promotion.** The existing ten TradingView scripts remain exploratory. Returns below are model results, not live account performance.

## What the new tests establish

- Bitcoin: eight saved-rule adaptations, {len(btc)} period/cost/delay cases. None passes its full historical comparison after costs.
- Futures: {fm['candidates']} configurations/families reviewed; {fm['completed']} complete {len(fut)} period/cost/delay cases. {sum(r['pass'] for r in ffull)} pass the full-sample base comparison; {len(all_cases)} pass every tested period and stress. Passing remains insufficient for executable or prospective qualification.
- Daily futures: five additional saved Pattern_FindR rules complete {len(daily)} cases on the already verified daily archive, January 2018–June 2026. These use the same actual-contract accounting model.
- Options: two constructions complete four five-day base/stress runs. This checks the data joins and accounting; it does not establish a profitable long-run strategy.
- Forward observation: {pending} allocation decisions were recorded before Monday October 5's opening. Two say to remain in cash. No simulated fill or forward performance has occurred yet.
- Verification: 146 tests pass, including 11 new timing/accounting tests; {prefix_count} additional signal-prefix comparisons pass. One existing action-space warning remains. A prefix test asks whether adding future data changes past signals; it is not proof against every possible bias.

An independent replay of all 51 new full-case trade ledgers matches the reported final cash to within one millionth of a dollar. This separately checks the saved fills rather than reusing the backtest's account calculation. Drawdown means the largest fall from an earlier account-value peak; sampling limits still matter even when the arithmetic balances.

There are {batch_runs} completed calculations in this batch, including repeated benchmark executions, eight zero-cost diagnostics and four pilot cases. There are {len(full)} new full-sample candidate comparisons, not {batch_runs} independent strategies. The [combined scorecard](scorecard.html) now contains {len(score)} comparisons including the earlier checkpoint.

## Bitcoin results in plain language

The seven Pattern_FindR adaptations trade thousands of times. At the declared costs, they lose nearly all their simulated starting capital. The quant reversal loses about 80%, while holding Bitcoin gains about 30% over the same March 2024–May 2026 period. These results do not reproduce the optimistic labels in some saved filenames.

{table(bfull)}

![Costed Bitcoin results and separate zero-cost diagnostics](btc_cost_comparison.png)

A separately registered zero-cost diagnostic removes fees, slippage and short borrowing. Two Pattern_FindR rules then return about 39% and 47%; the quant reversal returns about 18%. Several other rules still lose money before costs. This distinguishes transaction-cost drag from weak signals. Zero-cost results are not investable results and were not used to choose replacement settings; their trading paths also change slightly because costs affect stops and account sizes.

Technical assumptions: 82,433 gap-free BITSTAMP 15-minute bars, December 31 2023–May 7 2026, cached in trading_view_mcp_quant. Scoring starts March 1 2024 after warmup. $10,000 accounts; 95% entry exposure, 100% buy-and-hold allocation; 10 basis points fee plus 5 basis points slippage each side; double-cost and extra-bar-delay cases. Saved quant shorts use a collateral/margin model and 5% annual borrowing, stressed to 10%; historical borrow availability is unverified. Signal decisions use completed bars, higher-timeframe values are delayed, fills occur at the next open, and ambiguous stop/target bars resolve adversely. The RSC regime filter is an explicit past-window clustering redesign, not an exact replay of earlier implementation behavior.

## Actual-contract futures results

The new model follows specific S&P 500, gold and oil futures contracts. It pays to sell the old contract and buy the new one. A price difference between contracts cannot become a fictitious profit. A separate linked price series is used only to calculate signals; actual raw contract prices determine account value.

{table(qfull+best)}

This table shows all three quant families plus the six Pattern_FindR results with the smallest return shortfall (or largest excess) versus their own passive benchmark. This descriptive ordering is not a new strategy-selection rule. All completed rows and stress results remain in [futures_intraday_results.json](futures_intraday_results.json).

C7, C8 and C11 are negative in these new full-period tests. Contract data, entry timing, stop handling and account sizing differ from their earlier diagnostics. These changes were made together, so this comparison does not isolate how much of the earlier result came from any single assumption.

The five additional daily rules have their own longer, matched January 2018–June 2026 sample:

{table(dfull)}

Their twelve cases each use full history, three subperiods, three-tick costs and an extra day's delay. See [daily_futures_results.json](daily_futures_results.json) and its separately registered [protocol](daily_futures_protocol.json). Daily open/high/low fills remain a model; these are not precise intraday execution records.

{audit_table}

The archive provided {im['files']:,} one-minute DBN files, all matching previously recorded SHA-256 values, with {im['rows']:,} selected-contract 15-minute rows after aggregation. Dates span January 2021–June 2026; scoring starts June 2021. No required source date is missing. Some selected contracts have no prints in individual buckets: those buckets create no artificial bars or fills, and the next observed price recognizes the intervening move. Reported drawdowns measure observed marked equity and can miss unobserved extremes.

Each strategy and matched rolled passive comparison starts with $1 million, targets at most 95% gross notional, uses integer contracts, charges $2.50 per contract side plus one tick (three under stress), and earns zero collateral yield. Opening exposure reductions maintain the modeled gross cap. Positions may breach it between observations. Prior completed UTC-day volume and early delivery-month cutoffs choose contracts. This can select a thinly traded deferred contract. Old and new opening prints may occur in different minutes within a bucket: executable spreads, size, latency, margin schedules and a dated roll calendar remain unverified. These assumptions are intentionally disclosed rather than described as a live-ready system. Results also differ from the earlier $50,000 one-contract continuous-series diagnostics.

Fill-ledger timestamps are bucket labels, not exact execution times. A modeled roll completes only where both contracts have prints in the same bucket. That overlap guard does not establish event-by-event order execution or remove the risk of one leg filling first; quote-level validation remains necessary.

## Configurations that cannot yet receive a trustworthy result

'''+ '\n'.join(f"- {r['strategy']}: {r.get('reason',r['status'])}." for r in blocked)+f'''

The original minimum-duration regime filter fails a concrete timing check: adding one future bar changes three earlier regime labels. Its source comment saying there is no look-ahead is contradicted by this counterexample. The wavelet implementation reconstructs the entire series using later observations; that is a source-code finding, not a numerical PyWavelets test. See [saved_timing_audit.json](saved_timing_audit.json). Causal rewrites would be new strategies and require their own frozen protocols; these source configurations were not silently simplified into claimed results.

Across the complete Pattern_FindR inventory, {counts['historically_tested_adaptation']} of 93 saved configurations now have explicit historical adaptations. {counts['needs_separate_intraday_or_contract_test']} still need other timeframe/instrument work; {len(blocked)} are blocked in this intraday batch, one earlier auxiliary-input configuration remains blocked, and one file is malformed. Most of the 176 quant Pine files still lack an independent retest. These are saved configurations, including duplicates, not independent discoveries.

## First options quote pilot

A covered call owns one ES future and sells one call. A cash-secured put sells one put while reserving enough cash for its full strike value. Both receive a premium but retain downside exposure. The put construction is not the paper's §2.3 covered put, which combines a short underlying with a short put. The covered-call pilot is a futures-based adaptation of §2.2.

{option_table}

![Five-day options pilot account returns](options_pilot_comparison.png)

Both lose money during this week, but less than the passive future. The same direction holds under one extra adverse tick per leg. This is not evidence of a persistent advantage: there is only one entry/exit episode per construction, substantial idle cash in each $1m account, and different payoff exposure in the put strategy.

The pilot uses June 1–5 2026 quotes for ESM6, EW2M6 C7760 and EW2M6 P7450, selected from definitions known before the June 1 decision. It buys at the ask and sells at the bid, requires quoted size of at least one, matches leg snapshots within one minute and charges $2.50 per side. These Friday weekly options are European style and the pilot closes before their June 12 expiry, so no early assignment or expiration delivery is simulated. [CME contract guidance](https://www.cmegroup.com/trading/equity-index/weekly-eom-options-faq.html).

The BBO interval end uses `ts_recv`; `ts_event` is the last trade's time and cannot serve as a quote timestamp. Interval quotes can carry forward unchanged values, so true quote-update age and latency remain unverified. [Databento schema](https://databento.com/docs/schemas-and-data-formats/bbo). Five full BBO files were decoded and freshly hashed; their original archive entries lacked reference hashes, so these are new integrity baselines, not matches to old hashes. Definition source hashes did match. There are 7,197 selected quote rows. One put-mark timestamp lacks a valid synchronized sized quote and is recorded as missing; it is not imputed. Marks cover only observed daytime windows, not complete overnight or intraday risk. The result column therefore says maximum sampled loss.

## Forward observation and TradingView

The four earlier frozen candidates remain 80/20 SPY/gold, TLT 50/200 averages, TLT 20/50/200 averages and Bitcoin 252-day positive momentum. At {forward['recorded_at']}, the first three had recorded decisions for October 5 at 13:30 UTC (8:30 a.m. Chicago): 80/20 SPY/gold, cash, cash. Bitcoin awaited its first completed UTC daily bar after the freeze. There were no simulated fills and no live orders.

The existing daily recorder remains scheduled at 5:15 p.m. America/Chicago. Its frozen source hashes and append-only events are preserved; future elapsed time cannot be backtested into existence. The current TradingView MCP read succeeded on BATS:TLT daily. The ten existing scripts remain four native strategies plus six portfolio-target indicators. This update does not claim new Pine compilation or execution parity for the new intraday adapters.

## What remains before any new promotion

The paper has still been read in full, but its entire catalogue has not been tested. The earlier 42 completed paper recipes and 27 adapted catalogue entries remain unchanged; the covered-call row now separately records a short quote pilot. The next evidence needed is broader options history and lifecycle coverage, executable futures quote/size tests with dated contract and margin rules, causal redesigns for the blocked saved rules, and genuinely new observations from the frozen forward simulation. Remaining non-price chapters also require point-in-time fundamentals, events, borrow, macro vintages or specialized cash flows.

Reproduction files: [BTC protocol](btc_protocol.json), [futures protocol](futures_intraday_protocol.json), [options protocol](options_pilot_protocol.json), [validation](verification.json), [promotion decision](qualification.json). Source projects and the external archive were read-only. No parameter search, live order, external notification or public Pine publication was performed in this batch.
'''
    (OUT/'REPORT.md').write_text(report)

    manifest=previous|{'asof':datetime.now(timezone.utc).isoformat(),
      'catalogue_status':dict(Counter(r['status'] for r in cat)),
      'patternfindr_configs_tested_total':counts['historically_tested_adaptation'],
      'patternfindr_new_intraday_configs_tested':len(tested_pf)-dm['configs'],'patternfindr_new_daily_futures_configs_tested':dm['configs'],'patternfindr_status':dict(counts),
      'patternfindr_runs':sum(r.get('origin')=='PatternFindr_adaptation' for r in read('extension/velocity_results.json'))+sum(r['origin']=='Pattern_FindR' for r in new),
      'patternfindr_run_count_convention':'Candidate cases only; excludes benchmark and cash executions. The archived first-pass count of 344 included 32 benchmark/cash cases.',
      'quant_new_actual_contract_families_tested':sum(r['origin']=='trading_view_mcp_quant' for r in ffull),
      'quant_btc_intraday_families_tested':1,'new_intraday_candidate_runs':len(btc)+len(fut),'new_daily_futures_candidate_runs':len(daily),
      'intraday_futures_source_files_hashed':im['files'],'intraday_contract_rows':im['rows'],
      'new_full_sample_passes':sum(r['pass'] for r in full),'new_all_period_stress_passes':len(all_cases),
      'new_batch_completed_calculations_including_benchmarks_and_pilots':batch_runs,
      'new_completed_runs_including_diagnostics_and_benchmarks':previous['new_completed_runs_including_diagnostics_and_benchmarks']+batch_runs,
      'combined_full_sample_comparisons':len(score),'unit_and_regression_checks_passed':146,
      'new_signal_prefix_checks_passed':prefix_count,'signal_prefix_checks_passed':previous['signal_prefix_checks_passed']+prefix_count,
      'historical_option_strategy_runs':len(options),'long_history_option_strategy_runs':0,'option_pilot_constructions':2,'option_pilot_runs':len(options),
      'forward_pending_allocation_decisions_at_report':pending,'forward_completed_trades_at_report':0,
      'shortlist_decision':'Keep existing ten exploratory scripts. No new execution-qualified promotion.',
      'previous_checkpoint':'history/2026-10-02-first-pass/current_manifest.json',
      'remaining_work':[f"{counts['needs_separate_intraday_or_contract_test']} saved Pattern_FindR configurations need other timeframe/instrument tests; three causal/state-machine blockers, one auxiliary-input blocker, one malformed file.",
       'Futures quotes/size, sparse-price risk, dated roll calendars/margins; broader options quote history and expiry/delivery lifecycle.',
       'Most quant Pine files and most paper catalogue entries still need independent data and implementation.',
       'Frozen prospective performance requires new future observations.']}
    manifest['sources']=dict(previous['sources'])
    for p in ['btc_manifest.json','futures_intraday_manifest.json','daily_futures_manifest.json','options_pilot_manifest.json','scorecard.json','qualification.json']:
        manifest['sources']['next_batch/'+p]=digest(OUT/p)
    save('current_manifest.json',manifest)
    (HERE/'CURRENT_RESULTS.md').write_text(f'''# Trading strategy research current results

**The paper and both local strategy projects have been reviewed, and another {len(full)} candidate comparisons are complete. We still have no established strategy that reliably beats buy and hold after costs with no worse drawdown.** Updated {manifest['asof']}.

The full 361-page paper was read earlier; [the plain-language review](PAPER_REVIEW.md) remains available. It is a catalogue of ideas, not proof that every strategy makes money. Its contents were treated as evidence, not instructions.

[Open the combined scorecard](next_batch/scorecard.html) · [Read the new detailed findings](next_batch/REPORT.md) · [See every paper entry](CATALOGUE.md)

| Workstream | Completed | Result and limit |
|---|---|---|
| Paper | 42 completed recipes, 1,101 repeated runs; 27 of 175 catalogue rows adapted, plus one separately labeled short options pilot | Eleven original full-sample leads; zero broadly qualified families. Most catalogue entries remain untested. |
| Pattern_FindR | {counts['historically_tested_adaptation']} of 93 saved configurations now have declared historical adaptations | The new intraday results use stricter timing and costs. Three saved wavelet/regime configurations are blocked by causal/state-machine issues. |
| Bitcoin intraday | Seven Pattern_FindR configurations plus one quant reversal; {len(btc)} cases, March 2024–May 2026 | No full-sample pass. Quant reversal about −80%, versus buy-and-hold about +30%; frequent Pattern_FindR rules lose almost all starting capital after costs. |
| Futures intraday | {fm['completed']} completed adaptations including C7/C8/C11; {len(fut)} cases, June 2021–June 2026 | {sum(r['pass'] for r in ffull)} full-sample base passes; {len(all_cases)} all-period/stress passes. Actual-contract P&L replaces adjusted-price fills. Executable quotes, margins and sparse-price risk remain unverified. |
| More daily futures rules | Five saved Pattern_FindR configurations; {len(daily)} cases, January 2018–June 2026 | {sum(r['pass'] for r in dfull)} full-sample passes. Actual-contract modeled prices, not executable-quote validation. |
| Options pilot | Covered call and cash-secured put, June 1–5 2026; four base/stress cases | Both lose less than the passive future during this week. One episode per rule and incomplete risk sampling cannot establish an edge. |
| Forward simulation | Four frozen candidates; {pending} precommitted allocation decisions for October 5, including two cash decisions | No fills or forward performance yet. Daily recorder remains active at 5:15 p.m. Chicago. |
| TradingView | MCP read succeeds; existing ten private scripts retained | Four native strategies and six allocation indicators remain exploratory. No new scripts promoted as proven winners. |

The data include frozen Yahoo daily prices from the earlier study, a venue-specific BITSTAMP cache for the new Bitcoin tests, and your Expansion drive's Databento futures/option archive. The new extraction checks 1,706 minute files against recorded hashes and produces 398,833 actual-contract 15-minute rows. Five option BBO files were fully decoded and freshly hashed; their original manifest had no comparison hashes. No market-data sources were silently blended.

There are {len(score)} full-period comparisons in the combined scorecard. The whole local suite passes 146 tests, including 11 new timing/accounting checks, and {prefix_count} additional signal-prefix checks pass. These verify mechanics, not future profitability. All new rules are disclosed adaptations; earlier headline returns are not treated as validated results.

Still incomplete: {counts['needs_separate_intraday_or_contract_test']} Pattern_FindR configurations need other timeframe/instrument work, three new causal/state-machine blockers remain, plus one earlier auxiliary-input rule and one malformed file. Most of the 176 quant Pine files and most paper entries are not independently tested. Long options histories, expiry/delivery, executable liquidity, dated margin and specialized fundamental/event data remain missing. The [detailed report](next_batch/REPORT.md) explains the exact limits.

Earlier work is preserved in [the first October 2 checkpoint](history/2026-10-02-first-pass/CURRENT_RESULTS.md) and [the September checkpoint](history/2026-09-29/CURRENT_RESULTS.md). No live order or public publication was made. The evidence supports continuing research, not promising a winner.
''')
    (HERE/'DATA_INVENTORY.md').write_text(f'''# Data found and fitness for this research

Updated {manifest['asof']}. Source projects and the Expansion archive were read-only. Earlier detailed inventories are preserved in history/2026-09-29/ and history/2026-10-02-first-pass/.

| Data | Verified use | Remaining limit |
|---|---|---|
| Frozen Yahoo daily ETFs | Earlier single-market tests through September 23 2026, portfolios through September 21; prices, distributions and stated cash inputs | Revised public research prices; matched study dates differ. |
| Frozen Yahoo daily Bitcoin | 4,390 daily rows, September 2014–September 23 2026 | Aggregate prices, not executable venue quotes; not blended with intraday BTC. |
| BITSTAMP BTC cache from quant project | 82,433 gap-free 15-minute bars, December 31 2023–May 7 2026; eight candidates, 96 cases | Fixed fee/slippage model; historical short availability unverified. |
| Expansion actual daily futures | 2,972 files matched saved hashes; 109,738 ES/NQ/GC/CL/ZN rows, 2017–June 2026 | UTC daily OHLC, including partial Sunday buckets; no executable-quote proof. Eight earlier paper recipes and five new PF daily rules tested. |
| Expansion actual intraday futures | 1,706 minute files matched saved hashes; 398,833 selected-contract 15-minute rows, January 2021–June 2026 | Sparse prints, early-roll approximation, no synchronous roll bid/ask or dated margins. Per-root details in next_batch feed audits. |
| Five-day ES options pilot | Prior-known definitions and ESM6/call/put joins; five BBO files fully decoded and freshly hashed; 7,197 selected rows, June 1–5 2026 | Original BBO hashes absent, so new baselines only; quote-update age/latency unverified, daytime marks only, one missing put mark. Not full historical chain validation. |
| Fama-French factors | Current and historical annual archives with conservative prior-release availability | Archived version can be stale; revised variants remain diagnostic. |
| Pattern_FindR configurations | 93 inventoried, 92 valid, sanitized snapshots; {counts['historically_tested_adaptation']} now historically adapted | Two other timeframe/instrument cases, three wavelet/regime cases, one auxiliary-input rule and one malformed file remain; duplicates are not independent ideas. |
| Quant Pine catalogue | 176 files inventoried; C7/C8/C11 have new actual-contract intraday adapters and one BTC reversal was tested | Most files remain untested; new adapters have no asserted native Pine parity. Earlier continuous-price diagnostics remain quarantined. |
| Prospective Yahoo snapshots | Hash-preserved snapshots and append-only log; first three decisions recorded before October 5 execution | No fills or prospective performance yet. Needs future data and a running local environment. |

Full-period minute/quote coverage is not inferred from archive job metadata. Only the dated extracts above have the described validation. Existing Pattern_FindR option-zone/OI summaries and backfilled sentiment still do not constitute executable option chains or historically available sentiment. Fundamental universes with delistings, borrow/event histories, release-vintage macro and specialized cash-flow datasets remain unverified.

Evidence: [minute extraction](next_batch/intraday_manifest.json), [BTC protocol](next_batch/btc_protocol.json), [options data manifest](next_batch/options_pilot_data_manifest.json), [updated saved-rule status](next_batch/patternfindr_status.json), [detailed results](next_batch/REPORT.md). Costs, capital, exact periods and signal adaptations belong to each experiment's protocol. None of the retrospective samples is an untouched holdout.
''')
    (HERE/'DERIVATIVES_READINESS.md').write_text(f'''# Futures and options research readiness

**Actual-contract futures tests and a five-day options quote pilot are complete. Neither establishes an execution-ready winner.** Updated {manifest['asof']}.

Eight earlier paper futures recipes completed 176 daily strategy/benchmark cases; none passed the full sample. The calendar-spread experiment still has 16 failed-closed cases because required deferred prices were absent. The new batch completes {fm['completed']} intraday adapters in {len(fut)} candidate cases, including C7/C8/C11, plus five saved daily rules in 60 cases. These use actual raw contract prices for P&L, a separate forward-linked signal series, both roll legs, integer positions, $1m starting cash and a 95% target gross cap. Daily and intraday bar fills are modeled, not a quote-level execution claim.

The new feed replaces the quant cache's same-day final-volume selection and adjusted execution prices. Selection now uses prior completed UTC-day information; trades use the next observed opening print. Trailing stops use only prior highs; ambiguous bars resolve stop first. Sparse selected-contract buckets create no invented prices or fills, and subsequent actual prices recognize intervening moves. This leaves unobserved intraday risk. Historical roll calendars, bid/ask size, quote timing, market impact, margins and collateral yields remain unverified. The original continuous-series diagnostic results are retained separately.

The options pilot joins definitions known before its decision with ESM6 and two June 12 European weekly options. It evaluates a fully funded future plus short call, and a strike-reserved short put, from June 1 to June 5 2026. Both pay observed bid/ask spreads and commissions, require visible size and synchronized snapshots, and close before expiry. All four base/stress cases finish with losses smaller than the passive future during that week. One episode per construction cannot validate profitability. Daytime snapshots and one missing put valuation also cannot establish full maximum drawdown.

CME's weekly-option exercise terms support choosing European contracts and closing before expiry; they do not validate our execution assumptions. [CME reference](https://www.cmegroup.com/trading/equity-index/weekly-eom-options-faq.html). BBO interval timing uses `ts_recv`, while `ts_event` is the last trade time; unchanged quotes can carry forward. [Databento reference](https://databento.com/docs/schemas-and-data-formats/bbo).

Still required before promotion: broader contract/quote coverage across market conditions, quote-update age and size checks, dated financing/margin/roll rules, explicit expiry/exercise/delivery handling, and prospective confirmation. American early assignment and physical delivery into futures are not implemented by this pilot. The earlier generic derivatives module and its synthetic tests remain useful accounting components, not historical validation on their own.

See [new detailed results](next_batch/REPORT.md), [options protocol](next_batch/options_pilot_protocol.json), [original module capability report](history/2026-09-29/DERIVATIVES_READINESS.md), and [preserved earlier futures checkpoint](history/2026-10-02-first-pass/DERIVATIVES_READINESS.md).
''')
    (HERE/'STATE.md').write_text(f'''# Research checkpoint {manifest['asof']}

Latest user authorized all previously proposed next steps: BTC intraday, actual-contract futures, options pilot, frozen prospective recording and conditional TradingView promotion. Read CURRENT_RESULTS.md and next_batch/REPORT.md; current_manifest.json has counts. Earlier checkpoint preserved at history/2026-10-02-first-pass/.

This batch: eight BTC adaptations / 96 cases; {fm['completed']} actual-contract intraday adaptations / {len(fut)} cases out of {fm['candidates']} reviewed (38 PF plus C7/C8/C11); five more PF daily actual-futures configs / {len(daily)} cases; four five-day options cases; eight zero-cost BTC diagnostics. {sum(r['pass'] for r in ffull)} intraday futures full-base passes and {sum(r['pass'] for r in dfull)} daily passes; {len(all_cases)} all-period/stress passes; zero execution-qualified new promotions. PF total historical adaptations {counts['historically_tested_adaptation']}/93, remaining separate timeframe/instrument {counts['needs_separate_intraday_or_contract_test']}, three blocked wavelet/regime, one prior auxiliary blocker, one malformed. 146 tests pass, one pre-existing warning. Additional prefix passes {prefix_count}. New scorecard {len(score)} full-sample comparisons. Paper's prior 42 completed recipes / 1101 cases / 27 adapted catalogue rows unchanged; §2.2 separately has historical_quote_pilot_only. Short put is a building-block variant, not §2.3 covered put.

Data: 1706 minute DBN files prior-hash matched; 398833 selected actual-contract 15m rows Jan2021–Jun2026. Sparse selected-contract buckets omit prices/fills and are reported in root feed audits. Forward multiplicative linking is signal-only; actual raw-price P&L, both roll legs, $1m/95% gross integer sizing, next observed open, prior stops, stop-first ambiguity. Actual first-print minutes can differ across roll legs; no quote/margin/dated expiry proof. BTC 82433 cached BITSTAMP 15m rows Dec2023–May2026; score Mar2024 onward. Costed rules fail, two gross PF variants beat BH only in zero-cost diagnostics. No retuning.

Options: European Friday weekly June12 expiry, ESM6/EW2M6 C7760/EW2M6 P7450, definitions known before June1 13:59; pilot June1 14:00–June5 19:55 UTC, closes before expiry. $1m with one fully funded contract. 7197 quote rows; observed bid/ask+size; ts_recv is quote interval clock, ts_event is last trade. One missing put mark, no imputation; only daytime windows. Four pilot cases pass within week but all lose money; no qualification. Five BBO source hashes newly computed because original manifest hashes null (not mismatches); definition hashes match. Files and protocols in next_batch. Do not turn five days into a long historical result.

Prospective freeze v2 unchanged 2026-10-02T19:23:59.611201Z. First post-close event {forward['recorded_at']} has three pending Oct5 13:30 UTC decisions: SPY/GLD80/20 and two TLT cash targets; BTC waiting first post-freeze completed UTC bar. Zero fills. Recorder research/paper151/prospective.py; daily heartbeat paper-151-prospective-strategy-recorder at17:15America/Chicago remains ACTIVE. Never backdate, overwrite frozen code or retune; automation appends local simulated decisions only. Existing ten Pine scripts unchanged; MCP read BATS:TLT daily succeeded. This turn does not claim new Pine compile/parity. Private chart layout omitted from the public source release.

Code: stage_intraday_contracts.py, intraday_common.py, retest_btc_intraday.py, btc_cost_diagnostic.py, futures_intraday_retest.py, retest_pf_daily_futures.py, prepare_options_pilot.py, options_pilot.py, make_next_batch_report.py. Original velocity_retest.py and prospective.py unchanged. Source projects/drive remain read-only; never import webhook-bearing modules or print secrets. Pure function snapshots and sanitized configs are safe. No live orders/messages/PRs. Use .venv/bin/python and PYTHONDONTWRITEBYTECODE=1; LOKY_MAX_CPU_COUNT=4. Earlier generators overwrite stale checkpoint counts; use make_next_batch_report.py for this checkpoint, and preserve frozen experiments before new changes.

Remaining: other {counts['needs_separate_intraday_or_contract_test']} PF configurations and blockers, most176quantPinefiles, executable futures quotes and margin calendars, broad options/lifecycle, missing calendar-spread deferred bars, most paper specialized data. New causal variants must receive separate protocols and labels, not replace source results silently. Await genuinely future observations; no winner promised.
''')
    print({'comparisons':len(score),'pf_tested':counts['historically_tested_adaptation'],'new_cases':len(new),'full_passes':sum(r['pass'] for r in full),'all_cases_passes':all_cases},flush=True)

if __name__=='__main__':main()
