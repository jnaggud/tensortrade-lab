"""Reproducible paper TOC inventory; no missing strategy silently disappears."""
from pathlib import Path
import json
import re
from collections import Counter

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CHAPTERS = {2:'Options',3:'Stocks',4:'ETFs',5:'Fixed income',6:'Indexes',7:'Volatility',8:'FX',9:'Commodities',10:'Futures',11:'Structured assets',12:'Convertibles',13:'Tax arbitrage',14:'Miscellaneous assets',15:'Distressed assets',16:'Real estate',17:'Cash',18:'Cryptocurrencies',19:'Global macro',20:'Infrastructure'}
REQUIRES = {
 2:'Historical contract-level option quotes, strikes, expiries, underlying prices, multipliers, exercise/assignment and collateral. Daily closes alone do not prove executable multi-leg prices.',
 3:'Point-in-time universe, delistings, corporate actions and strategy-specific features; borrow/collateral for short legs.',
 4:'Aligned ETF OHLC, distributions and explicit allocation/rebalance rules; borrow for short legs.',
 5:'Bond-level cash flows, yields, durations, issuer/default histories and executable quotes; swaps/CDS where relevant.',
 6:'Synchronized constituents/index/futures/options quotes, financing and explicit settlement/rolls as applicable.',
 7:'Individual VIX futures/ETN histories or full options/variance-swap quotes, rolls, borrow, hedging costs and collateral.',
 8:'Synchronized FX bid/ask quotes, rates/forward points, rollover funding and currency conversions.',
 9:'Individual commodity contracts, term structures, rolls and point-in-time COT/physical fundamentals as applicable.',
 10:'Individual futures contracts, multipliers, tick sizes, margin, session calendar, rolls and financing; not back-adjusted prices as executable fills.',
 11:'Historical tranche/CDS/MBS quotes, collateral cash flows/defaults/correlations/prepayments and funding.',
 12:'Convertible contract terms, bond quotes, stock/borrow prices and credit/volatility inputs.',
 13:'Historical security cash flows plus investor-specific tax jurisdiction/treaty/eligibility; current legal review. Not an OHLC rule.',
 14:'Instrument-specific swaps/TIPS/weather/energy quotes, contracts, cash flows and hedged business exposure.',
 15:'Point-in-time bankruptcy/default/reorganization events, debt quotes and actual recoveries; corporate control/cost assumptions.',
 16:'Transaction/appraisal/rent/financing/renovation/property/geographic histories. Liquid REIT proxies change the strategy.',
 17:'Cash flows, collateral/default/funding/operating costs appropriate to the activity; most entries are not chart-trading rules.',
 18:'Venue-specific crypto OHLC/order execution; timestamped text and vocabulary/labels for sentiment.',
 19:'Release-vintage macro series, announcement timestamps and tradable assets; final revised macro series would leak future information.',
 20:'Project-level investment and operating cash flows, financing, maintenance and exit valuations; listed ETFs are only proxies.'}

# Concrete implementations associated with entries. All parameterized adaptations
# remain labeled adaptations even when the underlying economic idea is explicit.
RECIPES = {
 '3.1':['Momentum252','SectorMomentumSkip21'], '3.4':['SectorLowVol'],
 '3.9':['SectorContrarian'], '3.11':['SMA200'], '3.12':['MA50_200'],
 '3.13':['MA20_50_200'], '3.14':['PivotBounce'],
 '3.15':['Donchian55_20','ChannelBounce20'], '3.17':['KNN'],
 '3.20':['AlphaCombo'], '4.1':['SectorRotation'],
 '4.1.1':['SectorRotationMA'], '4.1.2':['SectorRotationDual'],
 '4.4':['IBS','SectorIBS'], '4.6':['MultiTrend','MultiTrendVol','MultiTrendVar'],
 '6.5':['Vol12'], '9.3':['GoldDiversification'],
 '10.4':['TrendVol12','Momentum252'], '18.2':['ANN']}
DETAILS = {
 '3.1':'Single-asset absolute momentum and fixed ETF cross-sectional long-only momentum; neither is a reproduced survivorship-free stock long/short universe.',
 '3.2':'Needs earnings values as released, publication timestamps and surprise history.',
 '3.3':'Needs point-in-time book values and stock universe, including delistings.',
 '3.4':'Long-only lowest-volatility ETF ranking; not the paper stock long/short factor.',
 '3.5':'Derived ES option-IV features found locally; stock option bid/ask panels and executable prices still require inspection.',
 '3.6':'Requires value data plus momentum, common universe, factor normalization and portfolio constraints.',
 '3.7':'Requires dated Fama-French factors and rolling 36-month regressions; no substitution with unlagged full-sample residuals.',
 '3.8':'Needs two independently tradable legs, causal spread fit, borrow and spread-stability checks.',
 '3.9':'ETF single-cluster long-only residual allocation; full dollar neutrality is a separate implementation.',
 '3.9.1':'Needs time-varying cluster membership and a verified long/short ledger.',
 '3.10':'Needs risk exposures, residual regression weights and a verified long/short ledger.',
 '3.14':'Prior-day pivot support bounce; exit at prior resistance or after five held bars. All fills at next open, not assumed at touched pivot.',
 '3.15':'Separate breakout and bounce operationalizations; paper presents channel methods rather than one fixed optimized setting.',
 '3.16':'Needs historical merger announcements, terms, deal failures and short borrow.',
 '3.17':'Causal rolling KNN; training-only scaling, known labels, fixed settings. Target adjusted to next-open execution.',
 '3.18':'Needs expected-return forecasts, covariance estimates, bounds and a verified constrained portfolio optimizer; algebra alone is not evidence of profit.',
 '3.19':'Requires executable quotes, queue/latency/adverse-selection model and inventory accounting; trade aggregates are not order-book fills.',
 '3.20':'Fixed equal-vote combination of three declared signals; no selection using future performance.',
 '4.2':'Requires Fama-French factor regressions and contemporaneously investable ETF universe.',
 '4.3':'Requires point-in-time actively managed fund universe and factor-model R-squared estimates.',
 '4.4':'Both a single-symbol IBS threshold adaptation and a cross-sectional long-only ETF ranking; paper also describes shorts.',
 '4.5':'Requires inverse/leveraged pair histories, historical borrow and tail/collateral modeling; daily decay is not free arbitrage.',
 '4.6':'Three momentum weighting formulas tested separately; local multiasset accounting required.',
 '6.5':'Historical-volatility forecast replaces the paper implied-volatility formulation; unlevered 12% target.',
 '8.1':'HP filtering must be rerun on historical prefixes; a full-sample two-sided filter is invalid for trading signals.',
 '9.2':'COT observations must be lagged until their public release, not their measurement date.',
 '9.3':'80/20 SPY/GLD monthly-rebalanced example, not a diversified futures commodity basket.',
 '10.4':'Spot/ETF absolute trend and volatility-scaled adaptation; continuous futures archive is not an executable roll ledger.',
 '18.2':'Fixed small annual-refit ANN on crypto, with known labels and training-only scaling; paper supplies an architecture discussion, not trained weights.',
 '18.3':'No historical timestamped sentiment corpus with point-in-time labels located yet.',
 '19.5':'Requires genuine announcement-release timestamps and executable intraday quotes.'}

def build():
    recipes={k:list(v) for k,v in RECIPES.items()};details=dict(DETAILS)
    signed_metrics=HERE/'signed_results/metrics.json'
    if signed_metrics.exists():
        extra={'3.1':'LSMomentum','3.4':'LSLowVol','3.8':'LSPair','3.9':'LSCluster','3.9.1':'LSMultiCluster','3.10':'LSWeightedResidual','4.4':'LSIBS'}
        for section,recipe in extra.items():recipes.setdefault(section,[]).append(recipe)
        details.update({
            '3.1':'Absolute, long-only cross-sectional and dollar-neutral ETF momentum adaptations; no survivorship-free stock-universe reproduction.',
            '3.4':'Long-only and dollar-neutral low-volatility ETF rankings, with assumed short availability/borrow costs.',
            '3.8':'Fixed XLP/XLY daily relative-return pair; trailing correlation filter, signed cash/share ledger and assumed borrow. No optimized pair search or cointegration claim.',
            '3.9':'Both long-only and dollar-neutral ETF one-cluster residual allocations tested; signed version pays short dividends and borrow.',
            '3.9.1':'Dollar-neutral mean reversion within three fixed ETF groups; declared broad groupings are not validated stock industry clusters.',
            '3.10':'Inverse-variance weighted residual holdings from intercept/prior-window SPY-beta regression; target dollar/beta neutrality verified. ETF adaptation with assumed borrow.',
            '4.4':'Single-symbol IBS and long-only/dollar-neutral cross-sectional ETF variants. Daily signed version includes borrowing and paid distributions.'})
    if (HERE/'factor_results/metrics.json').exists():
        recipes['3.7']=['ResidualMomentumLong','ResidualMomentumLS']
        recipes['4.2']=['AlphaRotation']
        details.update({'3.7':'36-month three-factor regression, final-12-month standardized residual momentum on nine ETFs. Three factor-vintage modes; only archived mode eligible, none qualified. Long-only and signed adaptations.',
                        '4.2':'36-month monthly factor intercept ranking of nine sector ETFs. Annual archived factors plus revised-data sensitivity tests; no broad qualification. Replaces the paper typical one-year daily/weekly fit.'})
    if (HERE/'extension/futures_results.json').exists():
        recipes['9.1']=['CommodityCarryRank2']
        recipes['9.5']=['CommoditySkew252Rank2']
        recipes['10.2']=['CalendarSlope2']
        recipes['10.3']=['FutContrarian5','FutContrarianVol5','FutContrarianVar5']
        recipes['10.4']+=['FutTrendSign252','FutTrendSmooth252','FutTrendNeutral252']
        details.update({
            '9.1':'Actual-contract GC/CL two-commodity ranking adaptation. Prior-day volume selects contracts; observed near/deferred ratios; monthly trades. Completed modeled daily-bar tests, no full-sample qualification. Two markets are not a wide commodity universe.',
            '9.5':'Actual-contract GC/CL two-commodity skewness ranking over 252 completed UTC daily buckets. Integer positions, actual rolls and costs. Completed historical model, no full-sample qualification.',
            '10.2':'CalendarSlope2 implemented but all 16 block/cost cases stopped for absent preselected deferred-contract prices. No historical return claimed. Slope signal is an adaptation, not the paper fundamental supply/demand forecast.',
            '10.3':'Three actual-contract five-market mean-reversion weighting adaptations; five UTC-bar return formation, weekly rebalancing. Daily close equity and both roll costs; no full-sample qualification.',
            '10.4':'Earlier ETF/spot tests plus three actual-contract five-market trend weighting rules, 252 UTC-bar momentum and 63-bar volatility. Long/short unlevered gross target; completed historical model, no full-sample qualification. Quote-size and dated margin execution still unverified.'})
    text='\n'.join((ROOT/f'tmp/pdfs/151-strategies/page-{n:03}.txt').read_text() for n in range(2,8))
    rows=[]
    for line in text.splitlines():
        m=re.match(r'^(\d+(?:\.\d+)+)\s+Strategy:\s*(.*?)\s*\.\s*\.\s*\..*?\s(\d+)\s*$',line)
        if m:
            rows.append(dict(section=m[1],title=m[2],printed_page=int(m[3]),toc_strategy_label=True))
    assert len(rows)==173, 'Unexpected source/TOC parsing: inspect before proceeding'
    rows.extend([dict(section='3.10',title='Mean-reversion – weighted regression',printed_page=49,toc_strategy_label=False),dict(section='20',title='Infrastructure',printed_page=123,toc_strategy_label=False)])
    rows.sort(key=lambda r:tuple(map(int,r['section'].split('.'))))
    assert len({r['section'] for r in rows})==175
    for row in rows:
        s=row['section']; ch=int(s.split('.')[0])
        row.update(chapter=CHAPTERS[ch],pdf_page=row['printed_page']+1,requirements=REQUIRES[ch],recipes=recipes.get(s,[]),status='needs_data_or_engine',pine='external accounting/data required',notes=details.get(s,''))
        if s in recipes:
            row.update(status='declared_local_adaptation',pine='native strategy feasible' if any(x in recipes[s] for x in ['SMA200','MA50_200','MA20_50_200','PivotBounce','Donchian55_20','ChannelBounce20','Momentum252','IBS','Vol12','TrendVol12','AlphaCombo','KNN']) else 'multiasset indicator or external model; not native portfolio Strategy Tester')
        if ch==2:
            row['status']='needs_historical_option_quotes'
            row['pine']='payoff visualization possible; multi-leg historical PnL requires external accounting'
        if ch in (13,16,20) or s in ['15.2','15.2.1','15.2.2','15.2.3','17.3','17.4','17.5']:
            row['status']='not_a_standard_ohlc_backtest'
        if s in ('17.2','17.6'):
            row.update(status='excluded_unlawful_activity',pine='not applicable',notes='Descriptive material in the source is not an instruction or a candidate for implementation.')
    results=[]
    for relative in ['expanded_results/metrics.json','signed_results/metrics.json','factor_results/metrics.json','extension/futures_results.json']:
        metrics=HERE/relative
        if metrics.exists():
            results.extend({**r,'market':r.get('market','Sectors'),'evidence':relative} for r in json.loads(metrics.read_text()))
    if results:
        tested={r['strategy'] for r in results}
        for row in rows:
            if row['recipes'] and set(row['recipes'])<=tested:
                row['status']='historically_tested_adaptation'
                row['evidence']=sorted({r['evidence'] for r in results if r['strategy'] in row['recipes']})
                row['market_recipe_tests']=len({(r['market'],r['strategy']) for r in results if r['strategy'] in row['recipes']})
    if (HERE/'extension/futures_errors.json').exists():
        next(r for r in rows if r['section']=='10.2').update(status='blocked_missing_deferred_contract_prices',evidence=['extension/futures_errors.json'])
    (HERE/'catalogue.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    header='# Complete strategy coverage ledger\n\n173 explicitly labeled entries plus §3.10 and §20 = 175 rows. Nested variants are retained; this is a source coverage count, not 175 independent alphas. Status describes actual completion. Missing-data entries are not backtested and are not ranked.\n\n'
    parts=[header]
    for ch,name in CHAPTERS.items():
        parts.append(f'## {ch}. {name}\n\n| Section / page | Strategy | Current disposition | Implementation / limitation |\n|---|---|---|---|\n')
        for r in rows:
            if int(r['section'].split('.')[0])!=ch:continue
            note=r['notes'] or r['requirements']
            if r['recipes']:note='Recipes: '+', '.join(r['recipes'])+'. '+note
            parts.append(f"| {r['section']} / {r['printed_page']} | {r['title']} | {r['status']} | {note} |\n")
        parts.append('\n')
    (HERE/'CATALOGUE.md').write_text(''.join(parts))
    print(json.dumps({'rows':len(rows),'status_counts':dict(Counter(r['status'] for r in rows))},indent=2))
    return rows

if __name__=='__main__':build()
