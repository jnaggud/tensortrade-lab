"""Dated SEC filing adapters for paper earnings and value signals on AAPL.

Fixed single surviving company: an adapter test, not cross-sectional replication.
"""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import json,sys
import numpy as np
import pandas as pd
from full_catalogue_options import save,sha
from run_study import read_frame,panel_from
from tensortrade_lab.portfolio import Ledger,performance

HERE=Path(__file__).resolve().parent;OUT=HERE/'data_expansion'

def dated_facts(facts,submissions,tag,unit):
    f=pd.DataFrame(facts['facts']['us-gaap'][tag]['units'][unit])
    f=f[f.form.isin(['10-K','10-Q','10-K/A','10-Q/A'])].copy()
    f['end']=pd.to_datetime(f.end,utc=True)
    f['filed']=pd.to_datetime(f.filed,utc=True)
    f=f.sort_values(['filed','accn','end'],kind='stable')
    acc={r['accessionNumber']:r['acceptanceDateTime'] for r in submissions}
    f['accepted_at']=pd.to_datetime(f.accn.map(acc),utc=True,errors='coerce')
    if f.accepted_at.isna().any():raise ValueError('Missing SEC accession acceptance time')
    # Filing day is date-only. Defer until next New York midnight and at least one
    # minute after recorded acceptance, whichever is later. Never assume pre-open.
    f['available_at']=[max(pd.Timestamp(d.date()+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC'),a+pd.Timedelta(minutes=1)) for d,a in zip(f.filed,f.accepted_at)]
    return f

def first_quarters(f):
    f=f.copy();f['start']=pd.to_datetime(f.start,utc=True)
    f=f[(f.end-f.start).dt.days.between(70,110)]
    # Separate revised comparative facts from original disclosures.
    f=f.sort_values(['available_at','accn']).drop_duplicates(['start','end'],keep='first')
    f['quarter']=(f.end-pd.Timedelta(days=7)).dt.tz_localize(None).dt.to_period('Q')
    if f.quarter.duplicated().any():raise ValueError('Ambiguous fiscal-quarter facts')
    return f.sort_values('quarter')

def sue_at(q,splits,at):
    known=q[q.available_at<=at].set_index('quarter')
    if known.empty:return None,'no known quarter'
    latest=known.index.max()
    if at-known.loc[latest,'available_at']>pd.Timedelta(days=180):return None,'stale earnings'
    need=pd.period_range(latest-11,latest,freq='Q')
    if not need.isin(known.index).all():return None,'missing consecutive quarters'
    use=known.loc[need]
    if (use.end<pd.Timestamp('2015-01-01',tz='UTC')).any():return None,'outside split-verified warmup'
    eps=[]
    for row in use.itertuples():
        factor=float(splits.loc[(splits.timestamp>row.filed)&(splits.timestamp<=at),'split'].prod())
        eps.append(row.val/factor)
    eps=np.asarray(eps);unexpected=eps[4:]-eps[:-4];sd=unexpected.std(ddof=1)
    if sd<=1e-12:return None,'zero earnings variation'
    return float(unexpected[-1]/sd),None

def main():
    facts=json.loads((OUT/'sec_aapl_facts.json').read_text())
    recent=json.loads((OUT/'sec_aapl_submissions.json').read_text())['filings']['recent']
    older=json.loads((OUT/'sec_aapl_submissions_001.json').read_text())
    submissions=pd.concat([pd.DataFrame(recent),pd.DataFrame(older)],ignore_index=True).to_dict('records')
    q=first_quarters(dated_facts(facts,submissions,'EarningsPerShareDiluted','USD/shares'))
    book=dated_facts(facts,submissions,'StockholdersEquity','USD')
    shares=dated_facts(facts,submissions,'CommonStockSharesOutstanding','shares')
    f,meta=read_frame('data/yahoo/AAPL_1d.parquet');p=panel_from({'AAPL':f})
    splits=f.loc[f['split']>0,['timestamp','split']]
    proto=OUT/'filing_protocol.json'
    if not proto.exists():save(proto,{'registered_at':datetime.now(timezone.utc).isoformat(),
      'sections':['3.2','3.3'],'asset':'AAPL, one fixed surviving stock. No cross-sectional top/bottom decile claim.',
      'earnings':'First filed, explicitly tagged quarterly diluted EPS, not latest restated value. Eight consecutive unexpected-earnings differences vs four quarters earlier, sample standard deviation. Adjust historical EPS only for splits already effective at decision, based on original filing share basis. Missing consecutive quarters means CASH; do not manufacture Q4 by subtracting weighted annual EPS.',
      'value':'Latest available dated stockholders equity divided by latest available common shares, adjusted for splits since share date, times contemporaneous raw-equivalent price. Long if positive book/market exceeds median of preceding monthly observations (at least12); otherwise cash. Monthly updates. Time-series adapter, not cross-sectional paper replication.',
      'availability':'Later of next New York midnight after filed date and acceptance timestamp+1minute. New signal at completed close; execution next open. Fundamentals older than180days unavailable.',
      'execution':'Long/cash,monthlyrebalance,$10k,5bpfee+5bpslip,sameLedgerasbenchmark,dividendsandprior-knowncashyield. Additional doubled-cost,two-bar-delay,no-interest stresses.',
      'evaluation':'2018-Sep2026,2018-2020,2021-2023,2024-latest. Annual reports cease directly tagging some Q4 EPS from2021: preserve missing data and cash outcomes. This is not evidence for an entire stock universe.',
      'source_sha256':{n:sha(OUT/n) for n in ['sec_aapl_facts.json','sec_aapl_submissions.json','sec_aapl_submissions_001.json']},'prices_sha256':meta['sha256']})
    decision=[];targets={'EarningsSUE':np.full(len(f),np.nan),'BookToMarket':np.full(len(f),np.nan)};ratios=[]
    for j in range(len(f)):
        # First observed session of month: only its completed close drives next open.
        if j and f.timestamp.iloc[j].month==f.timestamp.iloc[j-1].month:continue
        at=p.bar_end[j];sue,reason=sue_at(q,splits,at)
        targets['EarningsSUE'][j]=float(sue>0) if sue is not None else 0.
        bk=book[book.available_at<=at].sort_values(['end','available_at']).tail(1)
        sh=shares[shares.available_at<=at].sort_values(['end','available_at']).tail(1)
        bm=None;value_reason=None
        if bk.empty or sh.empty:value_reason='missing dated equity/shares'
        elif any(at-r.available_at.iloc[0]>pd.Timedelta(days=180) for r in [bk,sh]):value_reason='stale equity/shares'
        else:
            # Yahoo prices are split adjusted. Undo only the stock split scale,
            # and forward-adjust the as-filed share count to this decision date.
            future_factor=float(splits.loc[splits.timestamp>at,'split'].prod())
            known_factor=float(splits.loc[(splits.timestamp>sh.filed.iloc[0])&(splits.timestamp<=at),'split'].prod())
            marketcap=sh.val.iloc[0]*known_factor*p.close[j,0]*future_factor
            if marketcap>0:bm=float(bk.val.iloc[0]/marketcap)
        threshold=float(np.median(ratios)) if len(ratios)>=12 else None
        targets['BookToMarket'][j]=float(bm>0 and threshold is not None and bm>threshold) if bm is not None else 0.
        decision.append({'decision_at':str(at),'sue':sue,'earnings_missing_reason':reason,'book_to_market':bm,'past_monthly_median':threshold,'value_missing_reason':value_reason,'earnings_target':targets['EarningsSUE'][j],'value_target':targets['BookToMarket'][j]})
        if bm is not None:ratios.append(bm)
    save(OUT/'filing_decisions.json',decision)
    q.assign(quarter=q.quarter.astype(str)).to_parquet(OUT/'aapl_first_filed_quarters.parquet',index=False)
    rows=[]
    blocks=[('full','2018-01-01','2026-09-24'),('2018_2020','2018-01-01','2021-01-01'),('2021_2023','2021-01-01','2024-01-01'),('2024_latest','2024-01-01','2026-09-24')]
    for block,begin,finish in blocks:
        ix=np.flatnonzero((p.timestamp>=pd.Timestamp(begin,tz='UTC'))&(p.timestamp<pd.Timestamp(finish,tz='UTC')));a,b=int(ix[0]),int(ix[-1])
        for case,cost,delay,nointerest in [('base',1,1,False),('double_cost',2,1,False),('extra_bar_delay',1,2,False),('no_interest',1,1,True)]:
            pp=replace(p,cash_rate=np.zeros(len(f))) if nointerest else p
            benchmark=Ledger(pp,a,b,commission=.0005*cost,slippage=.0005*cost)
            for t in ix:benchmark.step(np.ones(1) if t==a else None,drip=True)
            bm=performance(benchmark)
            for name,target in targets.items():
                led=Ledger(pp,a,b,commission=.0005*cost,slippage=.0005*cost)
                for t in ix:
                    j=t-delay;w=None
                    if np.isfinite(target[j]):w=np.array([target[j]])
                    elif t==a:
                        known=np.flatnonzero(np.isfinite(target[:j+1]));w=np.array([target[known[-1]] if len(known) else 0.])
                    led.step(w)
                m=performance(led);rows.append({'strategy':name,'section':'3.2' if name=='EarningsSUE' else '3.3','block':block,'case':case,**m,'benchmark_return':bm['total_return'],'benchmark_drawdown':bm['max_drawdown'],'pass':bool(m['total_return']>bm['total_return'] and m['max_drawdown']<=bm['max_drawdown'])})
                if block=='full':
                    pd.DataFrame(led.history).to_parquet(OUT/(name+'_'+case+'.parquet'),index=False);save(OUT/(name+'_'+case+'_fills.json'),led.fills)
    save(OUT/'filing_results.json',rows)
    save(OUT/'filing_manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'candidate_cases':len(rows),'quarterly_eps_records':len(q),'protocol_sha256':sha(proto),'code_sha256':sha(__file__),'qualified':0,'missing_eps_consecutive_quarters_after_2021':'No proxy Q4 EPS fabricated; consult decisions for cash months.'})
    print('filings complete',len(rows),flush=True)

if __name__=='__main__':main()
