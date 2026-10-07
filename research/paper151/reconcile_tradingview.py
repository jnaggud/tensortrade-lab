"""Saved TLT native reports: empirical emulator audit, separate from rankings."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,math
from decimal import Decimal,ROUND_HALF_UP
import numpy as np
import pandas as pd
from expanded_study import load_jobs,prepare,single_target
from run_study import panel_from
HERE=Path(__file__).parent;OUT=HERE/'tradingview/reconciliation'

def emulate(rule,f):
    p=panel_from({'TLT':f},True,252);d,_=prepare(f,p,models=False)
    start=int(np.flatnonzero(p.timestamp>=pd.Timestamp('2017-01-01',tz='UTC'))[0])
    qty=0;cash=10000.;fills=[];history=[];held=0;rejected=[]
    for t in range(start,len(f)):
        held=held+1 if qty else 0
        w=single_target(rule,t-1,held,qty>0,f,d);price=float(Decimal(str(f.open.iloc[t])).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP));date=str(p.timestamp[t].date())
        if not qty and w:
            proposed=math.floor(cash*.998/(float(f.close.iloc[t-1])*1.001))
            if proposed*price>cash:
                rejected.append(date)
            else:
                qty=proposed;cash-=qty*price*1.001
                fills.append({'date':date,'role':'entry','units':qty,'price':price})
                if cash<0:
                    cover=min(qty,math.ceil(-4*cash/price));cash+=cover*price*.999;qty-=cover
                    fills.append({'date':date,'role':'margin','units':-cover,'price':price})
        elif qty and not w:
            cash+=qty*price*.999;fills.append({'date':date,'role':'exit','units':-qty,'price':price});qty=0
        history.append({'timestamp':str(p.timestamp[t]),'equity':cash+qty*float(f.close.iloc[t])})
    return fills,history,rejected,start

def native_events(trades,cutoff):
    groups=defaultdict(float)
    for t in trades:
        for leg,sgn in [('e',1),('x',-1)]:
            x=t[leg]
            if leg=='x' and not x.get('c'):continue # Open-position mark is not an execution.
            date=str(pd.Timestamp(x['tm'],unit='ms',tz='UTC').date())
            if date>cutoff:continue
            role='entry' if leg=='e' else ('margin' if x['c']=='Margin call' else 'exit')
            groups[(date,role,x['p'])]+=sgn*t['q']
    order={'entry':0,'margin':1,'exit':2}
    return [{'date':d,'role':role,'units':q,'price':p} for (d,role,p),q in sorted(groups.items(),key=lambda kv:(kv[0][0],order[kv[0][1]],kv[0][2]))]

def compare(fills,expected):
    dates=len(fills)==len(expected) and all((a['date'],a['role'])==(b['date'],b['role']) for a,b in zip(fills,expected))
    return {'events_match':dates,'quantities_match':dates and all(a['units']==b['units'] for a,b in zip(fills,expected)),
            'max_fill_price_difference':max(abs(a['price']-b['price']) for a,b in zip(fills,expected)) if dates else None,
            'local_fill_count':len(fills),'native_fill_count':len(expected)}

def replay(f,expected,start):
    cash=10000.;q=0;cursor=0;closed_peak=10000.;dd_dollars=0.;dd_percent=0.;daily=[]
    def risk(value):return max(0,closed_peak-value)
    for t in range(start,len(f)):
        date=str(f.timestamp.iloc[t].date())
        while cursor<len(expected) and expected[cursor]['date']==date:
            x=expected[cursor];units=x['units'];price=x['price']
            if q:
                loss=risk(cash+q*price);dd_dollars=max(dd_dollars,loss);dd_percent=max(dd_percent,loss/closed_peak)
            if not q:closed_peak=max(closed_peak,cash)
            cash-=units*price+abs(units*price)*.001;q+=units;cursor+=1
            if not q:
                loss=risk(cash);dd_dollars=max(dd_dollars,loss);dd_percent=max(dd_percent,loss/closed_peak)
                closed_peak=max(closed_peak,cash)
        if q:
            loss=risk(cash+q*float(f.low.iloc[t]));dd_dollars=max(dd_dollars,loss);dd_percent=max(dd_percent,loss/closed_peak)
        daily.append({'timestamp':str(f.timestamp.iloc[t]),'equity':cash+q*float(f.close.iloc[t])})
    assert cursor==len(expected),'Native events missing local price bars'
    eq=np.array([x['equity'] for x in daily]);eq_with_initial=np.r_[10000.,eq]
    return {'return':eq[-1]/10000-1,'daily_peak_drawdown':float((1-eq_with_initial/np.maximum.accumulate(eq_with_initial)).max()),'closed_peak_intrabar_drawdown':dd_percent,'closed_peak_intrabar_dollars':dd_dollars,'terminal_units':q},daily

def main():
    path=OUT/'TLT_native_reports.json';reports=json.loads(path.read_text());summary={}
    jobs,_=load_jobs();f=next(j[1]['TLT'] for j in jobs if j[0]=='TLT')
    prices=pd.read_json(OUT/'TLT_tradingview_prices.json');prices['timestamp']=pd.to_datetime(prices.time,unit='s',utc=True)
    ft=f.merge(prices[['timestamp','open','high','low','close','volume']],on='timestamp',suffixes=('_yahoo',''),validate='one_to_one')
    assert len(ft)==len(f),'Need common full chart history for same-feed audit'
    for name,data in reports.items():
        rule=name.split(' ')[2];tv=data['report'];expected=native_events(tv['trades'],str(f.timestamp.iloc[-1].date()))
        variants={};fills_by_feed={}
        for feed,fr in [('yahoo',f),('tradingview',ft)]:
            fills,h,rejected,start=emulate(rule,fr);fills_by_feed[feed]=fills
            variants[feed]={**compare(fills,expected),'independent_price_only_return':h[-1]['equity']/10000-1,'rejected_entry_dates':rejected}
        m,h=replay(ft,expected,start);perf=tv['performance']
        report={'scope':f'TLT / {rule}, 2017-01-03 through {f.timestamp.iloc[-1].date()}; not other symbols, intervals or configurations.',
          'method':'Independent completed-close signals; integer 99.8% order sizing with commission reserve at the signal close. Reject if next-open notional exceeds cash. Reproduce immediate fee-induced 100%-margin liquidation. TLT fills round to the cent. Per-side fee 0.10%; no dividends/interest credited, no slippage. This is an empirical emulator diagnostic, not a change to the economic ranking.',
          'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_fills':expected,'local_fills_by_feed':fills_by_feed,
          'comparison':variants,'native_fill_replay':m,'native_report_return_through_latest_chart':perf['all']['netProfit']/10000,
          'native_report_max_drawdown_through_latest_chart':perf['maxStrategyDrawDownPercent'],'native_report_drawdown_dollars_through_latest_chart':perf['maxStrategyDrawDown'],
          'native_margin_exits_in_matched_sample':sum(x['role']=='margin' for x in expected),
          'drawdown_note':'Replay uses daily marked equity for the research risk measure; TradingView uses closed-trade equity peaks for displayed intrabar drawdown. Its latest-chart net profit can have a different endpoint/open position. Quotes and quantities are reconciled separately from the full local dividend/cash-interest model.'}
        (OUT/f'TLT_{rule}_audit.json').write_text(json.dumps(report,indent=2));pd.DataFrame(h).to_parquet(OUT/f'TLT_{rule}_native_fills_daily_equity.parquet',index=False)
        summary[rule]={k:v for k,v in report.items() if k not in ('native_fills','local_fills_by_feed')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
