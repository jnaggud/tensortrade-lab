"""Fixed long/short ETF expansion; keeps the earlier experiment immutable."""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
from expanded_study import load_jobs, BLOCKS
from run_study import panel_from
from tensortrade_lab.portfolio import Ledger

OUT=HERE/'signed_results'
RULES=['LSMomentum','LSLowVol','LSIBS','LSCluster','LSMultiCluster','LSWeightedResidual','LSPair']
BENCHMARKS=['BuyHold','PassiveBasket','Cash']
GROUPS=((0,1,2,3),(5,6,7),(4,8))

class SignedLedger:
    """Signed cash/share ledger with explicit, simplified short collateral.

    Research convention, not broker margin emulation. A close breach can only
    liquidate at a later open. No future high/low is used to choose fills.
    """
    def __init__(self,p,start,end,fee=.0005,slip=.0005,borrow=.02,initial=10000.):
        if start<1 or end<=start or end>=len(p.timestamp):raise ValueError('Invalid interval')
        if min(fee,slip,borrow)<0:raise ValueError('Negative cost')
        self.p=p;self.start=start;self.end=end;self.i=start-1
        self.fee=fee;self.slip=slip;self.borrow_rate=borrow;self.initial=initial
        self.cash=initial;self.units=np.zeros(len(p.symbols))
        self.fees=self.slippage=self.borrow_paid=self.interest=self.long_dividends=self.short_dividends=self.turnover=0.
        self.pending_liquidation=False;self.margin_liquidations=0;self.bankrupt=False
        self.fills=[];self.history=[];self.record()

    def short_value(self,prices):return float(-np.minimum(self.units,0)@prices)
    def equity_at(self,prices):return float(self.cash+self.units@prices)
    def free_cash(self,prices):return self.cash-1.5*self.short_value(prices)
    def execute(self,delta,prices,reason):
        prices=np.asarray(prices)
        if not np.isfinite(prices).all() or np.any(prices<=0):raise ValueError('Bad prices')
        fill=prices*(1+self.slip*np.sign(delta))
        fees=np.abs(delta*fill)*self.fee
        self.cash-=float(delta@fill+fees.sum());self.units+=delta
        self.fees+=float(fees.sum());self.slippage+=float(np.abs(delta*prices).sum()*self.slip)
        self.turnover+=float(np.abs(delta*prices).sum())
        for k in np.flatnonzero(np.abs(delta)>1e-10):
            self.fills.append({'timestamp':str(self.p.bar_end[self.i] if reason=='terminal' else self.p.timestamp[self.i]),
                'symbol':self.p.symbols[k],'units':float(delta[k]),'reference_price':float(prices[k]),
                'price':float(fill[k]),'fee':float(fees[k]),'reason':reason})

    def rebalance(self,weights,prices):
        w=np.asarray(weights,dtype=float)
        if w.shape!=self.units.shape or not np.isfinite(w).all() or np.abs(w).sum()>1+1e-10:
            raise ValueError('Invalid target weights')
        if np.any((w<0)&(prices<5)):raise ValueError('Low-price short outside this engine scope')
        equity=self.equity_at(prices)
        if equity<=0:raise ArithmeticError('Insolvent account cannot initiate positions')
        net=equity
        for _ in range(100):
            delta=w*net/prices-self.units
            cost=np.abs(delta*prices)*(self.slip+self.fee*(1+self.slip*np.sign(delta)))
            nxt=equity-float(cost.sum())
            if abs(nxt-net)<1e-10:net=nxt;break
            net=nxt
        if net<=0:raise ArithmeticError('Costs exceed equity')
        desired=w*net/prices
        prospective_cash=net-float(desired@prices)
        if prospective_cash-1.5*float(-np.minimum(desired,0)@prices)<-1e-7:
            raise ArithmeticError('Insufficient short collateral')
        self.execute(desired-self.units,prices,'rebalance')

    def step(self,weights=None):
        if self.i>=self.end:raise ValueError('Finished')
        prev=self.i;self.i+=1;t=self.i;p=self.p
        if self.bankrupt:self.record();return
        days=(p.timestamp[t].normalize()-p.timestamp[prev].normalize()).days
        earned=max(self.free_cash(p.close[prev]),0.)*p.cash_rate[t]*days/365
        borrow=self.short_value(p.close[prev])*self.borrow_rate*days/365
        self.cash+=earned-borrow;self.interest+=earned;self.borrow_paid+=borrow
        dist=self.units*p.dividend[t]
        self.cash+=float(dist.sum())
        self.long_dividends+=float(np.maximum(dist,0).sum())
        self.short_dividends+=float(-np.minimum(dist,0).sum())
        breach=self.pending_liquidation or self.free_cash(p.opening[t]) < -1e-7
        if breach:
            self.execute(-self.units.copy(),p.opening[t],'collateral_liquidation')
            self.margin_liquidations+=1
        elif weights is not None:self.rebalance(weights,p.opening[t])
        self.pending_liquidation=self.free_cash(p.close[t]) < -1e-7
        # Preserve any debt, rather than forgiving a loss beyond initial capital.
        if self.equity_at(p.close[t])<=0:
            self.pending_liquidation=True
            if not np.any(self.units):self.bankrupt=True
        if t==self.end and np.any(self.units):
            self.execute(-self.units.copy(),p.close[t],'terminal')
            self.pending_liquidation=False
        if self.cash<=0 and not np.any(self.units):self.bankrupt=True
        self.record()

    def record(self):
        prices=self.p.close[self.i]
        self.history.append({'timestamp':self.p.bar_end[self.i],'equity':self.equity_at(prices),'cash':self.cash,
            'short_collateral':1.5*self.short_value(prices),'free_cash':self.free_cash(prices),
            'gross_notional':float(np.abs(self.units*prices).sum()),'net_notional':float(self.units@prices),
            'fees':self.fees,'slippage':self.slippage,'borrow':self.borrow_paid,'interest':self.interest,
            'long_dividends':self.long_dividends,'short_dividends':self.short_dividends,
            'pending_liquidation':self.pending_liquidation,'bankrupt':self.bankrupt,
            **{f'units_{s}':float(u) for s,u in zip(self.p.symbols,self.units)}})

def features(p,frames):
    log_returns=np.diff(np.log(p.total_index),axis=0,prepend=np.log(p.total_index[:1]))
    beta=np.full_like(log_returns,np.nan);variance=np.full_like(log_returns,np.nan)
    correlation=np.full(len(log_returns),np.nan)
    market=p.symbols.index('SPY');a=p.symbols.index('XLP');b=p.symbols.index('XLY')
    for j in range(64,len(log_returns)):
        past=log_returns[j-63:j]  # strictly before the signal session
        variance[j]=past.var(axis=0,ddof=1)
        x=past[:,market]-past[:,market].mean();v=x@x
        beta[j]=x@(past-past.mean(axis=0))/v if v>1e-16 else 0
        correlation[j]=np.corrcoef(past[:,a],past[:,b])[0,1] if np.std(past[:,a])*np.std(past[:,b])>1e-16 else 0
    ibs=np.column_stack([np.divide(f.close-f.low,f.high-f.low,out=np.full(len(f),.5),where=f.high!=f.low) for f in frames.values()])
    return {'r':log_returns,'beta':beta,'var':variance,'corr':correlation,'ibs':ibs}

def normalize(v,gross=1.):
    den=np.abs(v).sum()
    return v*(gross/den) if den>1e-12 else np.zeros_like(v)

def target(rule,j,p,d):
    w=np.zeros(len(p.symbols));r=d['r'][j,:9]
    if rule in ('LSMomentum','LSLowVol','LSIBS'):
        if rule=='LSMomentum':score=p.total_index[j-21,:9]/p.total_index[j-252,:9]-1
        elif rule=='LSLowVol':score=-p.volatility[j,:9]
        else:score=-d['ibs'][j,:9]
        if np.ptp(score)<1e-12:return w
        ix=np.argsort(-score,kind='stable');w[ix[:3]]=1/6;w[ix[-3:]]=-1/6
    elif rule=='LSCluster':w[:9]=normalize(-(r-r.mean()))
    elif rule=='LSMultiCluster':
        for group in GROUPS:
            ix=np.array(group);w[ix]=normalize(-(r[ix]-r[ix].mean()),1/3)
    elif rule=='LSWeightedResidual':
        exposures=np.column_stack([np.ones(9),d['beta'][j,:9]])
        z=1/np.maximum(d['var'][j,:9],1e-8)
        fit=np.linalg.lstsq(exposures*np.sqrt(z[:,None]),r*np.sqrt(z),rcond=None)[0]
        w[:9]=normalize(-z*(r-exposures@fit))
    elif rule=='LSPair':
        a=p.symbols.index('XLP');b=p.symbols.index('XLY')
        if d['corr'][j]>=.60 and abs(r[a]-r[b])>1e-12:
            w[a]=-.5*np.sign(r[a]-r[b]);w[b]=-w[a]
    else:raise ValueError(rule)
    if not np.isfinite(w).all() or abs(w.sum())>1e-7 or np.abs(w).sum()>1+1e-7:
        raise ArithmeticError('Invalid signed signal')
    return w

def run(p,frames,d,rule,start,end,fee=.0005,slip=.0005,borrow=.02,delay=1,zero_cash=False,record=False):
    pp=replace(p,cash_rate=np.zeros_like(p.cash_rate)) if zero_cash else p
    benchmark=rule in BENCHMARKS
    ledger=Ledger(pp,start,end,fee,slip) if benchmark else SignedLedger(pp,start,end,fee,slip,borrow)
    decisions=[]
    for t in range(start,end+1):
        j=t-delay;w=None
        assert p.bar_end[j]<=p.timestamp[t]
        if benchmark and t==start:
            w=np.zeros(len(p.symbols))
            if rule=='BuyHold':w[p.symbols.index('SPY')]=1
            elif rule=='PassiveBasket':w[:9]=1/9
        elif not benchmark:
            monthly=rule in ('LSMomentum','LSLowVol')
            if not monthly or t==start or p.timestamp[t].month!=p.timestamp[t-1].month:w=target(rule,j,p,d)
        if w is not None and record:decisions.append({'decided_at':str(p.bar_end[j]),'execute_at':str(p.timestamp[t]),'weights':w.tolist()})
        if benchmark:ledger.step(w,drip=True)
        else:ledger.step(w)
    history=pd.DataFrame(ledger.history);eq=history.equity.to_numpy()
    days=(p.bar_end[end]-p.timestamp[start]).total_seconds()/86400
    m={'return':float(eq[-1]/10000-1),'cagr':float((eq[-1]/10000)**(365.25/days)-1) if eq[-1]>0 else None,
       'max_drawdown':float((1-eq/np.maximum.accumulate(eq)).max()),'fills':len(ledger.fills),
       'fees':ledger.fees,'slippage':ledger.slip_paid if benchmark else ledger.slippage,
       'interest':ledger.interest,'borrow':0 if benchmark else ledger.borrow_paid,
       'long_dividends':ledger.dividends if benchmark else ledger.long_dividends,
       'short_dividends':0 if benchmark else ledger.short_dividends,
       'collateral_liquidations':0 if benchmark else ledger.margin_liquidations,
       'bankrupt':False if benchmark else ledger.bankrupt,'days':days}
    return m,history,ledger.fills,decisions

def main():
    OUT.mkdir(exist_ok=True);(OUT/'ledgers').mkdir(exist_ok=True)
    jobs,sources=load_jobs();_,frames,_,_=next(j for j in jobs if j[0]=='Sectors')
    p=panel_from(frames,True,252);d=features(p,frames)
    cases={'base':{},'double_cost':{'fee':.001,'slip':.001},'high_borrow':{'borrow':.10},
           'extra_delay':{'delay':2},'zero_cash':{'zero_cash':True}}
    rows=[]
    for window,left,right in BLOCKS:
        ix=np.flatnonzero((p.timestamp>=pd.Timestamp(left,tz='UTC'))&(p.timestamp<pd.Timestamp(right,tz='UTC')))
        start,end=int(ix[0]),int(ix[-1])
        for case,args in cases.items():
            if window!='full' and case!='base':continue
            for rule in RULES+BENCHMARKS:
                record=window=='full' and case=='base'
                metrics,h,fills,decisions=run(p,frames,d,rule,start,end,record=record,**args)
                rows.append({'strategy':rule,'window':window,'case':case,**metrics})
                if record:
                    h.to_parquet(OUT/'ledgers'/f'{rule}.parquet',index=False)
                    (OUT/'ledgers'/f'{rule}.json').write_text(json.dumps({'fills':fills,'decisions':decisions},default=str))
            print(json.dumps({'window':window,'case':case,'completed_runs':len(rows)}),flush=True)
    result=pd.DataFrame(rows)
    for name,suffix in [('BuyHold','bh'),('PassiveBasket','basket'),('Cash','cash')]:
        b=result[result.strategy==name][['window','case','return','cagr','max_drawdown']]
        result=result.merge(b,on=['window','case'],suffixes=('',f'_{suffix}'),validate='many_to_one')
    result['pass']=(result['return']>result.return_bh)&(result.max_drawdown<=result.max_drawdown_bh)&~result.bankrupt
    ranked=[]
    for name in RULES:
        g=result[result.strategy==name];f=g[(g.window=='full')&(g['case']=='base')].iloc[0]
        ranked.append({'strategy':name,'qualified':bool(g['pass'].all() and not g.collateral_liquidations.any()),
            'passed_cases':int(g['pass'].sum()),'scored_cases':len(g),'full_return':f['return'],
            'max_drawdown':f.max_drawdown,'median_excess_cagr':float((g.cagr-g.cagr_bh).median())})
    ranking=sorted(ranked,key=lambda x:(-x['qualified'],-x['passed_cases'],-x['median_excess_cagr'],x['strategy']))
    (OUT/'metrics.json').write_text(result.to_json(orient='records',indent=2))
    (OUT/'ranking.json').write_text(json.dumps(ranking,indent=2))
    manifest={'sources':{k:v for k,v in sources.items() if k.startswith('sector_')},
        'protocol_sha256':hashlib.sha256((HERE/'SIGNED_PROTOCOL.md').read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'recipes':RULES,'runs':len(result),'start':str(p.timestamp[np.flatnonzero(p.timestamp>=pd.Timestamp('2017-01-01',tz='UTC'))[0]]),'end':str(p.bar_end[-1]),
        'borrow_status':'Assumed fees and availability; not historically verified',
        'scope':'Fixed ETF adaptations; separate from first-screen ranking'}
    cash_path=HERE.parents[1]/'data/rotation/cash_yield.parquet'
    manifest['sources']['cash_yield']={'path':str(cash_path),'sha256':hashlib.sha256(cash_path.read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'runs':len(result),'qualified':sum(r['qualified'] for r in ranking)}),flush=True)

def frictionless_diagnostic():
    """Post-screen explanation only: never used by the fixed ranking."""
    jobs,_=load_jobs();_,frames,_,_=next(j for j in jobs if j[0]=='Sectors')
    p=panel_from(frames,True,252);d=features(p,frames)
    start=int(np.flatnonzero(p.timestamp>=pd.Timestamp('2017-01-01',tz='UTC'))[0])
    rows=[]
    for rule in RULES:
        m,_,_,_=run(p,frames,d,rule,start,len(p.timestamp)-1,fee=0,slip=0,borrow=0,zero_cash=True)
        rows.append({'strategy':rule,'diagnostic':'Zero trading/borrow costs and zero cash interest; not ranked; added after screen to examine failure mechanism',**m})
    OUT.mkdir(exist_ok=True)
    (OUT/'frictionless_diagnostic.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    if sys.argv[1:]==['--diagnostics-only']:frictionless_diagnostic()
    elif sys.argv[1:]:raise SystemExit('Usage: signed_study.py [--diagnostics-only]')
    else:main()
