"""Three fixed factor rules, with archive availability separated from revisions."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from expanded_study import load_jobs,BLOCKS
from signed_study import SignedLedger,run as benchmark_run
from run_study import panel_from,Ledger

HERE=Path(__file__).parent;OUT=HERE/'factor_results'
RULES=['ResidualMomentumLong','ResidualMomentumLS','AlphaRotation']
MODES=['archived','revised_annual','revised_monthly']

def load_factors():
    manifest=json.loads((HERE/'data/factors/manifest.json').read_text());out={}
    for s in manifest['sources']:
        path=Path(s['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==s['sha256']
        f=pd.read_parquet(path);f.index=pd.PeriodIndex(f.pop('month'),freq='M');out[s['label']]=f
    return out,manifest

def monthly_returns(p):
    months=p.timestamp.tz_localize(None).to_period('M')
    end=pd.DataFrame(p.total_index,index=months,columns=p.symbols).groupby(level=0).last()
    return end.pct_change(fill_method=None)

def factor_window(mode,decision,nominal_execution,factors):
    decision=pd.Timestamp(decision)
    eligible=[int(k) for k in factors if k!='latest' and pd.Timestamp(f'{k}-09-01',tz='UTC')<=decision]
    if not eligible:raise ValueError('No factor archive available by the decision')
    vintage=str(max(eligible));snapshot=factors[vintage]
    if mode=='archived':table=snapshot;cutoff=snapshot.index[-1]
    elif mode=='revised_annual':table=factors['latest'];cutoff=snapshot.index[-1]
    elif mode=='revised_monthly':
        table=factors['latest'];cutoff=pd.Timestamp(nominal_execution).tz_localize(None).to_period('M')-2
    else:raise ValueError(mode)
    if cutoff.end_time>=decision.tz_localize(None):raise ValueError('Factor month not complete')
    months=pd.period_range(end=cutoff,periods=36,freq='M')
    if not months.isin(table.index).all():raise ValueError('Missing factor months')
    return table.loc[months],{'vintage':vintage if mode=='archived' else 'latest',
        'archive_reference':vintage,'last_factor_month':str(cutoff),
        'factor_archive_eligible_from':f'{vintage}-09-01','factor_mode':mode}

def fit_scores(monthly,factors):
    y=monthly.loc[factors.index].iloc[:,:9].to_numpy()-factors.RF.to_numpy()[:,None]
    x=factors[['Mkt-RF','SMB','HML']].to_numpy();design=np.column_stack([np.ones(len(x)),x])
    if not np.isfinite(y).all() or not np.isfinite(x).all():raise ValueError('Missing monthly data')
    if np.linalg.matrix_rank(design)!=4:raise ValueError('Rank-deficient factor design')
    beta=np.linalg.lstsq(design,y,rcond=None)[0]
    residual=y[-12:]-x[-12:]@beta[1:]  # Eq.279 retains fitted alpha
    deviation=residual.std(axis=0,ddof=1)
    score=np.divide(residual.mean(axis=0),deviation,out=np.zeros(9),where=deviation>1e-12)
    return score,beta[0],beta

def signal(rule,mode,decision,execution,p,monthly,factors):
    f,audit=factor_window(mode,decision,execution,factors)
    residual,alpha,beta=fit_scores(monthly,f);score=alpha if rule=='AlphaRotation' else residual
    order=np.argsort(-score,kind='stable');w=np.zeros(len(p.symbols))
    if np.ptp(score)>1e-12:
        if rule.endswith('LS'):w[order[:3]]=1/6;w[order[-3:]]=-1/6
        else:w[order[:3]]=1/3
    audit.update(scores=score.tolist(),weights=w.tolist(),betas=beta.tolist(),observations=36)
    return w,audit

def simulate(p,monthly,factors,rule,mode,start,end,fee=.0005,slip=.0005,borrow=.02,delay=1,zero_cash=False,record=False):
    pp=replace(p,cash_rate=np.zeros_like(p.cash_rate)) if zero_cash else p
    signed=rule.endswith('LS');ledger=SignedLedger(pp,start,end,fee,slip,borrow) if signed else Ledger(pp,start,end,fee,slip)
    pending={};audits=[]
    for t in range(start,end+1):
        if t==start or p.timestamp[t].month!=p.timestamp[t-1].month:
            j=t-1;execute=t+delay-1
            w,a=signal(rule,mode,p.bar_end[j],p.timestamp[t],p,monthly,factors)
            assert p.bar_end[j]<=p.timestamp[t]
            if execute<=end:
                pending[execute]=w
                if record:audits.append({'decided_at':str(p.bar_end[j]),'execute_at':str(p.timestamp[execute]),**a})
        w=pending.get(t)
        if signed:ledger.step(w)
        else:ledger.step(w,drip=False)
    h=pd.DataFrame(ledger.history);eq=h.equity.to_numpy()
    days=(p.bar_end[end]-p.timestamp[start]).total_seconds()/86400
    m={'return':float(eq[-1]/10000-1),'cagr':float((eq[-1]/10000)**(365.25/days)-1) if eq[-1]>0 else None,
       'max_drawdown':float((1-eq/np.maximum.accumulate(eq)).max()),'fees':ledger.fees,
       'slippage':ledger.slippage if signed else ledger.slip_paid,'interest':ledger.interest,
       'borrow':ledger.borrow_paid if signed else 0,'fills':len(ledger.fills),
       'collateral_liquidations':ledger.margin_liquidations if signed else 0,
       'bankrupt':ledger.bankrupt if signed else False}
    return m,h,ledger.fills,audits

def main():
    OUT.mkdir(exist_ok=True);(OUT/'ledgers').mkdir(exist_ok=True)
    jobs,source=load_jobs();_,frames,_,_=next(j for j in jobs if j[0]=='Sectors')
    p=panel_from(frames,True,252);monthly=monthly_returns(p);factors,fm=load_factors()
    cases={'base':{},'double_cost':{'fee':.001,'slip':.001},'high_borrow':{'borrow':.10},
           'extra_delay':{'delay':2},'zero_cash':{'zero_cash':True}}
    rows=[]
    for window,left,right in BLOCKS:
        ix=np.flatnonzero((p.timestamp>=pd.Timestamp(left,tz='UTC'))&(p.timestamp<pd.Timestamp(right,tz='UTC')));start,end=int(ix[0]),int(ix[-1])
        for case,args in cases.items():
            if window!='full' and case!='base':continue
            for rule in RULES+['BuyHold','PassiveBasket','Cash']:
                modes=MODES if rule in RULES else ['benchmark']
                for mode in modes:
                    rec=window=='full' and case=='base'
                    if mode=='benchmark':m,h,fills,audits=benchmark_run(p,frames,{},rule,start,end,record=rec,**args)
                    else:m,h,fills,audits=simulate(p,monthly,factors,rule,mode,start,end,record=rec,**args)
                    rows.append({'market':'Sectors','strategy':rule,'factor_mode':mode,'window':window,'case':case,**m})
                    if rec:
                        h.to_parquet(OUT/'ledgers'/f'{rule}__{mode}.parquet',index=False)
                        (OUT/'ledgers'/f'{rule}__{mode}.json').write_text(json.dumps({'fills':fills,'decisions':audits},default=str))
            print(json.dumps({'window':window,'case':case,'runs':len(rows)}),flush=True)
    result=pd.DataFrame(rows)
    for name,suffix in [('BuyHold','bh'),('PassiveBasket','basket'),('Cash','cash')]:
        b=result[result.strategy==name][['window','case','return','cagr','max_drawdown']]
        result=result.merge(b,on=['window','case'],suffixes=('',f'_{suffix}'),validate='many_to_one')
    result['pass']=(result['return']>result.return_bh)&(result.max_drawdown<=result.max_drawdown_bh)&~result.bankrupt
    rankings=[]
    for (rule,mode),g in result[result.strategy.isin(RULES)].groupby(['strategy','factor_mode']):
        rankings.append({'strategy':rule,'factor_mode':mode,'qualified':bool(mode=='archived' and g['pass'].all() and not g.collateral_liquidations.any()),
            'passes':int(g['pass'].sum()),'scored_cases':len(g),'median_excess_cagr':float((g.cagr-g.cagr_bh).median()),
            'promotion_eligible_data_mode':mode=='archived'})
    (OUT/'metrics.json').write_text(result.to_json(orient='records',indent=2))
    (OUT/'ranking.json').write_text(json.dumps(rankings,indent=2))
    cash=HERE.parents[1]/'data/rotation/cash_yield.parquet'
    manifest={'factor_sources':fm,'price_sources':{k:v for k,v in source.items() if k.startswith('sector_')},
        'cash_sha256':hashlib.sha256(cash.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((HERE/'FACTOR_PROTOCOL.md').read_bytes()).hexdigest(),'runs':len(result),
        'note':'Three recipes for two paper ideas, across three availability/revision modes. No new independent holdout.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'runs':len(result),'qualified':sum(x['qualified'] for x in rankings)}))

if __name__=='__main__':main()
