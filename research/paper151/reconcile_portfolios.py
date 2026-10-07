"""Compare saved Pine daily targets against the frozen local decisions."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from expanded_study import load_jobs
from run_study import panel_from
HERE=Path(__file__).parent;OUT=HERE/'tradingview/reconciliation'

def main():
    summary={};jobs,_=load_jobs()
    for src in sorted(OUT.glob('portfolio_*.json')):
        if src.name=='portfolio_summary.json':continue
        data=json.loads(src.read_text())
        if not isinstance(data,dict) or 'name' not in data:continue
        rule=data['name'].split(' ')[2]
        if rule not in ('GoldDiversification','SectorLowVol','SectorRotationMA','SectorRotationDual','SectorRotation','SectorMomentumSkip21'):continue
        market='MultiAsset' if rule=='GoldDiversification' else 'Sectors'
        frames=next(j[1] for j in jobs if j[0]==market);p=panel_from(frames,True,252)
        raw=HERE/'expanded_results/ledgers'/f'{market}__{rule}.json'
        decisions=json.loads(raw.read_text())['decisions']
        cols=[data['styles'][x['id']]['title'].replace(' target %','') for x in data['plots'] if x['type']=='line']
        tv=pd.DataFrame([r[:len(cols)+1] for r in data['rows']],columns=['timestamp']+cols)
        tv.timestamp=pd.to_datetime(tv.timestamp,unit='s',utc=True);tv=tv.set_index('timestamp')/100
        df=pd.DataFrame([d['weights'] for d in decisions],index=pd.to_datetime([d['execute_at'] for d in decisions],utc=True),columns=p.symbols)
        df['Cash']=1-df.sum(axis=1);df=df.reindex(p.timestamp).ffill().loc[lambda x:x.index>=pd.Timestamp('2017-01-01',tz='UTC'),cols]
        tv=tv.reindex(df.index);error=(df-tv).abs();ok=(error<1e-9).all(axis=1)&tv.notna().all(axis=1)
        decision_ix=df.index.intersection(pd.to_datetime([d['execute_at'] for d in decisions],utc=True))
        bad=[]
        for day in df.index[~ok]:
            bad.append({'date':day.isoformat(),'local':df.loc[day].to_dict(),'pine':tv.loc[day].to_dict()})
        result={'strategy':rule,'market':market,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'local_decisions_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
                'daily_targets_compared':len(df),'daily_targets_matched':int(ok.sum()),'rebalance_days_compared':len(decision_ix),'rebalance_days_matched':int(ok.loc[decision_ix].sum()),
                'max_weight_difference':float(error.max().max()),'mismatch_months':sorted(set(x['date'][:7] for x in bad)),'missing_daily_rows':int(tv.isna().any(axis=1).sum()),
                'all_historical_targets_match':bool(ok.all()),'scope':'Fixed saved indicator on a US daily chart; local Yahoo economic panel through Sep21 2026 versus TradingView cross-symbol signals. Target weights, not drifted holdings, fills or portfolio PnL.'}
        (OUT/f'target_audit_{rule}.json').write_text(json.dumps({'summary':result,'mismatches':bad},indent=2));summary[rule]=result
    (OUT/'portfolio_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
