"""Independent diagnostics of exported Pine metrics; no rule/parameter tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from expanded_study import load_jobs,portfolio_target
from run_study import panel_from
HERE=Path(__file__).parent;OUT=HERE/'tradingview/reconciliation'

def main():
    x=json.loads((OUT/'portfolio_input_metrics.json').read_text())
    cols=[x['styles'][p['id']]['title'] for p in x['plots']]
    tv=pd.DataFrame(x['rows'],columns=['timestamp']+cols);tv.timestamp=pd.to_datetime(tv.timestamp,unit='s',utc=True);tv=tv.set_index('timestamp')
    frames=next(j[1] for j in load_jobs()[0] if j[0]=='Sectors');p=panel_from(frames,True,252)
    ix=np.flatnonzero((p.timestamp>=pd.Timestamp('2017-01-01',tz='UTC'))&(p.timestamp<=pd.Timestamp('2026-09-21 23:59',tz='UTC')))
    dates=[t for t in ix if t==ix[0] or p.timestamp[t].month!=p.timestamp[t-1].month]
    diagnostics=[]
    for sym in p.symbols[:9]:
        k=p.symbols.index(sym);fr=frames[sym]
        for t in dates:
            j=t-1;date=p.timestamp[t];r=tv.loc[date]
            local={'close':p.close[j,k],'ma200':p.close[j-199:j+1,k].mean(),'momentum':p.total_index[j,k]/p.total_index[j-252,k]-1,'skip':p.total_index[j-21,k]/p.total_index[j-252,k]-1,'vol':p.volatility[j,k],'return':p.total_index[j,k]/p.total_index[j-1,k]-1}
            diagnostics.append({'date':str(date),'symbol':sym,'local':local,'pine':{s:float(r[f'{sym} {s}']) for s in local}})
    (OUT/'portfolio_metric_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    for metric in ['close','ma200','momentum','skip','vol','return']:
        err=np.array([a['pine'][metric]-a['local'][metric] for a in diagnostics]);print(metric,'max',abs(err).max(),'mean',err.mean(),'p95',np.quantile(abs(err),.95))
    print('Sample 2017 April all sectors:')
    for r in diagnostics:
        if r['date'].startswith('2017-04'):print(r['symbol'],'local mom',r['local']['momentum'],'Pine mom',r['pine']['momentum'],'close',r['local']['close'],r['pine']['close'],'vol',r['local']['vol'],r['pine']['vol'])
if __name__=='__main__':main()
