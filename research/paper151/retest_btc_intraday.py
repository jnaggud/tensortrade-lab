"""Seven saved BTC configs plus the saved quant reversal; no parameter search."""
from intraday_common import *
from cross_project_inventory import QUANT

def main():
    source=QUANT/'tmp/bitstamp_btcusd_15m_20231231_20260507.json';payload=json.loads(source.read_text())
    f=pd.DataFrame(payload['bars']);f['timestamp']=pd.to_datetime(f.time,unit='s',utc=True);f['bar_end']=f.timestamp+pd.Timedelta(minutes=15);f=f.drop(columns='time').sort_values('timestamp').reset_index(drop=True)
    if f.timestamp.duplicated().any() or f[['open','high','low','close']].isna().any().any():raise ValueError('Bad BTC input')
    gaps=f.timestamp.diff()>pd.Timedelta(minutes=15)
    if gaps.any():raise ValueError('Missing BTC bars: '+str(int(gaps.sum())))
    f.to_parquet(OUT/'BTC_BITSTAMP_15m.parquet',index=False)
    configs=[r for r in json.loads((HERE/'extension/patternfindr_configs.json').read_text()) if r.get('config',{}).get('ticker')=='BTC-USD' and r['config'].get('interval')=='15m']
    report=QUANT/'reports/mtf_15m_reversal_tvtester_64k_min60.json';params=json.loads(report.read_text())['best']['params']
    periods=[('full','2024-03-01','2026-05-08'),('2024','2024-03-01','2025-01-01'),('2025','2025-01-01','2026-01-01'),('2026','2026-01-01','2026-05-08')]
    cases=[('base',1,1,.05),('double_cost',2,1,.10),('extra_bar_delay',1,2,.05)]
    save('btc_protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),'source':str(source),'source_sha256':sha(source),'bars':len(f),'gaps':int(gaps.sum()),
      'configs':[{'id':r['id'],'source_sha256':r['source_sha256']} for r in configs],'quant_params':params,'periods':periods,'cases':cases,
      'capital':10000,'execution':'95% exposure, next bar open; prior-bar ATR/stops, gap-aware, stop first on ambiguous bars; every 15-minute and UTC daily account marked. Fees 10bp and slippage 5bp each side, doubled in cost stress. Fractional BTC. Zero cash yield.',
      'shorts':'Saved quant rule permits shorts: modeled 100% collateral with 5%/10% annual borrow assumption and margin liquidation. Historical BITSTAMP borrow availability is unverified; short results research-only.',
      'regime_adaptation':'RSC now fits only prior 100 feature rows every 10 bars and predicts each current observation; no backwards filling or retrospective relabeling. Saved threshold unchanged. This is a declared causal redesign.',
      'status':'Retrospective diagnostics; prior searches already used part of history. Saved optimization date range disabled for full-period transport.'})
    names={'ema','rma','rsi','atr','macd_hist','stoch','resample_ohlcv','cross_over','cross_under','build_features','make_signals'}
    quant=pure_copy(QUANT/'scripts/optimize_15m_mtf_reversal.py','btc_reversal_signals.py',names)
    rows=[];causal=[]
    candidates=[(r['id'],'pf',r['config']) for r in configs]+[('BTC_TVRange_Reversal_64k','quant',params)]
    for name,kind,d in candidates:
        def calc(frame):
            if kind=='pf':return pf_signals(frame,d)
            z=frame.set_index('timestamp');feat=quant.build_features(z);reg,v,p=quant.make_signals(feat,d)
            return pd.DataFrame({'regime':reg,'valley':v,'peak':p,'atr':feat.atr.to_numpy()})
        print('signals',name,flush=True);signals=calc(f)
        cut=len(f)-1337;short=calc(f.iloc[:cut].copy())
        ok=np.allclose(signals.iloc[6000:cut].to_numpy(float),short.iloc[6000:].to_numpy(float),equal_nan=True,rtol=1e-10,atol=1e-10)
        causal.append({'strategy':name,'prefix_bars':cut,'passed':bool(ok)})
        if not ok:raise ValueError('Signal prefix failed '+name)
        for block,begin,finish in periods:
            ix=np.flatnonzero((f.timestamp>=pd.Timestamp(begin,tz='UTC'))&(f.timestamp<pd.Timestamp(finish,tz='UTC')));start,end=int(ix[0]),int(ix[-1])
            for case,cost,delay,borrow in cases:
                m,e,fill=spot_run(f,signals,d,start,end,fee=.001*cost,slip=.0005*cost,delay=delay,borrow_rate=borrow,kind=kind)
                b,_,_=spot_run(f,None,{},start,end,fee=.001*cost,slip=.0005*cost,kind='benchmark')
                rows.append({'strategy':name,'origin':'Pattern_FindR' if kind=='pf' else 'trading_view_mcp_quant','market':'BITSTAMP:BTCUSD 15m','block':block,'case':case,**m,
                    'benchmark_return':b['total_return'],'benchmark_drawdown':b['max_drawdown'],
                    'pass':bool(m['total_return']>b['total_return'] and m['max_drawdown']<=b['max_drawdown'])})
                if block=='full' and case=='base':
                    pd.DataFrame({'timestamp':f.bar_end.iloc[start:end+1],'equity':e}).to_parquet(OUT/(name+'_equity.parquet'),index=False);save(name+'_fills.json',fill)
        print('completed',name,flush=True);save('btc_results.json',rows);save('btc_causality.json',causal)
    save('btc_manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'candidates':len(candidates),'runs':len(rows),'benchmark_runs':len(rows),
      'code_sha256':sha(__file__),'common_code_sha256':sha(HERE/'intraday_common.py'),'protocol_sha256':sha(OUT/'btc_protocol.json'),'data_sha256':sha(OUT/'BTC_BITSTAMP_15m.parquet')})
if __name__=='__main__':main()
