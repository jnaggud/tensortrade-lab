"""Saved C7 and corrected C8 choices: diagnostic transport, never new optimization."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import importlib.util,sys,json
import pandas as pd
import numpy as np
from cross_project_inventory import OUT,QUANT,save,sha

def main():
    source=QUANT/'reports/c8_family_corrected_rerank_20260620.json'
    report=json.loads(source.read_text())
    wanted=['C7 Constrained Refine C1','C8 Successor 200k C5']
    specs=[x for x in report['ranked'] if x['name'] in wanted]
    assert len(specs)==2
    save('quant_family_protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),
        'source':str(source),'source_sha256':sha(source),'choices':specs,
        'selection':'Saved active C7 and previously corrected-ranked C8 C5. No new parameter search.',
        'transport':'Disable historical date gate; same 2021-2026 continuous feed, one ES contract, $50,000 initial capital. Default parity module for missing fields. This is a diagnostic adaptation, not exact saved TradingView feed reproduction.',
        'costs':'$5 round trip commission; 1,2,3,5 ticks adverse slippage per market side.',
        'eligibility':'Quarantined: same-day-volume continuous feed, adjusted fill prices, same-close execution and no dated margin history.'})
    spec=importlib.util.spec_from_file_location('family_parity',OUT/'sources/c11_parity.py')
    e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
    paths=sorted((QUANT/'quant/cache').glob('ES_continuous_minute_*.parquet'))
    frames=[]
    for path in paths:
        f=pd.read_parquet(path)
        for col in ['open','high','low','close']:f[col]=f.get('adj_'+col,f[col])
        frames.append(f[['ts','open','high','low','close','volume']])
    raw=pd.concat(frames).drop_duplicates('ts').sort_values('ts').set_index('ts')
    f=raw.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    feat=e.build_features(f);close=f.close.to_numpy();rows=[]
    for item in specs:
        p=e.default_c5_params();p.update(item['params']);p['use_date_range']=False
        bars=e.build_c5_signal_bars(feat,p)
        for ticks in [1,2,3,5]:
            e.SLIPPAGE_TICKS=float(ticks);bt=e.run_tv_compatible_signal_bars(bars,p)
            realized=np.zeros(len(f));unreal=np.zeros(len(f))
            for t in bt.trades:
                realized[t.exit_bar]+=t.pnl
                unreal[t.entry_bar:t.exit_bar]+=(close[t.entry_bar:t.exit_bar]-t.entry_price)*t.direction*50-5
            eq=50000+np.cumsum(realized)+unreal
            assert np.isclose(eq[-1]-50000,bt.net_profit)
            daily=pd.Series(eq,index=f.index).resample('1D').last().dropna()
            dd=float((1-daily/np.maximum.accumulate(np.r_[50000,daily])[1:]).max())
            rows.append({'strategy':item['name'],'case':f'{ticks}_ticks_per_market_side',
              'origin':'trading_view_mcp_quant','start':str(f.index[0]),'end':str(f.index[-1]),
              'initial_capital':50000,'contracts':1,'total_return':bt.net_profit/50000,'net_pnl':bt.net_profit,
              'max_drawdown':dd,'ever_insolvent':bool(eq.min()<=0),'trades':bt.n_trades,
              'status':'quarantined_continuous_feed_and_same_close_fills'})
            if ticks==1:
                tag='C7' if item['name'].startswith('C7') else 'C8'
                pd.DataFrame({'timestamp':f.index,'equity':eq}).to_parquet(OUT/(tag+'_marked_equity.parquet'),index=False)
            print(item['name'],ticks,bt.net_profit,dd,flush=True)
    save('quant_family_results.json',rows)
    save('quant_family_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'code_sha256':sha(__file__),'source_hashes':{str(p):sha(p) for p in paths},
        'protocol_sha256':sha(OUT/'quant_family_protocol.json'),'runs':len(rows),'qualified':0})

if __name__=='__main__':main()
