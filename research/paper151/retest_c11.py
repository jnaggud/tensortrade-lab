"""Independent marked-equity audit of frozen C11 on the existing quant cache.

This feed has a known same-day-volume roll limitation: all results are diagnostic,
not executable individual-contract evidence or eligible for winner promotion.
"""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from cross_project_inventory import OUT,QUANT,save,sha

def main():
    source=QUANT/'scripts/tv_c5_parity_engine.py'
    target=OUT/'sources/c11_parity.py';target.write_bytes(source.read_bytes())
    spec=importlib.util.spec_from_file_location('c11_parity',target)
    e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
    paths=sorted((QUANT/'quant/cache').glob('ES_continuous_minute_*.parquet'))
    frames=[]
    for p in paths:
        f=pd.read_parquet(p)
        for c in ['open','high','low','close']:
            if 'adj_'+c in f:f[c]=f['adj_'+c]
        frames.append(f[['ts','open','high','low','close','volume']])
    raw=pd.concat(frames).drop_duplicates('ts').sort_values('ts').set_index('ts')
    f=raw.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    p=e.default_c5_params();p.update(json.loads((OUT/'c11_frozen_params.json').read_text())['params'])
    p.update(use_date_range=False,trend_carry_exit_regime='none',trend_carry_exit_on_macd_roll=False)
    feat=e.build_features(f);bars=e.build_c5_signal_bars(feat,p)
    cut=len(f)-12001;prefix=e.build_c5_signal_bars(e.build_features(f.iloc[:cut]),p)
    def comparable(x):return {k:v for k,v in asdict(x).items() if not isinstance(v,float) or np.isfinite(v)}
    mismatches=[i for i in range(300,cut) if comparable(bars[i])!=comparable(prefix[i])]
    save('c11_prefix_check.json',{'prefix_bars':cut,'mismatches':len(mismatches),'first_mismatches':mismatches[:12]})
    rows=[]
    for ticks in [1,2,3,5]:
        e.SLIPPAGE_TICKS=float(ticks)
        bt=e.run_tv_compatible_signal_bars(bars,p)
        realized=np.zeros(len(f));unreal=np.zeros(len(f))
        for trade in bt.trades:
            realized[trade.exit_bar]+=trade.pnl
            unreal[trade.entry_bar:trade.exit_bar]+=(f.close.to_numpy()[trade.entry_bar:trade.exit_bar]-trade.entry_price)*trade.direction*50-5
        eq=50000+np.cumsum(realized)+unreal
        assert np.isclose(eq[-1]-50000,bt.net_profit)
        daily=pd.Series(eq,index=f.index).resample('1D').last().dropna()
        bh=50000+(f.close.to_numpy()-f.open.iloc[0])*50-2*ticks*.25*50-5
        dd=float((1-daily/np.maximum.accumulate(np.r_[50000,daily])[1:]).max())
        rows.append({'strategy':'C11','origin':'trading_view_mcp_quant','case':f'{ticks}_ticks_per_market_side',
           'start':str(f.index[0]),'end':str(f.index[-1]),'initial_capital':50000,'contracts':1,
           'total_return':bt.net_profit/50000,'net_pnl':bt.net_profit,'max_drawdown':dd,
           'bar_close_drawdown':float((1-eq/np.maximum.accumulate(np.r_[50000,eq])[1:]).max()),
           'source_closed_trade_drawdown':bt.max_drawdown/100,'trades':bt.n_trades,'profit_factor':bt.profit_factor,
           'ever_insolvent':bool(eq.min()<=0),'continuous_price_buyhold_return':float(bh[-1]/50000-1),
           'status':'quarantined_continuous_feed_and_same_close_fills',
           'warning':'No real roll trades or dated margin constraints. Account is diagnostically allowed to continue after insolvency, NOT investable recovery.'})
        if ticks==1:
            pd.DataFrame({'timestamp':f.index,'equity':eq}).to_parquet(OUT/'c11_marked_equity.parquet',index=False)
            save('c11_reproduced_trades.json',[asdict(t) for t in bt.trades])
        print('C11',ticks,'ticks',round(bt.net_profit,2),'daily DD',round(dd,4),flush=True)
    save('c11_retest.json',rows)
    save('c11_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sha(source),'snapshot_sha256':sha(target),'code_sha256':sha(__file__),
        'bars':len(f),'sources':{str(p):sha(p) for p in paths},
        'prefix_mismatches':len(mismatches),'eligible_for_promotion':False})

if __name__=='__main__':main()
