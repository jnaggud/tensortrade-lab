"""Frictionless diagnostic only; never eligible for promotion."""
from intraday_common import *

def main():
    save('btc_zero_cost_protocol.json',{'registered_at':datetime.now(timezone.utc).isoformat(),'purpose':'Explain cost drag after completed base tests; no retuning. No commission, slippage or short borrow; economically unattainable diagnostic.'})
    f=pd.read_parquet(OUT/'BTC_BITSTAMP_15m.parquet');configs=[r for r in json.loads((HERE/'extension/patternfindr_configs.json').read_text()) if r.get('config',{}).get('ticker')=='BTC-USD' and r['config'].get('interval')=='15m']
    p=json.loads((OUT/'btc_protocol.json').read_text())['quant_params'];spec=importlib.util.spec_from_file_location('btc_diag_signals',OUT/'sources/btc_reversal_signals.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    rows=[];start=int(np.flatnonzero(f.timestamp>=pd.Timestamp('2024-03-01',tz='UTC'))[0]);end=len(f)-1
    for name,kind,d in [(r['id'],'pf',r['config']) for r in configs]+[('BTC_TVRange_Reversal_64k','quant',p)]:
        if kind=='pf':s=pf_signals(f,d)
        else:
            feat=q.build_features(f.set_index('timestamp'));reg,v,p=q.make_signals(feat,d);s=pd.DataFrame({'regime':reg,'valley':v,'peak':p,'atr':feat.atr.to_numpy()})
        m,_,_=spot_run(f,s,d,start,end,fee=0,slip=0,borrow_rate=0,kind=kind);rows.append({'strategy':name,**m});print(name,m['total_return'],flush=True)
    save('btc_zero_cost_results.json',rows)
if __name__=='__main__':main()
