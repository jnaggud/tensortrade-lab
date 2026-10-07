"""Independent final-cash replay of saved fills and report-count checks."""
from intraday_common import *

def main():
    rows=[]
    for filename in ['btc_results.json','futures_intraday_results.json','daily_futures_results.json']:
        rows+=json.loads((OUT/filename).read_text())
    checks=[]
    for r in rows:
        if r['block']!='full' or r['case']!='base':continue
        fills=json.loads((OUT/(r['strategy']+'_fills.json')).read_text())
        if 'actual' in r['market']:
            multiplier={'ES':50.,'GC':100.,'CL':1000.}[r['market'].split()[0]]
            reconstructed=r['initial_capital']-sum(z['units']*z['modeled_price']*multiplier+z['fee'] for z in fills)
            exposure={symbol:sum(z['units'] for z in fills if z['symbol']==symbol) for symbol in {z['symbol'] for z in fills}}
            assert all(v==0 for v in exposure.values()),(r['strategy'],exposure)
        else:
            reconstructed=r['initial_capital']-sum(z['units']*z['price'] for z in fills)-r['fees']-r['borrow']
            assert abs(sum(z['units'] for z in fills))<1e-9,r['strategy']
        expected=r['initial_capital']*(1+r['total_return'])
        difference=abs(expected-reconstructed)
        assert difference<.01,(r['strategy'],expected,reconstructed,difference)
        equity=pd.read_parquet(OUT/(r['strategy']+'_equity.parquet'))
        assert abs(equity.equity.iloc[-1]-expected)<.0001
        assert equity.timestamp.is_monotonic_increasing and not equity.timestamp.duplicated().any()
        checks.append({'strategy':r['strategy'],'independent_final_cash':reconstructed,'reported_final_cash':expected,'difference_dollars':difference,'passed':True})
    save('fill_reconciliation.json',{'at':datetime.now(timezone.utc).isoformat(),'method':'Independent signed fill-cash ledger, fee and borrow debits; terminal net units zero per contract. Compare saved equity and metrics, without calling either strategy engine.','checks':checks})
    print('Independent full-case fill ledgers reconciled:',len(checks),'maximum dollar difference:',max(r['difference_dollars'] for r in checks))

if __name__=='__main__':main()
