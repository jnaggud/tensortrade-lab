"""Freeze supplemental public ETF data; no brokerage/account access."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from tensortrade_lab.sources import download_yahoo
from full_catalogue_options import OUT,save

def main():
    symbols=sys.argv[1:] or ['SPXL','SPXS','SHY','IEF','TLT','FXE','FXB','FXY','FXA','VNQ','VNQI','SPY']
    folder=OUT/'data';folder.mkdir(exist_ok=True)
    label='_additional' if sys.argv[1:] else ''
    save(OUT/('supplemental_data_request'+label+'.json'),{'symbols':symbols,'start':'2013-01-01','end_exclusive':'2026-09-24',
        'source':'yfinance, unadjusted-for-distributions OHLC with explicit distributions and split-adjusted prices',
        'purpose':'LETF pair, causal HP currency-ETF proxies, REIT diversification proxy. The FX ETFs are NOT spot FX or financed FX forwards; REITs are NOT property-level cashflows.'})
    errors=[]
    for symbol in symbols:
        path=folder/(symbol+'.parquet')
        if path.exists():continue
        try:
            m=download_yahoo(symbol,path,start='2013-01-01',end='2026-09-24')
            print(symbol,m['quality'],flush=True)
        except Exception as e:errors.append({'symbol':symbol,'error':str(e)});print('FAIL',symbol,str(e),flush=True)
    save(OUT/('supplemental_data_errors'+label+'.json'),errors)
if __name__=='__main__':main()
