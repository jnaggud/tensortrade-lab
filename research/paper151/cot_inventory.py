"""Normalize downloaded COT observations without inventing publication times."""
from pathlib import Path
from datetime import datetime,timezone
import zipfile,json
import pandas as pd
from full_catalogue_options import save,sha
OUT=Path(__file__).resolve().parent/'data_expansion'

def main():
    parts=[];sources=[]
    columns={'Market and Exchange Names':'market','As of Date in Form YYYY-MM-DD':'report_date','CFTC Contract Market Code':'market_code','Open Interest (All)':'open_interest','Noncommercial Positions-Long (All)':'noncommercial_long','Noncommercial Positions-Short (All)':'noncommercial_short','Commercial Positions-Long (All)':'commercial_long','Commercial Positions-Short (All)':'commercial_short'}
    for path in sorted((OUT/'cot').glob('deacot*.zip')):
        audit=json.loads(path.with_suffix(path.suffix+'.audit.json').read_text())
        if sha(path)!=audit['sha256']:raise ValueError('Changed COT archive')
        with zipfile.ZipFile(path) as z:f=pd.read_csv(z.open(z.namelist()[0]),dtype={'CFTC Contract Market Code':str},low_memory=False)
        f=f[list(columns)].rename(columns=columns);f.report_date=pd.to_datetime(f.report_date,utc=True)
        for c in list(columns.values())[3:]:f[c]=pd.to_numeric(f[c],errors='raise')
        if (f[list(columns.values())[3:]]<0).any().any():raise ValueError('Negative COT position')
        f['source_file']=path.name;parts.append(f);sources.append({'path':str(path),'sha256':sha(path),'rows':len(f)})
    allrows=pd.concat(parts,ignore_index=True).sort_values(['market_code','report_date'])
    if allrows.duplicated(['market_code','report_date']).any():raise ValueError('Overlapping/ambiguous COT observations')
    allrows['published_at']=pd.NaT
    allrows['availability_status']='report_date_only; actual publication unverified'
    allrows.to_parquet(OUT/'cot_observations.parquet',index=False)
    save(OUT/'cot_manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'archives':len(sources),'rows':len(allrows),'markets':allrows.market_code.nunique(),'first_report':str(allrows.report_date.min()),'last_report':str(allrows.report_date.max()),'sources':sources,'normalized_sha256':sha(OUT/'cot_observations.parquet'),
      'ready_for_causal_backtest':False,'why':'Measurement dates are not publication dates. Official CFTC FAQ states no complete historical publication-date list. Tentative current schedule and special shutdown/backlog notices do not prove every actual publication timestamp. Annual archive is a current downloaded snapshot, not a locally archived sequence of original releases. Preserve null published_at; never invent Friday availability.',
      'additional_requirement':'Paper9.2 needs broad tradable commodity cross-section and matched dated contract/roll/financing histories; verified GC/CL alone does not replicate paper quintiles.',
      'official_sources':['https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm','https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm'],
      'next_import_contract':{'required_columns':['market_code','report_date','published_at','source_url','source_sha256','version_or_correction_id'],'validation':'UTC publication>=report observation; verify source evidence, duplicates, missing weeks, correction availability and delays. Join only observations already published by decision.'}})
    print('COT inventory',len(allrows),allrows.market_code.nunique(),flush=True)

if __name__=='__main__':main()
