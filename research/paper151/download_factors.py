"""Download official French monthly factors and published annual snapshots."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin
import concurrent.futures
import datetime as dt
import hashlib
import io
import json
import re
import zipfile
import pandas as pd
import requests

HERE=Path(__file__).parent
DEST=HERE/'data/factors'
ARCHIVE='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors_archive.html'
CURRENT='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            if href:self.links.append(urljoin(ARCHIVE,href))

def parse_monthly(text):
    rows=[]
    for line in text.splitlines():
        parts=re.split(r'\s*,\s*|\s+',line.strip())
        if len(parts)>=5 and re.fullmatch(r'\d{6}',parts[0]):
            values=list(map(float,parts[1:5]))
            if min(values)<=-99:raise ValueError('Missing factor observation')
            rows.append([parts[0],*[v/100 for v in values]])
    f=pd.DataFrame(rows,columns=['month','Mkt-RF','SMB','HML','RF'])
    if f.empty or f.month.duplicated().any():raise ValueError('Invalid monthly table')
    return f

def fetch_one(label,url):
    r=requests.get(url,timeout=30);r.raise_for_status()
    archive=zipfile.ZipFile(io.BytesIO(r.content))
    name=next(n for n in archive.namelist() if n.lower().endswith(('.csv','.txt')))
    text=archive.read(name).decode('utf-8-sig',errors='replace')
    f=parse_monthly(text)
    raw=DEST/f'{label}.zip';raw.write_bytes(r.content)
    (DEST/f'{label}.txt').write_text(text)
    f.to_parquet(DEST/f'{label}.parquet',index=False)
    return {'label':label,'url':url,'raw_sha256':hashlib.sha256(r.content).hexdigest(),
        'path':str((DEST/f'{label}.parquet').resolve()),'sha256':hashlib.sha256((DEST/f'{label}.parquet').read_bytes()).hexdigest(),
        'rows':len(f),'first_month':str(f.month.iloc[0]),'last_month':str(f.month.iloc[-1])}

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    r=requests.get(ARCHIVE,timeout=30);r.raise_for_status();(DEST/'archive_index.html').write_text(r.text)
    parser=Links();parser.feed(r.text)
    jobs=[('latest',CURRENT)]
    for y in range(2016,2026):
        links=[u for u in parser.links if str(y) in u and 'csv' in u.lower() and u.lower().endswith('.zip')]
        if len(links)!=1:raise ValueError(f'Expected one archive for {y}, got {links}')
        jobs.append((str(y),links[0]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        sources=list(ex.map(lambda pair:fetch_one(*pair),jobs))
    manifest={'downloaded_at':dt.datetime.now(dt.timezone.utc).isoformat(),'source_page':ARCHIVE,
        'publication_convention':'Each July snapshot is documented as released in August; eligible from September 1. No exact publication day claimed.',
        'sources':sources}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(sources,indent=2))

if __name__=='__main__':main()
