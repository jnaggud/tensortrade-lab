"""Freeze public original-source responses. No keys, purchases, or live orders."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,sys
import requests
from bs4 import BeautifulSoup
from full_catalogue_options import save,sha

OUT=Path(__file__).resolve().parent/'data_expansion'
S=requests.Session()
S.headers['User-Agent']='TensorTradeLab historical academic research (local, low-rate requests)'

def fetch(name,url,params=None):
    p=OUT/name; a=p.with_suffix(p.suffix+'.audit.json')
    if p.exists() and a.exists():
        m=json.loads(a.read_text())
        if sha(p)!=m['sha256']:raise ValueError('Changed frozen download '+str(p))
        return p
    r=S.get(url,params=params,timeout=60)
    if r.status_code!=200:raise ValueError(f'HTTP {r.status_code}: {r.url}')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
    save(a,{'url':r.url,'downloaded_at':datetime.now(timezone.utc).isoformat(),'sha256':sha(p),'bytes':len(r.content),'content_type':r.headers.get('content-type')})
    time.sleep(.3)
    return p

def main():
    OUT.mkdir(exist_ok=True);log=[]
    tasks=[('sec_aapl_facts.json','https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json'),
      ('sec_aapl_submissions.json','https://data.sec.gov/submissions/CIK0000320193.json'),
      ('cftc_release_schedule.html','https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm'),
      ('cftc_special_announcements.html','https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm'),
      ('cftc_downloads.html','https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm'),
      ('alfred_CPIAUCSL.html','https://alfred.stlouisfed.org/series/downloaddata?seid=CPIAUCSL'),
      ('alfred_CPILFESL.html','https://alfred.stlouisfed.org/series/downloaddata?seid=CPILFESL')]
    for name,url in tasks:
        try:
            p=fetch(name,url);log.append({'file':name,'status':'downloaded','bytes':p.stat().st_size})
            if name=='cftc_downloads.html':
                links=BeautifulSoup(p.read_text(),'html.parser').find_all('a',href=True)
                for link in links:
                    href=link['href']
                    if any(href.endswith('deacot'+str(y)+'.zip') for y in range(2016,2027)):
                        q=fetch('cot/'+href.split('/')[-1],'https://www.cftc.gov'+href if href.startswith('/') else href)
                        log.append({'file':str(q.relative_to(OUT)),'status':'downloaded','bytes':q.stat().st_size})
        except (ValueError,requests.RequestException) as e:log.append({'file':name,'status':'failed','reason':str(e)})
        save(OUT/'download_status.json',log);print(log[-1],flush=True)
    # Use official download form parameters; do not guess an undocumented data endpoint.
    for series in ['CPIAUCSL','CPILFESL']:
        p=OUT/f'alfred_{series}.html'
        if not p.exists():continue
        soup=BeautifulSoup(p.read_text(),'html.parser')
        forms=[]
        for form in soup.find_all('form'):
            forms.append({'action':form.get('action'),'method':form.get('method'),
                'fields':[{'name':n.get('name'),'value':n.get('value'),'options':[{'value':v.get('value'),'text':v.get_text()} for v in n.find_all('option')]} for n in form.find_all(['input','select'])]})
        save(OUT/f'alfred_{series}_form.json',forms)

if __name__=='__main__':main()
