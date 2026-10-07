"""Bounded read-only archive schema sample; not a coverage or liquidity assertion."""
from pathlib import Path
import json,re
from datetime import datetime,timezone
import databento as db
from cross_project_inventory import ARCHIVE,save

def main():
    m=json.loads((ARCHIVE/'manifest.json').read_text());out=[]
    for schema in ['bbo-1m','definition','statistics']:
        files={f['path']:f for j in m['jobs'].values() if j['schema']==schema for f in j.get('files',[]) if f['path'].endswith('.dbn.zst')}
        path=max(files,key=lambda p:re.search(r'glbx-mdp3-(\d{8})',p)[1])
        store=db.DBNStore.from_file(path)
        f=next(iter(store.to_df(map_symbols=False,count=20000)))
        row={'schema':schema,'path':path,'sample_rows':len(f),'columns':list(f.columns),
             'status':'First 20000 rows only; no whole-file integrity or complete coverage claim'}
        if 'instrument_class' in f:row['instrument_classes_sample']=f.instrument_class.value_counts().to_dict()
        if 'raw_symbol' in f:
            row['example_futures']=[str(x) for x in f.raw_symbol if re.fullmatch(r'(ES|NQ|CL|GC|ZN)[FGHJKMNQUVXZ]\d{1,2}',str(x))][:5]
            row['option_like_examples']=[str(x) for x in f.raw_symbol if re.search(r' [CP]\d',str(x))][:5]
        if schema=='bbo-1m':
            mp=store.symbology['mappings'];symbol_count=len(mp)
            row['metadata_symbols']=symbol_count
            row['option_like_symbols_in_metadata']=sum(bool(re.search(r' [CP]\d',s)) for s in mp)
            row['valid_quote_rows_sample']=int(((f['bid_px_00']>0)&(f['ask_px_00']>=f['bid_px_00'])&(f['bid_sz_00']>0)&(f['ask_sz_00']>0)).sum())
        out.append(row);print(schema,len(f),flush=True)
    save('quote_schema_probe.json',{'at':datetime.now(timezone.utc).isoformat(),'samples':out})

if __name__=='__main__':main()
