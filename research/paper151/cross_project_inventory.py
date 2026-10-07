"""Read-only project/drive discovery; sanitized research copies stay in TensorTrade."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import ast
import hashlib
import json
import re
import os

HERE = Path(__file__).resolve().parent
OUT = HERE / 'extension'
PF = Path(os.environ.get('TENSORTRADE_PATTERN_PROJECT', str(HERE.parents[1] / 'external/pattern-findr'))).expanduser().resolve()
QUANT = Path(os.environ.get('TENSORTRADE_QUANT_PROJECT', str(HERE.parents[1] / 'external/quant'))).expanduser().resolve()
ARCHIVE = Path(os.environ.get('TENSORTRADE_ARCHIVE', str(HERE.parents[1] / 'external/market-archive'))).expanduser().resolve()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()
                if not re.search(r'webhook|secret|password|api.?key|token', k, re.I)}
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, float) and (obj != obj or abs(obj) == float('inf')):
        return None
    return obj

def save(name, data):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize(data), indent=2, default=str, allow_nan=False) + '\n')

def snapshot_functions(source, dest, names=None):
    """Copy only pure function definitions, never app/notification initialization."""
    source = Path(source)
    text = source.read_text()
    tree = ast.parse(text)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and (names is None or n.name in names)]
    header = ('from __future__ import annotations\nimport numpy as np\nimport pandas as pd\n'
              'from scipy import stats\nfrom typing import Dict, Tuple, Optional, List, Callable\n')
    target = OUT / 'sources' / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(header + '\n\n'.join(ast.get_source_segment(text, n) for n in selected) + '\n')
    return {'source': str(source), 'source_sha256': sha(source), 'snapshot': str(target),
            'snapshot_sha256': sha(target), 'functions': [n.name for n in selected]}

def main():
    configs = []
    for p in sorted((PF/'velocity_strategies').glob('*/velocity_config.json')):
        row = {'id': p.parent.name, 'source': str(p), 'source_sha256': sha(p)}
        try:
            d = json.loads(p.read_text())
            row.update(valid=True, config=sanitize(d))
        except (ValueError, OSError) as e:
            row.update(valid=False, error=type(e).__name__)
        configs.append(row)
    save('patternfindr_configs.json', configs)
    functions = [snapshot_functions(PF/'oscillator_indicators.py', 'pf_oscillators.py',
                 {n.name for n in ast.parse((PF/'oscillator_indicators.py').read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.lineno < 306}),
                 snapshot_functions(PF/'novel_indicators.py', 'pf_novel.py',
                 {n.name for n in ast.parse((PF/'novel_indicators.py').read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.lineno < 586}),
                 snapshot_functions(PF/'velocity_trading/core/backtest_engine.py', 'pf_original_engine.py')]
    # Save the exact preserved C11 source and its pure parity module separately.
    report = QUANT/'reports/c11_trend_carry_sleeve_200k_20260621.json'
    c11 = json.loads(report.read_text())
    save('c11_frozen_params.json', {'source':str(report), 'source_sha256':sha(report),
                                    'params':c11['best']['params']})
    pine = QUANT/'pine_strategies'
    pine_rows=[]
    for p in sorted(pine.glob('*.pine')):
        content=p.read_text(errors='replace')
        pine_rows.append({'name':p.name,'source':str(p),'sha256':sha(p),
                          'strategy':bool(re.search(r'^strategy\(',content,re.M)),
                          'title':next((x[:220] for x in content.splitlines() if x.startswith('strategy(')),''),
                          'same_close_fills':'process_orders_on_close=true' in content.replace(' ',''),
                          'lookahead_on':'lookahead_on' in content})
    save('quant_pine_inventory.json',pine_rows)
    manifest=json.loads((ARCHIVE/'manifest.json').read_text())
    schemas=defaultdict(list)
    for job in manifest['jobs'].values():schemas[job['schema']].append(job)
    coverage=[]; daily=[]
    for schema,jobs in sorted(schemas.items()):
        files={f['path']:f for j in jobs for f in j.get('files',[]) if f['path'].endswith('.dbn.zst')}
        dates=sorted(set(re.search(r'glbx-mdp3-(\d{8})',p).group(1) for p in files))
        coverage.append({'schema':schema,'jobs':len(jobs),'states':dict(Counter(j['state'] for j in jobs)),
                         'manifest_files':len(files),'unique_filename_dates':len(dates),
                         'first_file_date':dates[0] if dates else None,'last_file_date':dates[-1] if dates else None,
                         'manifest_bytes':sum(f['size'] for f in files.values()),
                         'status':'manifest_inventory_not_full_integrity_or_coverage_verification'})
        if schema=='ohlcv-1d':
            for p,f in files.items():daily.append({'date':re.search(r'glbx-mdp3-(\d{8})',p).group(1),**f})
    save('archive_manifest_inventory.json',{'checked_at':datetime.now(timezone.utc).isoformat(),
        'manifest_sha256':sha(ARCHIVE/'manifest.json'),'manifest_updated_at':manifest.get('updated_at'),
        'dataset':manifest['dataset'],'readable':True,'schemas':coverage})
    save('archive_daily_files.json',sorted(daily,key=lambda x:(x['date'],x['path'])))
    save('inventory_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'patternfindr_configs':len(configs),'valid_configs':sum(x['valid'] for x in configs),
        'quant_pine_files':len(pine_rows),'functions':functions,
        'scope':'Existing projects and external source data read only. No notifications, live trading, or source mutations.'})
    print(json.dumps({'configs':len(configs),'valid':sum(x['valid'] for x in configs),
                      'pine_files':len(pine_rows),'archive_schemas':len(coverage)}))

if __name__=='__main__':main()
