from pathlib import Path
import json,hashlib,shutil
HERE=Path(__file__).parent;dest=HERE/'tradingview/promoted';backup=HERE/'tradingview/pre_distribution_fix'
backup.mkdir(exist_ok=True)
old='request.dividends(syminfo.tickerid,dividends.gross,gaps=barmerge.gaps_on,lookahead=barmerge.lookahead_off,ignore_invalid_symbol=true)'
def fix(s):
    for lhs,indent in [('distribution','    '),('div','')]:
        a=f'{indent}{lhs}=nz({old},0.0)'
        b=f'''{indent}// Derive cash distributions on the same split-adjusted share basis as close.
{indent}// Adjacent adjustment-factor ratios cancel any common later rescaling.
{indent}adjustedClose=request.security(ticker.modify(syminfo.tickerid,adjustment=adjustment.dividends),timeframe.period,close,gaps=barmerge.gaps_on,lookahead=barmerge.lookahead_off)
{indent}priceFactor=close/adjustedClose
{indent}impliedDistribution=na(priceFactor[1])?0.0:close[1]*(1.0-priceFactor/priceFactor[1])
{indent}{lhs}=math.abs(impliedDistribution)<0.000001?0.0:impliedDistribution'''
        s=s.replace(a,b)
    return s
for file in list(dest.glob('*.pine'))+[HERE/'promote.py']:
    src=file.read_text();updated=fix(src)
    if src!=updated:
        if not (backup/file.name).exists():shutil.copy2(file,backup/file.name)
        file.write_text(updated)
manifest=dest/'manifest.json'
if not (backup/'manifest.json').exists():shutil.copy2(manifest,backup/'manifest.json')
rows=json.loads(manifest.read_text())
for row in rows:
    row['sha256']=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
    row.update(compile_status='pending_revalidation',account_save_status='pending_update',saved_source_verified=False)
    row.pop('runtime_reconciliation',None)
manifest.write_text(json.dumps(rows,indent=2))
print('Corrected distribution basis in generator and ten ports; original sources retained.')
