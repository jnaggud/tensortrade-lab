"""Generate the fixed-ranking research shortlist only after local results exist."""
from pathlib import Path
import json
import hashlib

HERE=Path(__file__).parent
DEST=HERE/'tradingview/promoted'
SECTORS=['AMEX:XLB','AMEX:XLE','AMEX:XLF','AMEX:XLI','AMEX:XLK','AMEX:XLP','AMEX:XLU','AMEX:XLV','AMEX:XLY','AMEX:SPY','NASDAQ:IEF']

def native(name,title):
    return f'''//@version=6
strategy("{title}", overlay=true, initial_capital=10000, currency=currency.USD, default_qty_type=strategy.percent_of_equity, default_qty_value=99.8, commission_type=strategy.commission.percent, commission_value=0.10, slippage=0, pyramiding=0, margin_long=100, margin_short=100, process_orders_on_close=false, calc_on_every_tick=false, calc_on_order_fills=false)
// EXPERIMENTAL. Selected by the declared local cross-market ranking, not a proven winner.
// Fixed rule: {name}. Local results and failures: research/paper151/expanded_results.
// Use standard DAILY candles with dividend adjustment OFF (splits allowed).
// 0.10% fee approximates US 5bps fee + 5bps slippage; set 0.15% for BTC.
// Pine slippage is ticks, so this proxy is not identical to the local ledger.
// No idle-cash interest or automatic dividend reinvestment in this emulator.
// Exact local-vs-Pine PnL reconciliation remains required; signals use completed bars.
start=input.time(timestamp("30 Dec 2016 21:00 +0000"),"First eligible signal close (US sample default)")
stop=input.time(timestamp("01 Jan 2030 00:00 +0000"),"Stop sending entry signals")
if not chart.is_standard or not timeframe.isdaily or timeframe.multiplier!=1
    runtime.error("Use standard 1-day candles.")
ma20=ta.sma(close,20)
ma50=ta.sma(close,50)
ma200=ta.sma(close,200)
hi55=ta.highest(high,55)[1]
lo20=ta.lowest(low,20)[1]
hi20=ta.highest(high,20)[1]
ibs=high>low?(close-low)/(high-low):0.5
// Derive cash distributions on the same split-adjusted share basis as close.
// Adjacent adjustment-factor ratios cancel any common later rescaling.
adjustedClose=request.security(ticker.modify(syminfo.tickerid,adjustment=adjustment.dividends),timeframe.period,close,gaps=barmerge.gaps_on,lookahead=barmerge.lookahead_off)
priceFactor=close/adjustedClose
impliedDistribution=na(priceFactor[1])?0.0:close[1]*(1.0-priceFactor/priceFactor[1])
div=math.abs(impliedDistribution)<0.000001?0.0:impliedDistribution
trReturn=na(close[1])?0.0:(close+div)/close[1]-1.0
logTR=ta.cum(math.log(1.0+trReturn))
mom=logTR>logTR[252]
var bool channelTrend=false
if close>hi55
    channelTrend:=true
else if close<lo20
    channelTrend:=false
var int entryBar=na
if strategy.position_size>0 and strategy.position_size[1]==0
    entryBar:=bar_index
if strategy.position_size==0
    entryBar:=na
held=na(entryBar)?0:bar_index-entryBar+1
bool enter=false
bool leave=false
''' + {
 'MA50_200':'enter:=ma50>ma200\nleave:=ma50<=ma200\n',
 'MA20_50_200':'enter:=ma20>ma50 and ma50>ma200\nleave:=not enter\n',
 'AlphaCombo':'votes=(close>ma200?1:0)+(mom?1:0)+(channelTrend?1:0)\nenter:=votes>=2\nleave:=votes<2\n',
 'IBS':'enter:=ibs<0.2\nleave:=ibs>0.8 or held>=5\n',
 'SMA200':'enter:=close>ma200\nleave:=close<=ma200\n',
 'Momentum252':'enter:=mom\nleave:=not mom\n',
 'Donchian55_20':'enter:=close>hi55\nleave:=close<lo20\n',
 'ChannelBounce20':'enter:=low<=lo20 and close>lo20\nleave:=close>=hi20 or held>=5\n',
 }[name]+'''
eligible=time_close>=start and time_close<stop and not na(ma200) and not na(logTR[252])
if barstate.isconfirmed
    if strategy.position_size>0 and (leave or time_close>=stop)
        strategy.close_all(comment="Next-open exit")
    else if strategy.position_size==0 and eligible and enter
        strategy.entry("Long",strategy.long)
plot(ma50,"SMA50",color=color.aqua)
plot(ma200,"SMA200",color=color.orange)
bgcolor(strategy.position_size>0?color.new(color.teal,91):na)
var table note=table.new(position.bottom_right,1,2)
if barstate.islastconfirmedhistory or barstate.isrealtime
    table.cell(note,0,0,"EXPLORATORY — not broadly qualified",text_color=color.orange,bgcolor=color.new(color.black,10))
    table.cell(note,0,1,"Local study is authoritative; reconcile this emulator",text_color=color.white,bgcolor=color.new(color.black,10))
'''

def portfolio(name,title):
    supported={'GoldDiversification','SectorLowVol','SectorRotationMA','SectorMomentumSkip21','SectorRotation','SectorRotationDual'}
    if name not in supported:
        raise ValueError(f'No reviewed portfolio Pine port for {name}; implement and verify its rules before promotion.')
    syms=['AMEX:SPY','AMEX:GLD'] if name=='GoldDiversification' else SECTORS
    n=len(syms)
    names=', '.join('"'+s+'"' for s in syms)
    source=f'''//@version=6
indicator("{title}", overlay=false, max_bars_back=300, dynamic_requests=true)
// Portfolio SIGNAL indicator: NOT a native Strategy Tester portfolio backtest.
// Local engine accounts for simultaneous legs, costs, distributions and cash.
// Shows target weights at the current bar open, based entirely on preceding bars.
// Between rebalances these are the last target weights, not drifted live weights.
// Fixed rule: {name}. Exploratory shortlist; no broad economic qualification.
start=input.time(timestamp("01 Jan 2017 00:00 +0000"),"First eligible execution day")
if not chart.is_standard or not timeframe.isdaily or timeframe.multiplier!=1
    runtime.error("Use a standard 1-day US equity/ETF chart.")
f_stats() =>
    // Derive cash distributions on the same split-adjusted share basis as close.
    // Adjacent adjustment-factor ratios cancel any common later rescaling.
    adjustedClose=request.security(ticker.modify(syminfo.tickerid,adjustment=adjustment.dividends),timeframe.period,close,gaps=barmerge.gaps_on,lookahead=barmerge.lookahead_off)
    priceFactor=close/adjustedClose
    impliedDistribution=na(priceFactor[1])?0.0:close[1]*(1.0-priceFactor/priceFactor[1])
    distribution=math.abs(impliedDistribution)<0.000001?0.0:impliedDistribution
    r=na(close[1])?0.0:(close+distribution)/close[1]-1.0
    logTR=ta.cum(math.log(1.0+r))
    strength=high>low?(close-low)/(high-low):0.5
    [close[1],ta.sma(close,200)[1],math.exp(logTR[1]-logTR[253])-1.0,math.exp(logTR[22]-logTR[253])-1.0,ta.stdev(r,63,false)[1]*math.sqrt(252.0),strength[1],r[1]]
var symbols=array.from({names})
var weights=array.new_float({n},0.0)
var bool started=false
prices=array.new_float({n},na)
mas=array.new_float({n},na)
mom=array.new_float({n},na)
skipMom=array.new_float({n},na)
vols=array.new_float({n},na)
strengths=array.new_float({n},na)
returns=array.new_float({n},na)
bool ready=true
for i=0 to {n-1}
    [p,m,v,s,vol,b,r]=request.security(array.get(symbols,i),timeframe.period,f_stats(),gaps=barmerge.gaps_on,lookahead=barmerge.lookahead_off)
    array.set(prices,i,p)
    array.set(mas,i,m)
    array.set(mom,i,v)
    array.set(skipMom,i,s)
    array.set(vols,i,vol)
    array.set(strengths,i,b)
    array.set(returns,i,r)
    ready:=ready and not na(v) and not na(m) and not na(vol)
rebalance=ready and time>=start and (timeframe.change("M") or not started)
if rebalance
    array.fill(weights,0.0)
    started:=true
'''
    # Each security expression is an unconditional call site. Do not execute
    # history-dependent functions through a loop-scoped function call.
    a=source.index('for i=0 to ')
    b=source.index('rebalance=ready',a)
    calls=[]
    for i,symbol in enumerate(syms):
        calls.append(f'[p{i},m{i},v{i},s{i},vol{i},b{i},r{i}]=request.security("{symbol}",timeframe.period,f_stats(),gaps=barmerge.gaps_on,lookahead=barmerge.lookahead_off)\n')
        for array,var in [('prices','p'),('mas','m'),('mom','v'),('skipMom','s'),('vols','vol'),('strengths','b'),('returns','r')]:
            calls.append(f'array.set({array},{i},{var}{i})\n')
        calls.append(f'ready:=ready and not na(v{i}) and not na(m{i}) and not na(vol{i})\n')
    source=source[:a]+''.join(calls)+source[b:]
    if name=='GoldDiversification':
        source+='    array.set(weights,0,0.8)\n    array.set(weights,1,0.2)\n'
    else:
        if name=='SectorRotationDual':
            source+='    defensive=array.get(prices,9)<=array.get(mas,9)\n    if defensive\n        array.set(weights,10,1.0)\n'
            indent='        ';source+='    else\n'
        else:indent='    '
        expression={'SectorLowVol':'-array.get(vols,i)','SectorMomentumSkip21':'array.get(skipMom,i)'}.get(name,'array.get(mom,i)')
        source+=indent+'scores=array.new_float(9,na)\n'+indent+'for i=0 to 8\n'+indent+'    array.set(scores,i,'+expression+')\n'
        source+=indent+'for selection=0 to 2\n'+indent+'    best=-1.0e30\n'+indent+'    chosen=0\n'+indent+'    for i=0 to 8\n'+indent+'        if array.get(scores,i)>best\n'+indent+'            best:=array.get(scores,i)\n'+indent+'            chosen:=i\n'
        if name=='SectorRotationMA':
            source+=indent+'    if array.get(prices,chosen)>array.get(mas,chosen)\n'+indent+'        array.set(weights,chosen,1.0/3.0)\n'
        else:source+=indent+'    array.set(weights,chosen,1.0/3.0)\n'
        source+=indent+'    array.set(scores,chosen,-1.0e30)\n'
    colors=['color.aqua','color.orange','color.lime','color.fuchsia','color.yellow','color.blue','color.red','color.teal','color.purple','color.white','color.silver']
    for i,s in enumerate(syms):source+=f'plot(started?array.get(weights,{i})*100:na,"{s.split(":")[1]} target %",color={colors[i]},style=plot.style_stepline)\n'
    source+=f'''plot(started?(1.0-array.sum(weights))*100:na,"Cash target %",color=color.gray,style=plot.style_stepline)
plotshape(rebalance,title="Rebalance using preceding close",style=shape.diamond,location=location.bottom,color=color.white,size=size.tiny)
var table dashboard=table.new(position.top_right,2,{n+3},border_width=1)
if barstate.islast
    table.cell(dashboard,0,0,"EXPLORATORY TARGETS",text_color=color.orange)
    table.cell(dashboard,1,0,"No portfolio backtest",text_color=color.orange)
    for i=0 to {n-1}
        table.cell(dashboard,0,i+1,array.get(symbols,i),text_color=color.white)
        table.cell(dashboard,1,i+1,started?str.tostring(100*array.get(weights,i),"#.##")+"%":"Waiting",text_color=color.white)
    table.cell(dashboard,0,{n+1},"Cash",text_color=color.white)
    table.cell(dashboard,1,{n+1},started?str.tostring(100*(1-array.sum(weights)),"#.##")+"%":"Waiting",text_color=color.white)
    table.cell(dashboard,0,{n+2},"Data",text_color=color.white)
    table.cell(dashboard,1,{n+2},ready?"Complete prior bars":"Missing / warming up",text_color=ready?color.lime:color.red)
'''
    return source

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    old_path=DEST/'manifest.json'
    old={m['title']:m for m in json.loads(old_path.read_text())} if old_path.exists() else {}
    selected=json.loads((HERE/'expanded_results/promotion.json').read_text());manifest=[]
    for n,c in enumerate(selected,1):
        name=c['strategy'];title=f'151 #{n:02d} {name} | Research'
        source=native(name,title) if c['pine_kind']=='native_strategy' else portfolio(name,title)
        path=DEST/f'{n:02d}_{name}.pine';path.write_text(source)
        entry={**c,'shortlist_rank':n,'title':title,'path':str(path),'sha256':hashlib.sha256(source.encode()).hexdigest(),'compile_status':'pending','account_save_status':'pending','economic_status':'exploratory; failed broad qualification'}
        previous=old.get(title,{})
        if previous.get('sha256')==entry['sha256']:
            for key in ('compile_status','account_save_status','compiler_errors','compiler_warnings','saved_script_id','saved_version','saved_source_verified','runtime_reconciliation','distribution_fix'):
                if key in previous:entry[key]=previous[key]
        manifest.append(entry)
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps([{'name':m['strategy'],'kind':m['pine_kind'],'path':m['path']} for m in manifest],indent=2))

if __name__=='__main__':main()
