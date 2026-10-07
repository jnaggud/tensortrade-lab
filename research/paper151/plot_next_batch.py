"""Standalone research figures. Diagnostic and qualified results stay distinct."""
from intraday_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    btc=[r for r in json.loads((OUT/'btc_results.json').read_text()) if r['block']=='full' and r['case']=='base']
    zero={r['strategy']:r for r in json.loads((OUT/'btc_zero_cost_results.json').read_text())}
    labels=['V15 algorithmic','V15 manual','V2','ARWO regime','Any reversal','Reversal, 0.37% stop','Cross/zone, 0.97% stop','Quant reversal']
    fig,ax=plt.subplots(figsize=(11,6.7));y=np.arange(len(btc))
    ax.barh(y-.18,[100*r['total_return'] for r in btc],height=.34,color='#a33439',label='After modeled costs')
    ax.barh(y+.18,[100*zero[r['strategy']]['total_return'] for r in btc],height=.34,color='#5c9bb4',label='Zero-cost diagnostic')
    ax.axvline(100*btc[0]['benchmark_return'],color='#245347',ls='--',lw=2,label='Buy and hold after costs')
    ax.axvline(0,color='#aaa',lw=.8);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(-108,67)
    ax.set_xlabel('Total return (%)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right',fontsize=9)
    fig.suptitle('Bitcoin intraday rules: trading costs erase the apparent edge',x=.02,ha='left',fontsize=16,fontweight='bold')
    fig.text(.02,.925,'BITSTAMP 15-minute bars · March 2024–May 2026 · Saved-rule adaptations, without a new parameter search',fontsize=10)
    fig.text(.02,.035,'Base costs: 0.10% fee + 0.05% slippage each side; modeled borrowing for quant shorts.\nZero-cost runs are diagnostics, with slightly different trading paths. None is an execution-qualified winner.',fontsize=9,color='#444')
    fig.tight_layout(rect=(0,.09,1,.90));fig.savefig(OUT/'btc_cost_comparison.png',dpi=180);plt.close(fig)
    options=[r for r in json.loads((OUT/'options_pilot_results.json').read_text()) if r['case']=='base']
    fig,ax=plt.subplots(figsize=(9.5,4.9))
    labels=['Covered call','Cash-secured put','Passive ES future'];values=[100*r['total_return'] for r in options]+[100*options[0]['benchmark_return']]
    ax.barh(labels,values,color=['#387e96','#387e96','#7c7c7c']);ax.invert_yaxis();ax.axvline(0,color='#aaa',lw=.8);ax.set_xlim(-1,.04)
    for i,v in enumerate(values):ax.text(v+.02,i,f'{v:.3f}%',va='center',color='white',fontsize=11)
    ax.set_xlabel('Total account return (%)');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Options pilot: smaller losses over one week',x=.02,ha='left',fontsize=16,fontweight='bold')
    fig.text(.02,.91,'June 1–5 2026 · One contract per rule · $1 million cash account · Observed bid/ask and commissions',fontsize=10)
    fig.text(.02,.035,'One episode per construction is not evidence of a persistent edge. Large cash reserves dilute returns.\nQuotes cover daytime snapshots only; one put valuation is missing. Pilot closes before European expiry.',fontsize=9,color='#444')
    fig.tight_layout(rect=(0,.10,1,.88));fig.savefig(OUT/'options_pilot_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
