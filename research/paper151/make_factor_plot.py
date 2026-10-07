"""Static figure from saved metrics; does not rerun or tune strategies."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
p=Path(__file__).parent/'factor_results'
r=pd.read_json(p/'metrics.json');r=r[(r.window=='full')&(r['case']=='base')]
names=['BuyHold','AlphaRotation','ResidualMomentumLong','ResidualMomentumLS']
labels=['SPY buy & hold','Alpha rotation','Residual momentum\nlong only','Residual momentum\nlong/short']
rows=[r[(r.strategy==n)&(r.factor_mode==('benchmark' if n=='BuyHold' else 'archived'))].iloc[0] for n in names]
fig,axes=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'wspace':.7});fig.patch.set_facecolor('#f6f8fb')
for axis,key,title in zip(axes,['return','max_drawdown'],['Cumulative return after modeled costs','Largest daily-equity drawdown']):
    vals=[100*x[key] for x in rows];axis.barh(labels,vals,color=['#243b53','#5087a0','#5087a0','#5087a0'],height=.62)
    axis.invert_yaxis();axis.set_title(title,fontsize=11,pad=16,loc='left');axis.set_facecolor('#f6f8fb')
    axis.spines[['top','right','left']].set_visible(False);axis.grid(axis='x',alpha=.18);axis.set_axisbelow(True);axis.tick_params(axis='y',length=0,labelsize=10)
    for i,v in enumerate(vals):axis.text(v+(4 if v>=0 else -4),i,f'{v:+.1f}%' if key=='return' else f'{v:.1f}%',va='center',ha='left' if v>=0 else 'right',fontsize=10)
    axis.set_xlim((-120,365) if key=='return' else (0,46));axis.set_xlabel('%')
fig.suptitle('New factor rules did not beat buy-and-hold on both measures',x=.04,y=.98,ha='left',fontsize=15,fontweight='bold')
fig.text(.04,.025,'3 Jan 2017–21 Sep 2026 • Archived factor releases • Fixed ETF adaptations • Retrospective results',fontsize=9,color='#46566b')
fig.subplots_adjust(left=.2,right=.98,bottom=.17,top=.82);fig.savefig(p/'comparison.png',dpi=180);plt.close(fig)
