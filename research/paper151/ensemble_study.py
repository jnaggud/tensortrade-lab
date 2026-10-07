"""Family-balanced, causal target ensembles using the real shared price ledger.

The old books are shadow experts, not future-selected winners. The new account
trades the blend at the NEXT open and pays its own costs. No frozen files edited.
"""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import json,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(HERE)]
from full_catalogue_options import save,sha
from run_study import read_frame,panel_from,SECTORS
from tensortrade_lab.portfolio import Ledger,performance

OUT=HERE/'ensemble'
FAMILIES={'Trend':['SMA200','MA50_200','MA20_50_200','Momentum252','Donchian55_20'],
 'Reversal':['ChannelBounce20','PivotBounce','IBS'], 'RiskSizing':['Vol12','TrendVol12'],
 'PaperML':['KNN','ANN']}
CASES=[('base',1,1),('double_cost',2,1),('extra_bar_delay',1,2)]
METHODS=['EqualFamilies','InverseVol','PastSharpe','RidgeAllocator','TensorTradePPO','EqualPlusPPO','WithoutPatternFindr']

def register():
    OUT.mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():return
    save(OUT/'protocol.json',{
      'registered_at':datetime.now(timezone.utc).isoformat(),
      'markets':['US_ETF_portfolio','BTC_daily'],'evaluation':'2020 onward, each calendar year trained only through preceding completed year. Prior 2017-2019 for initial training. Existing history was already explored: retrospective walk-forward, NOT an untouched holdout.',
      'experts':'All available saved paper shadow books in the selected family/market and all supported saved daily Pattern_FindR books. Group similar recipes before blending. No ranking by full-sample returns.',
      'target_adapter':'At completed close, recover each shadow book shares*current price/NAV, then average within each family. Trade blended long-only targets at next open. Shadows keep their own historical state; voting one day after their holdings were observed is a NEW strategy, not exact execution of their intrabar orders. Combined account has no borrowed shares or leverage.',
      'methods':METHODS,'families':FAMILIES,'cost_stresses':CASES,
      'execution':'Use unchanged self-financing Ledger. Stocks: 5bp commission+5bp slippage per side; BTC:10bp+5bp. Each case has matched passive and cash. Rebalance once weekly after Friday close (BTC Sunday UTC close), and at annual model refit. Stress retains frozen target stream but executes a bar later or at doubled costs. Idle ETF cash uses prior-available bill yield; BTC cash zero.',
      'past_weights':'Inverse volatility: trailing126-session shadow family net returns, floor1% annual vol. PastSharpe: softmax of clipped annualized trailing126 mean/std, temperature1, with cash score0. Ridge forecasts21-session family return with alpha10; scaling fit training rows only; labels must mature strictly before the annual fold.',
      'ppo':{'framework':'TensorTrade TradingEnv + stable-baselines3 PPO','timesteps':16384,'seeds':[11,29,47], 'hidden':[32,32], 'n_steps':256,'batch_size':64,'n_epochs':5,'learning_rate':0.0003,'gamma':0.99,'entropy':0.01,'selection':'Last fixed-budget checkpoint from EVERY seed; average seed weights. No checkpoint, budget, hyperparameter or best-seed selection. Fresh expanding-window fit each year.'},
      'gate':'After-cost return strictly higher and maximum daily drawdown no worse than matched buy-and-hold, in full sample AND each calendar year AND cost/delay stresses. Also compare cash. Historical gate alone never qualifies execution or future profitability.',
      'scope_limits':'This target experiment covers compatible daily stock/ETF and BTC books. Futures/15m quant books are evaluated separately as capital-sleeve diagnostics; unavailable-data, synthetic, noncausal and unimplemented candidates cannot enter. Research choices/old expert parameters themselves were selected retrospectively.',
    })

def load_market(market):
    crypto=market=='BTC_daily'
    symbols=['BTC'] if crypto else ['SPY','GLD','TLT','IEF','EFA','EEM']+SECTORS
    frames={};audit=[]
    for s in symbols:
        path='research/paper151/data/BTCUSD_yahoo_20260929.parquet' if crypto else f'data/rotation/{s}.parquet'
        f,m=read_frame(path);frames[s]=f;audit.append({'kind':'prices','symbol':s,'path':str(ROOT/path),'sha256':m['sha256']})
    start=pd.Timestamp('2016-12-30',tz='UTC');end=min(f.timestamp.max() for f in frames.values())
    if crypto:start=pd.Timestamp('2016-12-31',tz='UTC')
    # The prior study already documented absent Sep22 observations in this universe.
    # Use its last COMPLETE common session, never drop interior missing observations.
    if not crypto:end=min(end,pd.Timestamp('2026-09-22',tz='UTC'))
    for s,f in frames.items():frames[s]=f[(f.timestamp>=start)&(f.timestamp<=end)].reset_index(drop=True)
    p=panel_from(frames,cash_interest=not crypto,periods=365 if crypto else 252)
    # All source ledgers are complete-close observations; timestamps must match exactly.
    grouped={};coverage=[]
    def add(path,family):
        h=pd.read_parquet(path).set_index('timestamp')
        if h.index.duplicated().any():raise ValueError('Duplicate shadow marks')
        h=h.reindex(p.bar_end)
        if h.equity.isna().any():raise ValueError('Missing shadow marks '+str(path))
        if (h.equity<=0).any():raise ValueError('Insolvent shadow is not an investable expert '+str(path))
        w=np.zeros((len(h),len(symbols)))
        for k,s in enumerate(symbols):
            col='units_'+('BTC-USD' if crypto and 'units_BTC-USD' in h else s)
            if col in h:w[:,k]=h[col].to_numpy()*p.close[:,k]/h.equity.to_numpy()
        if (w<-1e-9).any() or (w.sum(axis=1)>1+1e-7).any():raise ValueError('Invalid expert allocation')
        r=h.equity.pct_change().fillna(0).to_numpy()
        grouped.setdefault(family,[]).append((w,r))
        row={'kind':'shadow_book','name':path.stem,'family':family,'path':str(path),'sha256':sha(path)}
        coverage.append(row);audit.append(row)
    for s in (['BTC'] if crypto else ['SPY','GLD','TLT','EFA']):
        for family,rules in FAMILIES.items():
            for rule in rules:
                path=HERE/'expanded_results/ledgers'/f'{s}__{rule}.parquet'
                if path.exists():add(path,family)
    if not crypto:
        for path in sorted((HERE/'expanded_results/ledgers').glob('Sectors__*.parquet')):
            if 'Passive' not in path.name and 'BuyHold' not in path.name:add(path,'SectorAllocation')
        for path in sorted((HERE/'expanded_results/ledgers').glob('MultiAsset__*.parquet')):
            if 'Passive' not in path.name and 'BuyHold' not in path.name:add(path,'MultiAssetAllocation')
    configs=json.loads((HERE/'extension/patternfindr_configs.json').read_text())
    for row in configs:
        d=row.get('config',{})
        if d.get('ticker')!=('BTC-USD' if crypto else 'SPY') or d.get('interval')!='1d':continue
        path=HERE/'extension/velocity_ledgers'/(row['id']+'.parquet')
        if path.exists():add(path,'PatternFindr')
    names=list(grouped)
    targets=np.stack([np.mean([z[0] for z in grouped[k]],axis=0) for k in names],axis=1)
    returns=np.column_stack([np.mean([z[1] for z in grouped[k]],axis=0) for k in names])
    save(OUT/market/'sources.json',audit);save(OUT/market/'coverage.json',coverage)
    return p,targets,returns,names,crypto

def features(p,targets,returns):
    r=pd.DataFrame(returns)
    xs=[r.rolling(k).sum().to_numpy() for k in (21,63,126)]
    xs += [r.rolling(63).std().to_numpy(),targets.sum(axis=2)]
    bench=pd.Series(p.total_index[:,0])
    xs += [np.column_stack([bench.pct_change(k) for k in (21,63,126)]),p.volatility[:,[0]]]
    return np.nan_to_num(np.column_stack(xs),nan=0,posinf=0,neginf=0)

def softmax(x):
    x=np.asarray(x);v=np.exp(x-np.max(x,axis=-1,keepdims=True));return v/v.sum(axis=-1,keepdims=True)

def simple_weights(returns,j,method):
    k=returns.shape[1]
    if method=='EqualFamilies':return np.r_[np.full(k,1/k),0.]
    x=returns[max(0,j-125):j+1]
    sd=np.maximum(x.std(axis=0),.01/np.sqrt(252))
    if method=='InverseVol':
        v=1/sd;return np.r_[v/v.sum(),0.]
    return softmax(np.r_[np.clip(x.mean(axis=0)/sd*np.sqrt(252),-3,3),0.])

def mature_labels(returns,horizon=21):
    # At feature row i, label covers i+1..i+21; unavailable until close i+21.
    out=np.full_like(returns,np.nan)
    for i in range(len(returns)-horizon):out[i]=np.prod(1+returns[i+1:i+1+horizon],axis=0)-1
    return out

def fit_ridge(x,returns,cut):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    y=mature_labels(returns[:cut])
    ix=np.arange(126,cut-21)
    if len(ix)<126:raise ValueError('Insufficient matured training labels')
    scale=StandardScaler().fit(x[ix]);model=Ridge(alpha=10.).fit(scale.transform(x[ix]),y[ix])
    return scale,model,{'training_rows':len(ix),'max_feature_index':int(ix[-1]),'max_label_index':int(ix[-1]+21),'cut':cut}

def make_env(p,targets,x,start,end,mean,scale,fee,slip):
    import gymnasium as gym
    from gymnasium.spaces import Box
    from tensortrade.env.generic import ActionScheme,Observer,TradingEnv
    from tensortrade.env.default.renderers import EmptyRenderer
    from tensortrade_lab.portfolio_env import PortfolioReward,PortfolioStopper,PortfolioInfo
    state=Ledger(p,start,end,commission=fee,slippage=slip)
    class Allocate(ActionScheme):
        @property
        def action_space(self):return Box(-5,5,(targets.shape[1]+1,),dtype=np.float32)
        def perform(self,env,action):
            j=state.index
            # Fixed weekday known without looking at a future bar.
            weekday=6 if len(p.symbols)==1 else 4
            scheduled=j==start-1 or p.timestamp[j].weekday()==weekday
            state.step(softmax(action)[:-1]@targets[j] if scheduled else None)
            state.reward_value*=100
        def reset(self):state.reset()
    class Observe(Observer):
        @property
        def observation_space(self):return Box(-10,10,(x.shape[1],),dtype=np.float32)
        def observe(self,env):return np.clip((x[state.index]-mean)/scale,-10,10).astype(np.float32)
        def reset(self,random_start=0):pass
    class Env(TradingEnv):
        def reset(self,*,seed=None,options=None):
            gym.Env.reset(self,seed=seed);return super().reset(seed=seed,options=options)
    env=Env(action_scheme=Allocate(),observer=Observe(),reward_scheme=PortfolioReward(state),
      stopper=PortfolioStopper(state),informer=PortfolioInfo(state),renderer=EmptyRenderer())
    env.state=state
    return env

def train_fold(market,year,p,targets,x,returns,indices,crypto):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    start=int(indices[0]);cut=start-1 # Strictly before the first decision close, conservative embargo.
    train=replace(p,**{key:getattr(p,key)[:cut] for key in p.__dataclass_fields__ if key!='symbols'})
    xx=x[:cut];mean=xx[126:].mean(axis=0);scale=np.maximum(xx[126:].std(axis=0),1e-6)
    folder=OUT/market/str(year);folder.mkdir(parents=True,exist_ok=True)
    pred=[];fits=[]
    for seed in [11,29,47]:
        dest=folder/f'ppo_{seed}.zip';meta=dest.with_suffix('.json')
        env=make_env(train,targets[:cut],xx,127,cut-1,mean,scale,.001 if crypto else .0005,.0005)
        identity={'market':market,'year':year,'seed':seed,'train_end':str(train.bar_end[-1]),'source_identity':sha(OUT/market/'sources.json'),'protocol_sha256':sha(OUT/'protocol.json')}
        if dest.exists():
            prior=json.loads(meta.read_text())
            if prior['identity']!=identity or prior['model_sha256']!=sha(dest):raise ValueError('Frozen PPO identity changed')
            model=PPO.load(dest,env=env,device='cpu')
        else:
            model=PPO('MlpPolicy',env,seed=seed,device='cpu',n_steps=256,batch_size=64,n_epochs=5,
                learning_rate=.0003,gamma=.99,ent_coef=.01,policy_kwargs={'net_arch':[32,32]},verbose=0)
            model.learn(total_timesteps=16384);model.save(dest)
            save(meta,{'identity':identity,'actual_timesteps':model.num_timesteps,'model_sha256':sha(dest),'scaler':{'mean':mean.tolist(),'scale':scale.tolist()}})
        action,_=model.predict(np.clip((x[indices-1]-mean)/scale,-10,10).astype(np.float32),deterministic=True)
        pred.append(softmax(action));fits.append(identity)
        env.close()
    scaler,model,audit=fit_ridge(x,returns,cut)
    rr=model.predict(scaler.transform(x[indices-1]))
    # A fixed five percentage-point 21-session forecast scale, not tuned per fold.
    ridge=softmax(np.column_stack([np.clip(rr/.05,-3,3),np.zeros(len(rr))]))
    save(folder/'training_audit.json',{'ppo':fits,'ridge':audit,'first_decision':str(p.bar_end[start-1]),'trained_strictly_before':str(p.bar_end[cut-1]),'feature_mean':mean.tolist(),'feature_scale':scale.tolist(),'ridge_coefficients':model.coef_.tolist(),'ridge_intercept':model.intercept_.tolist()})
    return np.mean(pred,axis=0),ridge

def execute(p,stream,start,end,fee,slip,delay=1,kind='candidate'):
    led=Ledger(p,start,end,commission=fee,slippage=slip)
    for t in range(start,end+1):
        w=None
        if kind=='benchmark' and t==start:w=np.eye(len(p.symbols))[0]
        elif kind=='cash' and t==start:w=np.zeros(len(p.symbols))
        elif kind=='candidate':
            j=t-delay
            if j>=0 and np.isfinite(stream[j]).all():w=stream[j]
        led.step(w,drip=kind=='benchmark')
    return led

def run_market(market):
    register();p,targets,returns,names,crypto=load_market(market)
    x=features(p,targets,returns);k=len(names)
    save(OUT/market/'family_names.json',names)
    first=int(np.flatnonzero(p.timestamp>=pd.Timestamp('2020-01-01',tz='UTC'))[0])
    streams={m:np.full((len(x),len(p.symbols)),np.nan) for m in METHODS}
    weights=[]
    for year in range(2020,2027):
        ix=np.flatnonzero(p.timestamp.year==year)
        if len(ix)==0:continue
        print(market,'fit',year,'rows',int(ix[0]),flush=True)
        ppo,ridge=train_fold(market,year,p,targets,x,returns,ix,crypto)
        for a,t in enumerate(ix):
            j=t-1
            if t!=ix[0] and p.timestamp[j].weekday()!=(6 if crypto else 4):continue
            allw={m:simple_weights(returns,j,m) for m in ['EqualFamilies','InverseVol','PastSharpe']}
            allw['RidgeAllocator']=ridge[a];allw['TensorTradePPO']=ppo[a]
            allw['EqualPlusPPO']=(allw['EqualFamilies']+ppo[a])/2
            z=allw['EqualFamilies'].copy();z[names.index('PatternFindr')]=0;z/=z.sum();allw['WithoutPatternFindr']=z
            for m,w in allw.items():
                streams[m][j]=w[:-1]@targets[j]
                weights.append({'decision_at':str(p.bar_end[j]),'method':m,'family_weights':w.tolist(),'target':streams[m][j].tolist()})
    np.savez_compressed(OUT/market/'targets.npz',**streams)
    save(OUT/market/'decisions.json',weights)
    rows=[]
    periods=[('full',first,len(x)-1)]+[(str(y),int(ix[0]),int(ix[-1])) for y in range(2020,2027) if len(ix:=np.flatnonzero(p.timestamp.year==y))>1]
    for block,start,end in periods:
        for case,mult,delay in CASES:
            fee=(.001 if crypto else .0005)*mult;slip=.0005*mult
            b=execute(p,None,start,end,fee,slip,kind='benchmark');bm=performance(b)
            cash=performance(execute(p,None,start,end,fee,slip,kind='cash'))
            for method,stream in streams.items():
                led=execute(p,stream,start,end,fee,slip,delay);m=performance(led)
                rows.append({'market':market,'method':method,'block':block,'case':case,**m,'benchmark_return':bm['total_return'],'benchmark_drawdown':bm['max_drawdown'],'cash_return':cash['total_return'],'pass':bool(m['total_return']>max(bm['total_return'],cash['total_return']) and m['max_drawdown']<=bm['max_drawdown'])})
                if block=='full':
                    dest=OUT/market/'ledgers';dest.mkdir(exist_ok=True)
                    pd.DataFrame(led.history).to_parquet(dest/f'{method}_{case}.parquet',index=False);save(dest/f'{method}_{case}_fills.json',led.fills)
            if block=='full':
                pd.DataFrame(b.history).to_parquet(OUT/market/'ledgers'/f'BuyHold_{case}.parquet',index=False);save(OUT/market/'ledgers'/f'BuyHold_{case}_fills.json',b.fills)
    save(OUT/market/'results.json',rows)
    save(OUT/market/'manifest.json',{'at':datetime.now(timezone.utc).isoformat(),'candidate_cases':len(rows),'families':len(names),'shadow_books':len(json.loads((OUT/market/'coverage.json').read_text())),'ppo_fits':21,'ppo_steps':21*16384,'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__),'source_sha256':sha(OUT/market/'sources.json'),'all_gates_pass':[m for m in METHODS if all(r['pass'] for r in rows if r['method']==m)],'execution_qualified':False})
    print(market,'complete',flush=True)

if __name__=='__main__':run_market(sys.argv[1])
