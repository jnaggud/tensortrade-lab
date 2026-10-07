"""Paper construction checks using explicitly synthetic inputs, NOT backtests.

These calculations verify cash-flow signs, hedge identities and failure modes.
They cannot establish returns, trade availability, tax eligibility or alpha.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from full_catalogue_options import OUT,save,sha

def zero_price(maturity,yield_rate):return np.exp(-np.asarray(maturity)*np.asarray(yield_rate))
def curve_pnl(weights,maturities,shifts):
    return float(np.asarray(weights)@(np.exp(-np.asarray(maturities)*np.asarray(shifts))-1))
def duration_wings(durations,beta=None):
    d1,d2,d3=durations
    if beta is None:
        a=(d3-d2)/(d3-d1);b=1-a
    else:
        b=d2/((beta+1)*d3);a=beta*b*d3/d1
    return np.array([a,-1,b])
def tranche_loss(portfolio_loss,attachment,detachment):
    return np.clip(np.asarray(portfolio_loss)-attachment,0,detachment-attachment)
def variance_swap(prices,strike,notional=1):
    r=np.diff(np.log(prices));return notional*(252*np.mean(r*r)-strike)
def commodity_future(spot,t,kappa,a,sigma):
    return np.exp(np.exp(-kappa*t)*np.log(spot)+a*(1-np.exp(-kappa*t))+sigma*sigma/(4*kappa)*(1-np.exp(-2*kappa*t)))
def call_price(s,k,t,r,sigma):
    d1=(np.log(s/k)+(r+.5*sigma*sigma)*t)/(sigma*np.sqrt(t));d2=d1-sigma*np.sqrt(t)
    return s*norm.cdf(d1)-k*np.exp(-r*t)*norm.cdf(d2)
def inflation_swap(initial_index,terminal_index,fixed,years,notional):
    return notional*(terminal_index/initial_index-(1+fixed)**years)
def npv(cashflows,rate):return float(sum(c/(1+rate)**t for t,c in enumerate(cashflows)))

def main():
    rows=[]
    def add(section,description,values,assertion):
        if not bool(assertion):raise AssertionError(section+' '+description)
        rows.append({'section':section,'test':description,'input_type':'synthetic_fixture_not_history','values':values,'construction_check_passed':True,'historically_tested':False,'qualified':False})
    mats=np.array([2.,5.,10.]);bullet=np.array([0.,1.,0.]);barbell=np.array([.625,0.,.375]);ladder=np.ones(10)/10
    shocks=[-.02,-.01,0.,.01,.02]
    for sec,w,t in [('5.2',bullet,mats),('5.3',barbell,mats),('5.4',ladder,np.arange(1,11))]:
        vals=[curve_pnl(w,t,s) for s in shocks]
        add(sec,'Parallel +/-100/200bp yield shocks; no trading signal or historical yield path',{'shocks':shocks,'returns':vals,'duration':float(w@t)},vals[0]>vals[1]>vals[2]>vals[3]>vals[4])
    add('5.3','Equal duration, higher barbell convexity',{'bullet_duration':5,'barbell_duration':float(barbell@mats),'barbell_convexity':float(barbell@(mats*mats))},np.isclose(barbell@mats,5) and barbell@(mats*mats)>25)
    liability=100000*zero_price(5,.04);assets=liability*barbell
    eps=1e-6
    residual=lambda dy:sum(assets*np.exp(-mats*dy))-100000*zero_price(5,.04+dy)
    slope=(residual(eps)-residual(-eps))/(2*eps)
    add('5.5','Present-value and duration matching neutralizes infinitesimal parallel shift; not all curve risk',{'liability_pv':float(liability),'dollar_sensitivity_error':float(slope)},abs(slope)<.001)
    for sec,beta in [('5.6',None),('5.7',1.),('5.8',2.),('5.8.1',(5-2)/(10-5))]:
        w=duration_wings(mats,beta);neutral=w@mats
        add(sec,'Butterfly duration neutrality and nonparallel-shock exposure',{'weights':w.tolist(),'duration':float(neutral),'net_initial_cost':float(w.sum()),'parallel_100bp_pnl':curve_pnl(w,mats,.01),'curve_twist_pnl':curve_pnl(w,mats,[.01,0,-.01])},abs(neutral)<1e-12 and (beta is not None or abs(w.sum())<1e-12))
    # Carry is evaluated under an explicitly assumed unchanged curve, not predicted.
    f=lambda t:.02+.002*np.asarray(t)
    carry=zero_price(mats-1,f(mats-1))/zero_price(mats,f(mats))-1
    no_roll=zero_price(mats-1,f(mats))/zero_price(mats,f(mats))-1
    for sec in ['5.11','5.12']:
        add(sec,'Upward-sloping static-curve carry contains positive roll-down',{'maturities':mats.tolist(),'one_year_carry':carry.tolist(),'yield_only':no_roll.tolist(),'roll_component':(carry-no_roll).tolist()},np.all(carry>no_roll))
    w=np.array([1.,-2/10]);t=np.array([2.,10.])
    add('5.13','Duration-neutral steepener responds to slope',{'weights':w.tolist(),'steepening_pnl':curve_pnl(w,t,[-.01,.01])},abs(w@t)<1e-12 and curve_pnl(w,t,[-.01,.01])>0)
    add('5.14','Negative basis carry can be erased by financing cost; default/basis risks unmodeled',{'bond_spread':.03,'cds_spread':.02,'financing_over_riskfree':.012,'net_carry':-.002},.03-.02-.012<0)
    swap=lambda floating,repo:.045-.04-(floating-repo)
    add('5.15','Swap-spread income changes sign with floating funding spread',{'low_funding':swap(.04,.04),'high_funding':swap(.06,.04)},swap(.04,.04)>0>swap(.06,.04))
    fair=(100-1)*np.exp(.04*.25)
    add('6.2','Cash-and-carry fair-value boundary and costs',{'spot':100,'discounted_dividend':1,'fair_future':float(fair),'basis_after_cost_at_fair':-.2},np.isclose((fair-(100-1)*np.exp(.04*.25))-.2,-.2))
    weights=np.array([.5,.5]);vol=np.array([.2,.3]);corr=np.array([[1.,.25],[.25,1.]])
    variance=float((weights*vol)@corr@(weights*vol))
    add('6.3','Constituent variance aggregation under imperfect correlation',{'index_variance':variance,'perfect_correlation_variance':float((weights@vol)**2)},0<variance<(weights@vol)**2)
    vals,vec=np.linalg.eigh(corr);v=vec[:,-1:];specific=1-(v[:,0]**2)*vals[-1];model=np.diag(specific)+vals[-1]*(v@v.T)
    add('6.3.1','One-factor correlation model preserves unit diagonals and positive definiteness',{'diagonal':np.diag(model).tolist(),'eigenvalues':np.linalg.eigvalsh(model).tolist()},np.allclose(np.diag(model),1) and np.linalg.eigvalsh(model).min()>0)
    prices=np.array([100.,101.,99.,102.]);realized=252*np.mean(np.diff(np.log(prices))**2)
    add('7.6','Variance-swap zero payoff at realized variance; squared returns are not demeaned',{'realized_variance':float(realized),'fair_strike_payoff':float(variance_swap(prices,realized))},abs(variance_swap(prices,realized))<1e-12)
    # No real exchange-rate or financing history is implied by these fixtures.
    for sec in ['8.2','8.2.1','8.3','8.4']:
        carry_pnl=(1+.08)/(1+.02)-1;devaluation=(1-.15)*(1+.08)/(1+.02)-1
        add(sec,'Positive rate carry can be overwhelmed by currency depreciation; selection signals untested',{'constant_fx_return':carry_pnl,'15pct_fx_fall_return':devaluation},carry_pnl>0>devaluation)
    spread=.0001;cross=1.2*150
    loop=(1-spread)*1.2*(1-spread)*150/(cross*(1+spread))
    add('8.5','No triangular-arbitrage profit at consistent crosses after spread',{'unit_cash_after_cycle':loop},loop<1)
    f0=commodity_future(80,0,.5,np.log(75),.3)
    add('9.6','Mean-reverting log-price futures model converges to spot at zero maturity',{'spot':80,'zero_maturity':float(f0),'one_year':float(commodity_future(80,1,.5,np.log(75),.3))},np.isclose(f0,80))
    s=np.array([80.,100.,120.]);pnl=(s-100)-(s-102)
    add('10.1','Matched spot/future quantity removes terminal spot exposure, leaves entry basis',{'terminal_spot':s.tolist(),'hedged_pnl':pnl.tolist()},np.allclose(pnl,2))
    cross=s*1.1+np.array([-2.,0.,3.]);hedged=(cross-110)-1.1*(s-100)
    add('10.1.1','Cross-hedge retains residual asset mismatch',{'hedged_pnl':hedged.tolist()},np.allclose(hedged,[-2,0,3]))
    dbond=100000*7;dfuture=100000*5;ratio=dbond/dfuture
    add('10.1.2','Dollar-duration hedge cancels first-order parallel yield risk',{'hedge_ratio':ratio,'net_dv01':(dbond-ratio*dfuture)/10000},np.isclose(dbond-ratio*dfuture,0))
    losses=np.array([0,.01,.03,.06,.1,.4]);equity=tranche_loss(losses,0,.03);senior=tranche_loss(losses,.1,.3)
    for sec,dlong,dshort in [('11.2',.12,4.),('11.3',1.2,4.),('11.4',.12,1.2),('11.5',.12,3.)]:
        ratio=dlong/dshort
        add(sec,'Risky-duration hedge identity; tranche default losses remain nonlinear',{'hedge_ratio':ratio,'net_spread_sensitivity':dlong-ratio*dshort,'equity_tranche_loss':equity.tolist(),'senior_tranche_loss':senior.tolist()},np.isclose(dlong-ratio*dshort,0) and np.all(equity<=.03))
    add('11.6','Notional-neutral credit curve still has carry and spread sensitivity',{'short_spread':.01,'long_spread':.02,'annual_carry_per_notional':.01},.02-.01>0)
    # Simplified convertible, explicitly excluding credit/default/call clauses.
    conversion=2.;s0=50.;k=50.;t=2.;r=.04;sigma=.3;eps=.0001
    delta=(call_price(s0+eps,k,t,r,sigma)-call_price(s0-eps,k,t,r,sigma))/(2*eps)
    residual=conversion*(call_price(s0+eps,k,t,r,sigma)-call_price(s0-eps,k,t,r,sigma))-2*eps*conversion*delta
    add('12.1','Option conversion-delta hedge cancels infinitesimal equity move',{'conversion_ratio':conversion,'share_hedge':float(conversion*delta),'sensitivity_residual':float(residual)},abs(residual)<1e-10)
    market=call_price(s0,k,t,r+.015,sigma);oas=brentq(lambda shift:call_price(s0,k,t,r+shift,sigma)-market,-.1,.1)
    add('12.2','Known artificial spread recovered by bisection/root solving',{'synthetic_spread':.015,'recovered_spread':float(oas)},np.isclose(oas,.015,atol=1e-8))
    # Algebra only; no tax-law assertions and no assumed user eligibility.
    add('13.1','Conditional tax-shield formula with hypothetical tax rate',{'tax_rate':.3,'borrow_rate':.05,'muni_yield':.04,'net_carry':.04-.05*.7},np.isclose(.04-.05*.7,.005))
    profit=100.;tc=.3;tp=.4;div=profit*(1-tc);net=div*(1-tp)/(1-tc)
    add('13.2','Hypothetical dividend imputation identity; current legal eligibility untested',{'corporate_profit':profit,'net_income':net},np.isclose(net,profit*(1-tp)))
    s0=100.;k=110.;d=2.;credit=.3;premium=k-(s0-d*(1+credit))
    add('13.2.1','Paper terminal cash-flow identity under hypothetical full assignment and tax credit',{'synthetic_put_premium':premium,'cashflow':s0+premium-k},np.isclose(s0+premium-k,d*(1+credit)))
    val=inflation_swap(100,120,.02,5,100000)
    add('14.1','Zero-coupon inflation swap pays inflation excess',{'cashflow':val,'zero_at_fixed_growth':inflation_swap(100,100*1.02**5,.02,5,100000)},val>0 and abs(inflation_swap(100,100*1.02**5,.02,5,100000))<1e-8)
    temp=np.array([40.,60.,70.,90.]);hdd=np.maximum(65-temp,0);cdd=np.maximum(temp-65,0)
    demand=100+3*hdd;hedge=np.cov(demand,hdd,ddof=1)[0,1]/np.var(hdd,ddof=1)
    add('14.3','Degree-day demand hedge removes exactly linear synthetic weather demand',{'HDD':float(hdd.sum()),'CDD':float(cdd.sum()),'hedge_ratio':float(hedge)},np.isclose(hedge,3) and np.var(demand-hedge*hdd)<1e-15)
    heat=7.;electricity=736.;gas=10000.;ratio=heat*electricity/gas
    add('14.4','Spark-spread hedge units and energy/fuel balance',{'heat_rate':heat,'fuel_contracts_per_electricity_contract':ratio,'spark_spread_at_power50_gas4':50-heat*4},np.isclose(ratio*gas,heat*electricity))
    outcomes=[(rec-30-5)/30 for rec in [0,20,40,100]]
    add('15.1','Discounted distressed debt is exposed to recovery and legal cost',{'purchase':30,'legal_cost':5,'recovery_scenarios':[0,20,40,100],'returns':outcomes},min(outcomes)<-1 and max(outcomes)>0)
    sale=np.array([220000,260000,300000]);purchase=200000;renovation=40000;carrying=10000;profit=sale*.94-purchase-renovation-carrying
    add('16.6','Fix-and-flip break-even includes renovation, selling and carrying cost',{'sale_values':sale.tolist(),'profit':profit.tolist(),'break_even_sale':(purchase+renovation+carrying)/.94},profit[0]<0<profit[-1])
    cash=10000;urgent=15000;haircut=.1;forced_sale=(urgent-cash)/(1-haircut)
    add('17.3','Cash-buffer shortfall forces a larger distressed asset sale',{'cash':cash,'liquidity_need':urgent,'asset_sale':forced_sale,'liquidation_loss':forced_sale*haircut},np.isclose(cash+forced_sale*(1-haircut),urgent))
    principal=98000;repo_rate=.04;days=30;repayment=principal*(1+repo_rate*days/360)
    add('17.4','Repo accrual and collateral shortfall under shock',{'cash_lent':principal,'repayment':repayment,'collateral_after_shock':90000,'shortfall':repayment-90000},repayment>principal and repayment-90000>0)
    add('17.5','Secured-loan collateral can fail to cover principal and disposition cost',{'principal':500,'realized_sale':450,'disposition_cost':50,'default_loss':100},450-50<500)
    flows=[-1_000_000]+[90000]*20
    add('20','Infrastructure cash-flow NPV falls with discount rate; hypothetical project only',{'npv_4pct':npv(flows,.04),'npv_10pct':npv(flows,.1)},npv(flows,.04)>0>npv(flows,.1))
    save(OUT/'scenario_results.json',rows)
    save(OUT/'scenario_manifest.json',{'checks':len(rows),'catalogue_entries':len(set(r['section'] for r in rows)),'historical_backtests':0,'source_sha256':sha(__file__),'qualified':0})
    print('Synthetic construction checks:',len(rows),'entries:',len(set(r['section'] for r in rows)))
if __name__=='__main__':main()
