"""Contract-level research ledger. Synthetic validation is not a historical backtest.

USD only; integer contracts; synchronous bid/ask execution; no portfolio-margin
netting. American exercise/assignment must be provided as explicit dated events.
No stock borrow, FX conversion, futures delivery or options-on-futures delivery.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from copy import deepcopy
import math

class DataError(ValueError): pass
class CollateralError(ValueError): pass
class UnsupportedContract(ValueError): pass

def utc(x):
    if not isinstance(x,datetime) or x.tzinfo is None or x.utcoffset()!=timedelta(0):
        raise DataError('An explicit UTC datetime is required')
    return x

@dataclass(frozen=True)
class Contract:
    symbol:str
    kind:str
    multiplier:float=1
    expiry:datetime|None=None
    underlying:str|None=None
    strike:float|None=None
    right:str|None=None
    style:str|None=None
    settlement:str|None=None
    currency:str='USD'
    def __post_init__(self):
        if self.kind not in ('stock','future','option') or self.currency!='USD':
            raise UnsupportedContract('Only USD stock, future and option contracts')
        if not math.isfinite(self.multiplier) or self.multiplier<=0:raise DataError('Invalid multiplier')
        if self.kind=='stock' and self.multiplier!=1:raise DataError('Stock multiplier must be one')
        if self.kind!='stock':utc(self.expiry)
        if self.kind=='option' and (not self.underlying or self.right not in ('C','P') or self.style not in ('american','european') or self.settlement not in ('cash','physical') or self.strike is None or not math.isfinite(self.strike) or self.strike<0):
            raise DataError('Incomplete option contract terms')

@dataclass(frozen=True)
class Quote:
    asof:datetime
    bid:float
    ask:float
    mark:float
    adjusted:bool=False
    def validate(self,c,now,max_age):
        utc(self.asof)
        if not all(math.isfinite(x) for x in (self.bid,self.ask,self.mark)):raise DataError('Nonfinite quote')
        if self.adjusted:raise DataError('Adjusted series cannot be executable contract quotes')
        if self.asof>now or now-self.asof>max_age:raise DataError('Future or stale quote')
        if self.bid>self.ask or not self.bid<=self.mark<=self.ask:raise DataError('Crossed quote or mark outside spread')
        if c.kind!='future' and self.bid<0:raise DataError('Negative stock/option quote')

@dataclass(frozen=True)
class Margin:
    initial:float
    maintenance:float
    known_at:datetime
    effective_from:datetime
    effective_until:datetime
    def validate(self,now):
        for t in (self.known_at,self.effective_from,self.effective_until):utc(t)
        if not (self.known_at<=now and self.effective_from<=now<self.effective_until):raise DataError('Margin schedule unavailable or expired')
        if not (math.isfinite(self.initial) and math.isfinite(self.maintenance) and 0<=self.maintenance<=self.initial):raise DataError('Invalid margin schedule')

class Book:
    def __init__(self,contracts,cash=10000.,fee_per_unit=0.,american_policy=None,max_quote_age=timedelta(seconds=1)):
        self.contracts={c.symbol:c for c in contracts}
        if len(self.contracts)!=len(contracts):raise DataError('Duplicate contract')
        if not math.isfinite(cash) or cash<0 or not math.isfinite(fee_per_unit) or fee_per_unit<0:raise DataError('Invalid initial capital/fee')
        self.cash=float(cash);self.positions={};self.last_marks={};self.fees=0.;self.events=[];self.now=None
        self.fee_per_unit=fee_per_unit;self.american_policy=american_policy;self.max_quote_age=max_quote_age
        self.margin_call=False
    def _validate_clock(self,now):
        utc(now)
        if self.now is not None and now<self.now:raise DataError('Events must be chronological')
    def _validate_quotes(self,quotes,symbols,now,allow_expiry=False):
        self._validate_clock(now)
        for symbol in symbols:
            if symbol not in self.contracts or symbol not in quotes:raise DataError(f'Missing contract/quote: {symbol}')
            c=self.contracts[symbol];quotes[symbol].validate(c,now,self.max_quote_age)
            if c.expiry and (now>c.expiry if allow_expiry else now>=c.expiry):raise DataError('Contract expired: process dated settlement event first')
        if symbols and max(quotes[s].asof for s in symbols)-min(quotes[s].asof for s in symbols)>self.max_quote_age:
            raise DataError('Legs are not synchronized')
    def _mark(self,now,quotes):
        for s,q in self.positions.items():
            c=self.contracts[s]
            if c.kind=='future':self.cash+=q*c.multiplier*(quotes[s].mark-self.last_marks[s])
        for s in self.positions:self.last_marks[s]=quotes[s].mark
        self.now=now
    def equity(self):
        return self.cash+sum(q*self.contracts[s].multiplier*self.last_marks[s] for s,q in self.positions.items() if self.contracts[s].kind!='future')
    def collateral(self,now,margins,maintenance=False):
        required=0.;groups=defaultdict(list);stock={s:q for s,q in self.positions.items() if self.contracts[s].kind=='stock'}
        for s,q in self.positions.items():
            c=self.contracts[s]
            if c.kind=='future':
                if s not in margins:raise DataError('Missing dated contract margin')
                margins[s].validate(now);required+=abs(q)*(margins[s].maintenance if maintenance else margins[s].initial)
            elif c.kind=='option':groups[(c.underlying,c.expiry)].append((c,q))
            elif q<0:raise UnsupportedContract('Stock borrow and naked stock shorts are unsupported')
        for (underlying,_),legs in sorted(groups.items()):
            tail=sum(q*c.multiplier for c,q in legs if c.right=='C')
            cover=min(stock.get(underlying,0),max(0,-tail));stock[underlying]=stock.get(underlying,0)-cover
            if tail+cover < -1e-9:raise CollateralError('Unbounded short-call loss; no naked calls')
            points={0.}|{c.strike for c,q in legs}
            def payoff(spot):
                return cover*spot+sum(q*c.multiplier*max(0,spot-c.strike if c.right=='C' else c.strike-spot) for c,q in legs)
            required+=max(0,-min(payoff(s) for s in points))
        return required
    def mark(self,now,quotes,margins):
        """Record observed marks/variation cash; flag a breach without erasing debt."""
        self._validate_quotes(quotes,set(self.positions),now)
        trial=deepcopy(self);trial._mark(now,quotes)
        trial.margin_call=trial.cash+1e-8<trial.collateral(now,margins,maintenance=True)
        trial.events.append({'time':now.isoformat(),'type':'mark','equity':trial.equity(),'margin_call':trial.margin_call})
        self.__dict__.update(trial.__dict__)
    def rebalance(self,now,targets,quotes,margins,decision_at):
        """All targets replace the book; all changed legs fill bid/ask atomically.

        No implicit liquidity or market-impact assumption is hidden: this is an
        explicit synchronous-fill research convention, requiring quote-size data
        and a fill policy before historical performance can qualify.
        """
        utc(decision_at)
        if decision_at>=now:raise DataError('Decision must precede execution')
        if any(not isinstance(q,int) or isinstance(q,bool) for q in targets.values()):raise DataError('Integer units required')
        targets={s:q for s,q in targets.items() if q}
        symbols=set(self.positions)|set(targets);self._validate_quotes(quotes,symbols,now)
        for s,q in targets.items():
            c=self.contracts[s]
            if c.kind=='option' and c.style=='american' and q<0 and self.american_policy!='explicit_events':
                raise UnsupportedContract('American shorts need explicit exercise/assignment events')
        trial=deepcopy(self);trial._mark(now,quotes);old=dict(trial.positions)
        for s in sorted(symbols):
            delta=targets.get(s,0)-old.get(s,0)
            if not delta:continue
            c=self.contracts[s];quote=quotes[s];fill=quote.ask if delta>0 else quote.bid;fee=abs(delta)*self.fee_per_unit
            if c.kind=='future':trial.cash+=delta*c.multiplier*(quote.mark-fill)
            else:trial.cash-=delta*c.multiplier*fill
            trial.cash-=fee;trial.fees+=fee
            trial.events.append({'time':now.isoformat(),'decision_at':decision_at.isoformat(),'type':'fill','symbol':s,'units':delta,'price':fill,'fee':fee})
        trial.positions=targets;trial.last_marks={s:quotes[s].mark for s in targets}
        required=trial.collateral(now,margins)
        if targets and trial.cash+1e-8<required:raise CollateralError(f'Cash {trial.cash:.2f} below reserve {required:.2f}; atomic order rejected')
        trial.margin_call=trial.cash+1e-8<required;self.__dict__.update(trial.__dict__)
    def exercise(self,now,events,spots,quotes,margins,*,expiry=False,source):
        """Explicit full/partial exercise or assignment, using supplied settlement.

        events maps option symbol to positive number of contracts exercised or
        assigned. At expiry it must include every expiring position, including
        OTM contracts (expired worthless). source identifies the official
        settlement/assignment record; this engine cannot manufacture that data.
        """
        self._validate_clock(now)
        if not source:raise DataError('Settlement/assignment source required')
        trial=deepcopy(self)
        if expiry:
            due={s for s,q in self.positions.items() if self.contracts[s].kind=='option' and self.contracts[s].expiry==now}
            if set(events)!=due:raise DataError('Settle every due leg together')
        for s,n in events.items():
            if not isinstance(n,int) or n<=0 or s not in trial.positions or abs(trial.positions[s])<n:raise DataError('Invalid exercised quantity')
            c=self.contracts[s];held=trial.positions[s]
            if c.kind!='option' or (expiry and (c.expiry!=now or n!=abs(held))):raise DataError('Invalid expiry event')
            if not expiry and (c.style!='american' or now>=c.expiry):raise DataError('Early event requires American option before expiry')
            if held<0 and c.style=='american' and self.american_policy!='explicit_events':raise UnsupportedContract('Missing assignment policy')
            spot=spots.get(c.underlying)
            if spot is None or not math.isfinite(spot) or spot<0:raise DataError('Missing official settlement/underlying spot')
            intrinsic=max(0,spot-c.strike if c.right=='C' else c.strike-spot)
            signed=n if held>0 else -n
            # Explicit early events can exercise OTM options; expiry auto-exercise uses positive intrinsic only.
            if c.settlement=='cash':trial.cash+=signed*c.multiplier*intrinsic
            elif not expiry or intrinsic>0:
                if c.underlying not in trial.contracts or trial.contracts[c.underlying].kind!='stock':raise UnsupportedContract('Physical delivery requires a supported stock underlying')
                units=signed*c.multiplier*(1 if c.right=='C' else -1)
                if not float(units).is_integer():raise UnsupportedContract('Fractional delivery unsupported')
                trial.positions[c.underlying]=trial.positions.get(c.underlying,0)+int(units)
                trial.cash-=units*c.strike
            trial.positions[s]-=signed
            trial.events.append({'time':now.isoformat(),'type':'expiry' if expiry else 'exercise_assignment','symbol':s,'contracts':signed,'spot':spot,'source':source})
        trial.positions={s:q for s,q in trial.positions.items() if q}
        # Advance futures using their previous marks before replacing current marks.
        remaining=set(trial.positions);trial._validate_quotes(quotes,remaining,now)
        for s in remaining:
            if trial.contracts[s].kind=='future':trial.cash+=trial.positions[s]*trial.contracts[s].multiplier*(quotes[s].mark-trial.last_marks[s])
        trial.last_marks={s:quotes[s].mark for s in remaining};trial.now=now
        # Exercise is a forced contractual event. Preserve a resulting cash deficit;
        # unsupported stock borrowing fails explicitly before any mutation.
        trial.margin_call=trial.cash+1e-8<trial.collateral(now,margins,maintenance=True)
        self.__dict__.update(trial.__dict__)

def front_contract(rows,decision_at,eligible):
    """Select from the most recent completed session whose volume was published.

    rows: dicts with contract, session_end, available_at and volume. A same-day
    final volume record is ineligible while that day's session is still running.
    """
    utc(decision_at);valid=[]
    for r in rows:
        utc(r['session_end']);utc(r['available_at'])
        if r['contract'] not in eligible:continue
        if r['available_at']<r['session_end']:raise DataError('Final session volume published before close')
        if r['session_end']>=decision_at or r['available_at']>decision_at:continue
        if not math.isfinite(r['volume']) or r['volume']<0:raise DataError('Invalid session volume')
        valid.append(r)
    if not valid:raise DataError('No completed-session volume available')
    latest=max(r['session_end'] for r in valid);latest_rows=[r for r in valid if r['session_end']==latest]
    if {r['contract'] for r in latest_rows}!=set(eligible):raise DataError('Incomplete comparable contract universe')
    if len({r['contract'] for r in latest_rows})!=len(latest_rows):raise DataError('Duplicate contract volume')
    return sorted(latest_rows,key=lambda r:(-r['volume'],r['contract']))[0]['contract']
