import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "research/paper151"))
from derivatives import (
    Book,
    CollateralError,
    Contract,
    DataError,
    Margin,
    Quote,
    UnsupportedContract,
    front_contract,
)

T = datetime(2024, 1, 2, 15, tzinfo=timezone.utc)
EXP = T + timedelta(days=30)


def quote(price, at=T, spread=0):
    return Quote(at, price - spread / 2, price + spread / 2, price)


def future(s="ESH4"):
    return Contract(s, "future", 50, EXP)


def margin(at=T):
    return Margin(1000, 800, T - timedelta(days=1), T - timedelta(days=1), EXP)


def option(s, right="C", strike=100, style="european", settlement="cash"):
    return Contract(s, "option", 100, EXP, "S", strike, right, style, settlement)


def buy(b, targets, quotes, margins={}, now=T):
    b.rebalance(now, targets, quotes, margins, now - timedelta(seconds=1))


def test_futures_variation_cash_not_notional_and_costs():
    b = Book([future()], 10000, 2.5)
    buy(b, {"ESH4": 2}, {"ESH4": quote(5000, spread=1)}, {"ESH4": margin()})
    assert b.cash == 9945 and b.equity() == 9945
    b.mark(T + timedelta(days=1), {"ESH4": quote(5010, T + timedelta(days=1))}, {"ESH4": margin()})
    assert b.cash == 10945
    buy(b, {}, {"ESH4": quote(5010, T + timedelta(days=1), 1)}, {"ESH4": margin()}, T + timedelta(days=1))
    assert b.cash == 10890 and b.fees == 10 and b.positions == {}


def test_roll_charges_both_legs_and_does_not_book_calendar_basis_as_profit():
    b = Book([future("old"), future("new")], 10000, 2)
    buy(b, {"old": 1}, {"old": quote(5000)}, {"old": margin()})
    buy(b, {"new": 1}, {"old": quote(5000, spread=1), "new": quote(5020, spread=1)}, {"new": margin()})
    assert b.cash == 9944  # Three $2 fees and two $25 half-spreads, no $1,000 basis profit.


def test_missing_old_contract_quote_rejects_roll_without_mutation():
    b = Book([future("old"), future("new")])
    buy(b, {"old": 1}, {"old": quote(5000)}, {"old": margin()})
    before = deepcopy(b.__dict__)
    with pytest.raises(DataError):
        buy(b, {"new": 1}, {"new": quote(5020)}, {"new": margin()})
    assert b.__dict__ == before


@pytest.mark.parametrize(
    "bad",
    [
        Quote(T + timedelta(seconds=1), 1, 2, 1.5),
        Quote(T - timedelta(seconds=2), 1, 2, 1.5),
        Quote(T, 2, 1, 1.5),
        Quote(T, 1, 2, 1.5, True),
        Quote(T, 1, 2, 3),
    ],
)
def test_invalid_quotes_fail_atomically(bad):
    b = Book([future()])
    before = deepcopy(b.__dict__)
    with pytest.raises(DataError):
        buy(b, {"ESH4": 1}, {"ESH4": bad}, {"ESH4": margin()})
    assert b.__dict__ == before


def test_missing_or_future_margin_schedule_rejected():
    b = Book([future()])
    with pytest.raises(DataError):
        buy(b, {"ESH4": 1}, {"ESH4": quote(5000)})
    with pytest.raises(DataError):
        buy(b, {"ESH4": 1}, {"ESH4": quote(5000)}, {"ESH4": Margin(1000, 800, T + timedelta(days=1), T, EXP)})


def test_credit_vertical_reserve_and_expiry_loss():
    b = Book([option("c100"), option("c105", strike=105)], 1000, 1)
    buy(b, {"c100": -1, "c105": 1}, {"c100": quote(3, spread=0.2), "c105": quote(1, spread=0.2)})
    assert b.cash == 1178 and b.collateral(T, {}) == 500
    b.exercise(
        EXP, {"c100": 1, "c105": 1}, {"S": 110}, {}, {}, expiry=True, source="synthetic official settlement"
    )
    assert b.cash == 678 and b.equity() == 678 and not b.positions


def test_insufficient_vertical_capital_rolls_back_all_legs():
    b = Book([option("c100"), option("c105", strike=105)], 100)
    with pytest.raises(CollateralError):
        buy(b, {"c100": -1, "c105": 1}, {"c100": quote(3), "c105": quote(1)})
    assert b.cash == 100 and b.positions == {} and b.events == []


def test_naked_calls_rejected_but_fully_cash_secured_puts_supported():
    b = Book([option("call"), option("put", "P")], 10000)
    with pytest.raises(CollateralError):
        buy(b, {"call": -1}, {"call": quote(2)})
    buy(b, {"put": -1}, {"put": quote(2)})
    assert b.collateral(T, {}) == 10000
    b.exercise(EXP, {"put": 1}, {"S": 0}, {}, {}, expiry=True, source="fixture")
    assert b.cash == 200


def test_american_assignment_is_explicit_and_delivers_covered_stock():
    contracts = [Contract("S", "stock"), option("call", style="american", settlement="physical")]
    b = Book(contracts, 20000)
    with pytest.raises(UnsupportedContract):
        buy(b, {"S": 100, "call": -1}, {"S": quote(99), "call": quote(2)})
    b = Book(contracts, 20000, american_policy="explicit_events")
    buy(b, {"S": 100, "call": -1}, {"S": quote(99), "call": quote(2)})
    b.exercise(T + timedelta(days=1), {"call": 1}, {"S": 102}, {}, {}, source="dated assignment fixture")
    assert not b.positions and b.cash == 20300


def test_american_short_put_assignment_delivers_stock_and_debits_strike():
    b = Book(
        [Contract("S", "stock"), option("put", "P", style="american", settlement="physical")],
        10000,
        american_policy="explicit_events",
    )
    buy(b, {"put": -1}, {"put": quote(2)})
    now = T + timedelta(days=1)
    b.exercise(now, {"put": 1}, {"S": 95}, {"S": quote(95, now)}, {}, source="fixture")
    assert b.positions == {"S": 100} and b.cash == 200 and b.equity() == 9700


def test_european_early_exercise_and_partial_expiry_rejected():
    b = Book([option("call")])
    buy(b, {"call": 2}, {"call": quote(2)})
    with pytest.raises(DataError):
        b.exercise(T, {"call": 1}, {"S": 101}, {}, {}, source="fixture")
    with pytest.raises(DataError):
        b.exercise(EXP, {"call": 1}, {"S": 101}, {}, {}, expiry=True, source="fixture")
    assert b.positions == {"call": 2}


def test_expiry_requires_every_due_leg_and_physical_future_delivery_rejected():
    b = Book([option("call"), option("put", "P")])
    buy(b, {"call": 1, "put": 1}, {"call": quote(1), "put": quote(1)})
    with pytest.raises(DataError):
        b.exercise(EXP, {"call": 1}, {"S": 101}, {}, {}, expiry=True, source="fixture")
    c = Contract("c", "option", 50, EXP, "ES", 5000, "C", "european", "physical")
    b = Book([c, future("ES")], 10000)
    buy(b, {"c": 1}, {"c": quote(1)})
    with pytest.raises(UnsupportedContract):
        b.exercise(EXP, {"c": 1}, {"ES": 5010}, {}, {}, expiry=True, source="fixture")


def test_negative_futures_prices_and_insolvency_remain_real_debt():
    b = Book([future()], 1000)
    buy(b, {"ESH4": 1}, {"ESH4": quote(10)}, {"ESH4": margin()})
    now = T + timedelta(days=1)
    b.mark(now, {"ESH4": quote(-20, now)}, {"ESH4": margin()})
    assert b.cash == -500 and b.equity() == -500 and b.margin_call
    buy(b, {}, {"ESH4": quote(-20, now)}, {}, now)
    assert b.cash == -500 and b.positions == {} and b.margin_call


def test_next_day_volume_cannot_change_front_selection():
    rows = [
        dict(contract=s, session_end=T - timedelta(days=1), available_at=T - timedelta(hours=23), volume=v)
        for s, v in [("old", 200), ("new", 100)]
    ]
    future_rows = [
        dict(contract=s, session_end=T + timedelta(hours=8), available_at=T + timedelta(hours=9), volume=v)
        for s, v in [("old", 1), ("new", 100000)]
    ]
    assert front_contract(rows, T, {"old", "new"}) == "old"
    assert front_contract(rows + future_rows, T, {"old", "new"}) == "old"
    assert front_contract(rows + future_rows, T + timedelta(days=1), {"old", "new"}) == "new"
    with pytest.raises(DataError):
        front_contract(rows[:1], T, {"old", "new"})


def test_same_bar_decisions_rejected():
    b = Book([future()])
    with pytest.raises(DataError):
        b.rebalance(T, {"ESH4": 1}, {"ESH4": quote(1)}, {"ESH4": margin()}, T)
