"""Signed-position ledger for TensorTrade's generic environment.

The built-in spot wallets cannot represent stock-loan liabilities. This execution
component explicitly records signed holdings, quote-currency commissions, borrow,
corporate actions and maintenance margin. No live broker connectivity.
"""

import numpy as np


class MarginLedger:
    def __init__(self, initial_cash):
        self.cash = float(initial_cash)
        self.units = 0.0
        self.borrow_cost = 0.0
        self.dividends = 0.0
        self.bankrupt = False

    def equity(self, price):
        return self.cash + self.units * price

    def transact(self, target, reference, commission, slippage_bps):
        equity = self.equity(reference)
        difference = target * max(0, equity) - self.units * reference
        if abs(difference) < 1e-8:
            return None
        sign = 1 if difference > 0 else -1
        price = reference * (1 + sign * slippage_bps / 10000)
        cost = abs(price - reference) + commission * price
        # Signed post-cost target allocation, including covering and direction changes.
        quantity = difference / (reference + target * sign * cost)
        fee = abs(quantity) * price * commission
        self.cash -= quantity * price + fee
        self.units += quantity
        if abs(self.units) < 1e-10:
            self.units = 0.0
        self.bankrupt = self.equity(reference) <= 0
        return {
            "side": "buy" if quantity > 0 else "sell",
            "units": abs(quantity),
            "price": price,
            "reference_price": reference,
            "fee": fee,
            "slippage": abs(quantity) * abs(price - reference),
            "notional": abs(quantity) * reference,
            "cash_after": self.cash,
            "units_after": self.units,
        }

    def corporate_action(self, split, dividend, adjustment):
        if adjustment == "raw":
            self.units *= split
        if adjustment != "total_return":
            amount = self.units * dividend
            self.cash += amount
            self.dividends += amount

    def accrue_borrow(self, reference_price, elapsed_days, apr):
        cost = max(0, -self.units) * reference_price * apr * elapsed_days / 365
        self.cash -= cost
        self.borrow_cost += cost
        return cost

    def margin_breached(self, price, maintenance):
        gross = abs(self.units) * price
        return gross > 0 and self.equity(price) <= gross * maintenance


def action_target(action, cap, current, long_short):
    if action == 0:
        return current
    if action <= 5:
        return (action - 1) * cap / 4
    if not long_short or action > 9:
        raise ValueError("Short action is unavailable")
    return -(action - 5) * cap / 4


def target_action(target, long_short):
    target = float(np.clip(target, -1 if long_short else 0, 1))
    step = round(abs(target) * 4)
    return 1 + step if target >= 0 else (5 + step if step else 1)
