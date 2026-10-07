"""TensorTrade components with explicit close-decision / next-open execution.

The real TensorTrade Broker, Order, Exchange, Portfolio and Wallet objects settle
all fills. Custom components make timing, state, risk checks and costs explicit.
"""

from collections import deque
from decimal import Decimal
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete
from tensortrade.core import Clock
from tensortrade.env.default.renderers import EmptyRenderer
from tensortrade.env.generic import ActionScheme, Informer, Observer, RewardScheme, Stopper, TradingEnv
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import ExchangePair, Instrument
from tensortrade.oms.orders import Broker, TradeSide, market_order
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet

from .config import Config
from .margin import MarginLedger, action_target

ACTION_LABELS = ["Hold", "Cash", "25% of limit", "50% of limit", "75% of limit", "100% of limit"]


class BarExchange(Exchange):
    """A single-pair simulated exchange whose quote is controlled by the event clock."""

    def __init__(self, config, pair):
        super().__init__(
            "paper",
            execute_order,
            ExchangeOptions(commission=config.execution.commission, max_trade_size=1e18),
        )
        self.pair = pair
        self.mark = 1.0

    def quote_price(self, trading_pair):
        if trading_pair != self.pair:
            raise ValueError("Unknown trading pair")
        return Decimal(str(self.mark)).quantize(Decimal("0.00000001"))

    def is_pair_tradable(self, trading_pair):
        return trading_pair == self.pair


class MarketState:
    def __init__(
        self,
        bars: pd.DataFrame,
        features: np.ndarray,
        config: Config,
        start_index: int | None = None,
        benchmark: bool = False,
        liquidate_on_end: bool = True,
    ):
        self.config, self.bars, self.features = config, bars.reset_index(drop=True), features
        self.closes = self.bars.close.to_numpy(float)
        self.opens = self.bars.open.to_numpy(float)
        self.times = self.bars.timestamp.tolist()
        self.time_labels = [t.isoformat() for t in self.times]
        self.splits = self.bars.get("split", pd.Series(1.0, index=self.bars.index)).to_numpy(float)
        self.dividends = self.bars.get("dividend", pd.Series(0.0, index=self.bars.index)).to_numpy(float)
        self.borrow_flags = self.bars.get(
            "borrow_available", pd.Series(config.risk.borrow_available, index=self.bars.index)
        ).to_numpy(bool)
        self.end_index = len(bars) - 1
        self.start_index = config.window - 1 if start_index is None else start_index
        if self.start_index < config.window - 1 or self.start_index >= len(bars) - 1:
            raise ValueError("Partition needs observation history and at least one execution candle")
        if len(features) != len(bars) or not np.isfinite(features).all():
            raise ValueError("Features must match candle count and contain finite values")
        self.benchmark = benchmark
        self.long_short = config.risk.direction == "long_short"
        self.use_margin = self.long_short or config.market.calendar == "XNYS"
        self.liquidate_on_end = liquidate_on_end
        cash_instrument = Instrument(config.market.currency, 8, "Cash")
        asset_instrument = Instrument(config.market.symbol, 8, "Asset")
        self.exchange = BarExchange(config, cash_instrument / asset_instrument)
        self.cash_wallet = Wallet(self.exchange, config.execution.initial_cash * cash_instrument)
        self.asset_wallet = Wallet(self.exchange, 0 * asset_instrument)
        self.portfolio = Portfolio(cash_instrument, [self.cash_wallet, self.asset_wallet])
        self.pair = ExchangePair(self.exchange, cash_instrument / asset_instrument)
        self.broker = Broker()
        self.bind_clock(Clock())
        self.reset()

    def bind_clock(self, clock):
        self.portfolio.clock = self.exchange.clock = self.broker.clock = clock

    @property
    def cash(self):
        return self.ledger.cash if self.use_margin else self.cash_wallet.total_balance.as_float()

    @property
    def units(self):
        return self.ledger.units if self.use_margin else self.asset_wallet.total_balance.as_float()

    @property
    def close(self):
        return float(
            self.closes[self.index] if getattr(self, "fast_data", False) else self.bars.close.iloc[self.index]
        )

    @property
    def equity(self):
        return self.cash + self.units * self.close

    @property
    def drawdown(self):
        return max(0.0, 1 - self.equity / self.peak)

    @property
    def exposure(self):
        return self.units * self.close / max(self.equity, 1e-12)

    def reset(self):
        self.portfolio.reset()
        self.broker.reset()
        self.ledger = MarginLedger(self.config.execution.initial_cash)
        self.actions = deque([0] * self.config.execution.delay_bars)
        self.index = self.start_index
        self.exchange.mark = self.close
        self.peak = self.config.execution.initial_cash
        self.entry_price = 0.0
        self.cooldown = 0
        self.halted = False
        self.pending_exit = ""
        self.last_reason = "initial"
        self.reward_value = 0.0
        self.step_turnover = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.fills = []
        self.history = [self.snapshot()]

    def snapshot(self):
        return {
            "timestamp": (
                self.time_labels[self.index]
                if getattr(self, "fast_data", False)
                else self.bars.timestamp.iloc[self.index].isoformat()
            ),
            "close": self.close,
            "equity": self.equity,
            "cash": self.cash,
            "units": self.units,
            "exposure": self.exposure,
            "drawdown": self.drawdown,
            "fees": self.total_fees,
            "slippage": self.total_slippage,
            "borrow_cost": self.ledger.borrow_cost,
            "dividends": self.ledger.dividends,
            "bankrupt": self.equity <= 0,
            "halted": self.halted,
            "reason": self.last_reason,
        }

    def trade_to(self, target: float, reference_price: float, reason: str, force: bool = False):
        """Rebalance using next-open equity; commission follows TT's source-wallet convention."""
        execution = self.config.execution
        value = self.cash + self.units * reference_price
        held = self.units * reference_price
        target = float(np.clip(target, -1 if self.long_short else 0, 1))
        difference = target * value - held
        if abs(difference) < 1e-7:
            return
        if not force and (
            abs(difference) < execution.min_trade_value
            or abs(difference) / max(value, 1e-12) < execution.rebalance_band
        ):
            return
        if self.use_margin:
            before = self.units
            fill = self.ledger.transact(target, reference_price, execution.commission, execution.slippage_bps)
            if fill:
                self.total_fees += fill["fee"]
                self.total_slippage += fill["slippage"]
                self.step_turnover += fill["notional"]
                if not self.units:
                    self.entry_price = 0.0
                elif before * self.units <= 0:
                    self.entry_price = fill["price"]
                elif abs(self.units) > abs(before):
                    self.entry_price = (
                        abs(before) * self.entry_price + abs(self.units - before) * fill["price"]
                    ) / abs(self.units)
                self.fills.append(
                    {
                        "timestamp": (
                            self.time_labels[self.index]
                            if getattr(self, "fast_data", False)
                            else self.bars.timestamp.iloc[self.index].isoformat()
                        ),
                        "reason": reason,
                        **fill,
                    }
                )
            return
        buy = difference > 0
        slip = execution.slippage_bps / 10000
        fill_price = reference_price * (1 + slip if buy else 1 - slip)
        fee = execution.commission
        # Solve the post-cost target allocation instead of buying target * pre-cost NAV.
        if buy:
            conversion = (1 - fee) * reference_price / fill_price
            spend = difference / (conversion + target * (1 - conversion))
            size = min(spend, self.cash)
            side = TradeSide.BUY
        else:
            conversion = (1 - fee) * fill_price / reference_price
            size = (-difference) / (reference_price * (1 - target * (1 - conversion)))
            size = min(size, self.units)
            side = TradeSide.SELL
        if size <= 1e-8:
            return
        self.exchange.mark = fill_price
        before_cash, before_units = self.cash, self.units
        order = market_order(side, self.pair, self.pair.price, size, self.portfolio)
        self.broker.submit(order)
        self.broker.update()
        # Rounding can leave a sub-precision remainder; never allow it to fill on another bar.
        if order.is_active:
            self.broker.cancel(order)
        executed = self.broker.trades.get(order.id, [])
        if not executed:
            raise RuntimeError("TensorTrade did not fill the market order")
        commission = sum(t.commission.as_float() * (1 if buy else fill_price) for t in executed)
        delta_units = self.units - before_units
        notional = abs(delta_units) * reference_price
        slippage = abs(delta_units) * abs(fill_price - reference_price)
        self.total_fees += commission
        self.total_slippage += slippage
        self.step_turnover += notional
        if buy and self.units > 0:
            self.entry_price = (before_units * self.entry_price + (before_cash - self.cash)) / self.units
        elif self.units < 1e-7:
            self.entry_price = 0.0
        self.fills.append(
            {
                "timestamp": (
                    self.time_labels[self.index]
                    if getattr(self, "fast_data", False)
                    else self.bars.timestamp.iloc[self.index].isoformat()
                ),
                "side": side.value,
                "units": abs(delta_units),
                "price": fill_price,
                "reference_price": reference_price,
                "fee": commission,
                "slippage": slippage,
                "notional": notional,
                "reason": reason,
                "cash_after": self.cash,
                "units_after": self.units,
            }
        )
        if self.cash < -1e-6 or self.units < -1e-8:
            raise RuntimeError("Negative wallet balance after settlement")

    def advance(self, action: int):
        if self.index >= self.end_index:
            raise RuntimeError("Episode is complete; reset before stepping again")
        if not Discrete(10 if self.long_short else 6).contains(action):
            raise ValueError(f"Invalid action: {action}")
        prior_equity, prior_drawdown = self.equity, self.drawdown
        self.actions.append(action)
        action = self.actions.popleft()
        prior_close = self.close
        prior_time = (
            self.times[self.index]
            if getattr(self, "fast_data", False)
            else self.bars.timestamp.iloc[self.index]
        )
        self.index += 1
        if getattr(self, "fast_data", False):
            from types import SimpleNamespace

            row = SimpleNamespace(timestamp=self.times[self.index])
            row.get = lambda name, default: {
                "split": self.splits[self.index],
                "dividend": self.dividends[self.index],
                "borrow_available": self.borrow_flags[self.index],
            }.get(name, default)
        else:
            row = self.bars.iloc[self.index]
        risk = self.config.risk
        if self.use_margin:
            self.ledger.accrue_borrow(
                prior_close, (row.timestamp - prior_time).total_seconds() / 86400, risk.borrow_apr
            )
            split = float(row.get("split", 1))
            self.ledger.corporate_action(
                split, float(row.get("dividend", 0)), self.config.market.price_adjustment
            )
            if self.config.market.price_adjustment == "raw" and self.entry_price:
                self.entry_price /= split
        opening = float(
            self.opens[self.index] if getattr(self, "fast_data", False) else self.bars.open.iloc[self.index]
        )
        open_equity = self.cash + self.units * opening
        self.step_turnover = 0.0
        self.last_reason = "hold"
        cap = 1.0 if self.benchmark else risk.max_position
        current_weight = self.units * opening / max(open_equity, 1e-12)
        target = action_target(action, cap, current_weight, self.long_short)
        reason = "rebalance"
        force = False
        entry_return = (opening / self.entry_price - 1) * np.sign(self.units) if self.entry_price else 0
        if not self.benchmark:
            # Risk checks may use the execution open, never this candle's high/low/close.
            if open_equity <= self.peak * (1 - risk.max_drawdown):
                self.halted = True
            if self.halted:
                target, reason, force = 0.0, "drawdown_halt", True
            elif self.pending_exit or (self.entry_price and entry_return <= -risk.stop_loss):
                target, reason, force = 0.0, self.pending_exit or "stop_loss", True
                self.cooldown = self.config.cooldown + 1
            elif self.entry_price and entry_return >= risk.take_profit:
                target, reason, force = 0.0, "take_profit", True
                self.cooldown = self.config.cooldown + 1
            elif abs(current_weight) > cap + 1e-6:
                target, reason, force = (
                    float(np.clip(target, -cap if self.long_short else 0, cap)),
                    "position_limit",
                    True,
                )
            elif self.cooldown and (abs(target) > abs(current_weight) or target * current_weight < 0):
                target = current_weight
                reason = "cooldown"
        if target < 0 and not bool(row.get("borrow_available", risk.borrow_available)):
            target, reason, force = 0.0, "borrow_unavailable", True
        if self.use_margin and (
            open_equity <= 0 or self.ledger.margin_breached(opening, risk.maintenance_margin)
        ):
            target, reason, force = 0.0, "margin_liquidation", True
            self.halted = True
        if self.halted:
            target = 0.0
        self.pending_exit = ""
        self.trade_to(target, opening, reason, force)
        self.last_reason = (
            reason if force or self.step_turnover else ("cooldown" if self.cooldown else "hold")
        )
        self.cooldown = max(0, self.cooldown - 1)
        self.peak = max(self.peak, self.equity)
        if not self.benchmark:
            if self.drawdown >= risk.max_drawdown:
                self.halted = True  # Liquidate at the next available open, with gap risk.
            close_return = (
                (self.close / self.entry_price - 1) * np.sign(self.units) if self.entry_price else 0
            )
            if self.entry_price and close_return <= -risk.stop_loss:
                self.pending_exit = "stop_loss"
            elif self.entry_price and close_return >= risk.take_profit:
                self.pending_exit = "take_profit"
        if self.liquidate_on_end and self.index == self.end_index:
            # Scheduled end-of-sample close liquidation applies identically to all policies.
            self.trade_to(0, self.close, "end_of_sample", force=True)
        self.exchange.mark = self.close
        self.reward_value = 100 * (
            np.log(max(self.equity, 1e-12) / max(prior_equity, 1e-12))
            - self.config.training.drawdown_penalty * max(0, self.drawdown - prior_drawdown)
            - self.config.training.turnover_penalty * self.step_turnover / max(prior_equity, 1e-12)
        )
        self.history.append(self.snapshot())


class TargetAllocation(ActionScheme):
    def __init__(self, state):
        super().__init__()
        self.state = state

    @property
    def action_space(self):
        return Discrete(10 if self.state.long_short else 6)

    def perform(self, env, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self.state.bind_clock(env.clock)
        self.state.advance(int(action))

    def reset(self):
        self.state.reset()


class WindowObserver(Observer):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.window = state.config.window
        self.feed = DataFeed(
            [
                Stream.source(state.features[:, j].tolist(), dtype="float").rename(f"f{j}")
                for j in range(state.features.shape[1])
            ]
        )
        self.feed.compile()
        self.history = deque(maxlen=self.window)
        self.consumed = -1

    @property
    def observation_space(self):
        return Box(-10, 10, shape=(self.window * self.state.features.shape[1] + 6,), dtype=np.float32)

    def observe(self, env):
        state = self.state
        while self.consumed < state.index:
            row = self.feed.next()
            self.history.append([row[f"f{j}"] for j in range(state.features.shape[1])])
            self.consumed += 1
        entry_return = (
            (state.close / state.entry_price - 1) * np.sign(state.units) if state.entry_price else 0
        )
        account = [
            state.exposure,
            state.cash / max(state.equity, 1e-12),
            state.drawdown,
            float(state.halted),
            state.cooldown / max(1, state.config.cooldown),
            entry_return,
        ]
        return np.clip(np.r_[np.array(self.history).reshape(-1), account], -10, 10).astype(np.float32)

    def reset(self, random_start=0):
        self.feed.reset()
        self.history.clear()
        self.consumed = -1


class NetEquityReward(RewardScheme):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def reward(self, env):
        return float(self.state.reward_value)


class EndOfData(Stopper):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def stop(self, env):
        # A risk halt continues in cash through the same evaluation horizon.
        return self.state.index == self.state.end_index


class AccountInfo(Informer):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def info(self, env):
        return self.state.snapshot()


class ResearchEnv(TradingEnv):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self, bars, features, config, start_index=None, benchmark=False, liquidate_on_end=True, training=False
    ):
        self.training = training
        self.state = MarketState(bars, features, config, start_index, benchmark, liquidate_on_end)
        self.state.fast_data = training
        super().__init__(
            action_scheme=TargetAllocation(self.state),
            observer=ArrayObserver(self.state) if training else WindowObserver(self.state),
            reward_scheme=NetEquityReward(self.state),
            stopper=EndOfData(self.state),
            informer=AccountInfo(self.state),
            renderer=EmptyRenderer(),
        )

    def reset(self, *, seed=None, options=None):
        # TensorTrade's base reset omits Gymnasium's RNG initialization.
        gym.Env.reset(self, seed=seed)
        if self.training and self.state.config.training.random_episode_bars:
            length = min(
                self.state.config.training.random_episode_bars,
                len(self.state.bars) - self.state.config.window,
            )
            self.state.start_index = int(
                self.np_random.integers(self.state.config.window - 1, len(self.state.bars) - length)
            )
            self.state.end_index = self.state.start_index + length
        return super().reset(seed=seed, options=options)


class ArrayObserver(WindowObserver):
    """TensorTrade observer with direct array windows for repeated training episodes."""

    def __init__(self, state):
        Observer.__init__(self)
        self.state, self.window = state, state.config.window

    def observe(self, env):
        state = self.state
        entry_return = (
            (state.close / state.entry_price - 1) * np.sign(state.units) if state.entry_price else 0
        )
        account = [
            state.exposure,
            state.cash / max(state.equity, 1e-12),
            state.drawdown,
            float(state.halted),
            state.cooldown / max(1, state.config.cooldown),
            entry_return,
        ]
        window = state.features[state.index - self.window + 1 : state.index + 1]
        return np.clip(np.r_[window.reshape(-1), account], -10, 10).astype(np.float32)

    def reset(self, random_start=0):
        pass
