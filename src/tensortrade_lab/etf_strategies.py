"""Small, predeclared ETF hypotheses and comparable-risk benchmarks."""

from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingRegressor

from .etf_features import economic_index
from .features import feature_columns
from .margin import target_action
from .strategies import SupervisedPolicy


@dataclass
class ETFPolicy:
    strategy: str
    lookback: int = 60
    fraction: float = 1.0
    target_volatility: float = 0.12
    bypass_risk: bool = False

    def action(self, bars, index):
        row = bars.iloc[index]
        if self.strategy == "fixed":
            return target_action(self.fraction, False)
        vol = float(row.context_own_volatility)
        allocation = min(1, self.target_volatility / max(vol, 1e-8))
        if self.strategy == "volatility":
            return target_action(allocation, False)
        trend = float(row[f"context_own_trend_{self.lookback}"])
        if self.strategy == "trend":
            return 5 if trend > 0 else 1
        if self.strategy == "volatility_trend":
            return target_action(allocation if trend > 0 else 0, False)
        if self.strategy == "relative_strength":
            return 5 if trend > 0 and float(row[f"context_relative_{self.lookback}"]) > 0 else 1
        raise ValueError(f"Unknown ETF strategy {self.strategy}")


def fit_etf_policy(candidate, train, seed):
    if candidate["strategy"] != "supervised":
        return ETFPolicy(candidate["strategy"], candidate.get("lookback", 60))
    columns = feature_columns(train)
    # Five-session, next-open economic return, with no labels crossing train end.
    index = economic_index(train)
    # An entry at the next open is not entitled to that session's ex-dividend payment.
    opening_economic = index * train.open / train.close
    labels = index.shift(-5) / opening_economic.shift(-1) - 1
    valid = labels.notna()
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=7,
        min_samples_leaf=50,
        learning_rate=0.03,
        l2_regularization=10,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(train.loc[valid, columns].to_numpy(), labels[valid].to_numpy())
    return SupervisedPolicy(model, columns, 0.002, False)
