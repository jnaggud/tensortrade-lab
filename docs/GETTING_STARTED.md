# Training and evaluation guide

Run these examples from the repository root after completing the [installation](../README.md#quick-start). The CLI creates local artifacts under the ignored `data/` and `runs/` directories.

## Your historical data

Provide a CSV in oldest-to-newest order. Timestamps identify **candle opens**, must use ISO-8601 dates, and are normalized to UTC. Include **only completed candles**. For example:

```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,42000,42500,41800,42300,123.45
2024-01-01T01:00:00Z,42300,42700,42100,42600,140.20
```

Real input needs substantially more history than this format example. The defaults need at least roughly 400 candles to survive feature warmup, purging, and splits; meaningful research needs much more history across different market conditions. Prices must be positive, volume nonnegative, OHLC bounds valid, and timestamps unique. Missing hourly bars, NaNs, and descending rows fail validation rather than being silently repaired. Volume should use a consistent unit across the dataset.

```bash
uv run --no-editable --extra cpu ttlab validate --data data/btc_usd_1h.csv --config configs/default.yaml
uv run --no-editable --extra cpu ttlab train --data data/btc_usd_1h.csv --config configs/default.yaml
uv run --no-editable --extra cpu ttlab walk-forward --data data/btc_usd_1h.csv --config configs/default.yaml --folds 3
```

`--timesteps 10000` overrides training duration. PPO completes rollout batches, so actual steps can exceed the requested number; the manifest records the actual count. Validation patience can stop training early.

To generate more test data:

```bash
uv run --no-editable --extra cpu ttlab sample --rows 6000 --output data/synthetic_hourly.csv
uv run --no-editable --extra cpu ttlab train --data data/synthetic_hourly.csv --timesteps 10000
```

The generator adds a metadata sidecar identifying synthetic data. Keep it with the CSV. For other synthetic files, pass `--synthetic`.

## Evaluate and paper replay

Replace `runs/run-...` with the checkpoint directory printed after training.

```bash
uv run --no-editable --extra cpu ttlab backtest --run runs/run-... --data data/btc_usd_1h.csv
uv run --no-editable --extra cpu ttlab paper --run runs/run-... --data data/btc_usd_1h.csv --state runs/paper/account.json
```

Backtesting evaluates the candles after the model's validation boundary and permits historical context for warmup. The automatic training report instead evaluates only the purged test partition with its own warmup; these horizons intentionally differ. For a new, untouched holdout, provide a CSV whose evaluation candles were never used to make development decisions.

Paper replay is **offline, append-only simulation**. It reconstructs the account from the same checkpoint and CSV, retains the final open position, and writes an account snapshot, equity CSV, fills CSV, and next action. Append newly completed candles and run the same command again. Unchanged input does not duplicate fills. Changed historical candles or a changed model are rejected. Run one writer per account state. There is no background market-data connection or automatic polling.

Only load trusted local checkpoints: Stable-Baselines3 model files contain serialized Python data.

## Model and execution

- **Policy:** PPO with a two-layer MLP, configurable dimensions, learning rate, rollout length, entropy, discount, seed, and training budget.
- **Features:** 14 causal indicators covering returns, trend, volatility, RSI, candle range/body, relative volume, and UTC time of day. A 24-candle observation window plus six account-state features is the default. Indicators never use backfilling or future data. The normalizer is fitted only on training rows and saved with the model.
- **Actions:** hold the existing position, go to cash, or target 25%, 50%, 75%, or 100% of the configured position limit. The default limit is 75% of account equity; no shorting or leverage.
- **Execution:** decisions after candle *t* closes; market fills at candle *t+1*'s open, with adverse slippage and TensorTrade's source-wallet commission convention. Target sizing solves for allocation after costs. A minimum order value and a rebalance band reduce small trades.
- **Risk:** allocation cap, stop-loss, take-profit, re-entry cooldown, and a permanent drawdown halt for that episode. Risk exits override minimum order and rebalance limits. After a halt the evaluation continues in cash through the same horizon. Limits apply when processing bars; gaps and intrabar price moves can exceed them.
- **Reward:** scaled log change in account equity after costs, minus incremental drawdown and turnover penalties. Cash does not earn a fictional short-position return.
- **Validation:** chronological 60/20/20 partitions with 24-bar gaps. Best checkpoint maximizes validation net return minus drawdown. Test results never choose a checkpoint. Walk-forward retrains independently on expanding historical windows and evaluates successive future partitions.
- **Benchmarks:** cash, 100%-invested buy-and-hold, and a 24-hour momentum strategy subject to the same limits as the agent. All share evaluation timing and cost assumptions. Final backtest positions are liquidated at the scheduled final close; paper positions remain open.

## Run artifacts

Each training run contains:

```text
manifest.json       # config, scaler, data SHA-256, library versions, time boundaries, model checksum
best_model.zip      # checkpoint selected by validation
last_model.zip      # final training checkpoint
validation.json     # checkpoint-selection history
report.html         # self-contained interactive report
 test/
  metrics.json
  agent_equity.csv / agent_fills.csv
  buy_hold_equity.csv / buy_hold_fills.csv
  momentum_equity.csv / momentum_fills.csv
  cash_equity.csv / cash_fills.csv
```

Walk-forward creates a full run per fold and a `folds.csv` / `summary.json`. Fold portfolios begin with fresh capital; reported fold returns are not a continuously compounded live portfolio.


## Compute and installation

Use `uv run --no-editable --extra cpu ttlab hardware` to inspect available devices and `uv run --no-editable --extra cpu ttlab benchmark` to measure the relevant kernels. More GPU activity is not necessarily faster for small policies; worker and thread limits are explicit in the configuration.

For macOS environments with hidden editable-install `.pth` files, use the wheel installation from the quick start. After modifying application source, run `uv sync --frozen --extra dev --extra research --extra cpu --no-editable --reinstall-package tensortrade-lab`.

## GPU setup

The quick start uses the `cpu` extra for a portable Linux/Windows installation. macOS uses the standard PyTorch build, which can expose Metal when available. CPU and GPU kernels can produce different floating-point results.

On a Linux machine with a compatible NVIDIA driver, use a separate environment and select the `cu130` extra instead of `cpu`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --frozen --extra dev --extra research --extra cu130 --no-editable
UV_PROJECT_ENVIRONMENT=.venv-cuda uv run --frozen --no-editable --extra cu130 ttlab hardware
UV_PROJECT_ENVIRONMENT=.venv-cuda uv run --frozen --no-editable --extra cu130 ttlab benchmark
```

This uses the locked CUDA 13.0 PyTorch build from the official PyTorch index. Driver compatibility and native library initialization must be verified on the target host before a large run; this release's Linux CI validates the CPU profile. Set the training device in your configuration only after `hardware` reports it available. The public release does not claim CUDA validation.

See [uv's PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for platform-specific package selection.
