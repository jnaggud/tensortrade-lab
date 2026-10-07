# Multi-market research

Use the dashboard at http://127.0.0.1:8501/. **Research searches** creates jobs, shows the leaderboard, lists dataset coverage, and reports compute benchmarks. **PPO training runs** retains the original workflow.

## Data already found and imported

`data/discovered_archives.json` lists 22 market archives found under `~/Documents/Pattern_FindR`. Original files are unchanged. Only OHLCV and corporate-action columns are imported; existing indicators, model files and strategy returns are excluded.

- `data/local/BTCUSD_1d.parquet`: 3,669 daily bars, December 2015–December 2025, no missing days.
- `data/local/BTCUSD_15m.parquet`: 7,685 bars, November 2025–February 2026, 84 missing bars. Suitable for exploratory research with explicit gaps; not a complete intraday archive.
- `data/yahoo/{SPY,QQQ,AAPL}_1d.parquet`: daily history beginning January 2016.
- `data/yahoo/{SPY,QQQ,AAPL}_15m.parquet`: recent intraday snapshots.

The database `~/Documents/Pattern_FindR/price_data.db` also contains SPY and futures. Saved SPY strategy bundles include about seven years of daily prices, but their adjustment convention needs verification before mixing them with fresh prices. Futures require contract multipliers, roll handling, margin and calendars not implemented here.

Every imported Parquet file has a `.metadata.json` with source path/request, hash, date range, price-adjustment convention and missing-bar count. Inputs are snapshots: refresh into a new filename. Research verifies checksums. No automatic splice across providers or incompatible adjustments occurs.

```sh
# Start in the project directory after installation.
.venv/bin/ttlab catalog
.venv/bin/ttlab download --ticker SPY --interval 15m --period 60d --output data/yahoo/SPY_15m_new.parquet
.venv/bin/ttlab download --ticker QQQ --interval 1d --start 2016-01-01 --output data/yahoo/QQQ_1d_new.parquet
.venv/bin/ttlab import-data --source /path/to/bars.csv --config configs/stocks.yaml --timezone America/New_York --output data/local/bars.parquet
```

Yahoo coverage varies by interval and service behavior. Requests can fail or return a shorter interval; inspect recorded actual dates. It cannot reconstruct years of 15-minute data. `auto_adjust=False`, explicit dividends, and split-adjusted OHLC avoid crediting splits twice. Daily prices map to exchange sessions. NYSE holidays, daylight saving and early closes are respected; the last hourly bar in a normal stock session is 30 minutes. Incomplete aggregation buckets are discarded, never fabricated. XNYS covers regular US stock/ETF sessions only.

## Searches and final evaluation

```sh
.venv/bin/ttlab search --spec configs/research_daily.yaml --output runs/research-daily-next
.venv/bin/ttlab search --spec configs/research_intraday.yaml --output runs/research-intraday-next
.venv/bin/ttlab search --spec configs/research_archive.yaml --output runs/research-archive-next
# Same command resumes. Increase --trials to a new total budget before finalization.
.venv/bin/ttlab search --spec configs/research_intraday.yaml --output runs/research-intraday-next --trials 96
.venv/bin/ttlab finalize --study runs/research-intraday-next
```

The supplied `runs/research-daily`, `runs/research-intraday`, and `runs/research-archive` studies are already frozen. Inspect them in the dashboard; the commands above create new studies. Their prior holdouts are now known, so future claims require a fresh untouched evaluation period.

Grid search traverses a reproducibly shuffled Cartesian product, bounded by the trial budget. TPE uses Optuna to adapt choices as completed results arrive. Its initial parallel batch necessarily explores before feedback is available. Invalid candidates are recorded as failures rather than terminating the entire study. Trials and errors persist in SQLite; only the coordinator writes the database. A process lock prevents competing coordinators. Interrupted trials are retried on resume. Seeded training is reproducible within the limits of GPU kernels; asynchronous TPE completion order is not exactly reproducible.

Each trial chooses a dataset/timeframe, direction, strategy and parameters. Strategies: momentum, mean reversion, breakout, a histogram gradient-boosted return forecaster, and PPO. `policy.*` parameters apply to rules/forecasters; `training.*` parameters apply to PPO. Cross-product grids can contain economically identical candidates with irrelevant parameter differences. Use smaller family-specific studies for efficient large searches.

A study first intersects source dates and reserves the last 20% of **calendar time**. Workers receive development-only snapshots. Expanding validation folds share date boundaries across assets/timeframes; each fold uses an inner training/checkpoint-validation partition and a separate scoring interval, with purge gaps. Scalers and forecaster labels use training rows only. PPO checkpoint selection uses inner validation only. Fold equity starts with fresh capital, not a stitched portfolio.

The ranking is mean(validation return − maximum drawdown) minus half its standard deviation across folds/seeds. The leaderboard also reports drawdown and frequency of beating buy-and-hold. This is a search objective, not a statistical significance test. Selecting from many candidates can still overfit validation.

`finalize` freezes `selection.json` **before** loading the final evaluation period, refits the selected candidate with the first predeclared seed, and evaluates it against buy-and-hold, momentum and cash. It also tests doubled costs, an extra execution-bar delay, and both together. Repeating finalization reads cached results. A frozen study rejects further tuning. Its saved policy works with `ttlab paper` using genuinely later completed data and matching cadence/configuration.

A short Yahoo intraday window does not contain enough daily observations for all folds. Use the long daily study for daily models and the recent intraday study for 15/60-minute comparisons. Incompatible or too-short dataset/timeframe pairs are excluded. Compare results on the common dates recorded by each study, not across different studies as though their returns covered identical markets.

The included 1,024-step PPO budgets are integration experiments. Increase timesteps and seeds for serious research; a brief search does not establish a profitable model.

## Compute

```sh
.venv/bin/ttlab hardware
.venv/bin/ttlab benchmark
```

This machine has an Apple M3 Ultra, 32 CPU cores, 80 GPU cores and 512 GiB of memory. `compute.workers: 0` makes all CPU cores eligible. Concurrency is bounded by the number of jobs, thread budget and available memory (an approximate 2 GiB per worker allowance). Each trial uses one environment and one BLAS/PyTorch thread by default, preventing nested thread explosions.

`device: hybrid` or `auto` reserves GPU slots when available and preferentially routes PPO there. Rules and scikit-learn models stay on CPU. Explicit `cpu`, `mps` and `cuda` are supported; unavailable explicit accelerators fail clearly. `gpu_jobs` controls concurrent models on each GPU. Metal/CUDA schedule tensor operations over GPU cores; individual GPU cores are not Python workers. There is no promise of full occupancy, distributed PPO, or linear scaling.

Standalone PPO supports `compute.vector_envs` using spawned environment processes. The default zero chooses automatically within CPU, memory, dataset and timestep budgets; a positive number requests an explicit count. Walk-forward folds also run in parallel. Outer search parallelism forces it to one. Run one heavy study at a time: independent CLI processes do not share a machine-wide resource reservation. Independent search trials parallelize training, simulation and validation across cores. One policy's time-ordered simulation remains sequential.

The saved kernel benchmark showed about 829 large-network forward/backward steps/s on Metal versus 56 with 32 CPU threads. For a small network, one CPU thread achieved about 4,734 steps/s versus 3,307 on Metal and 254 with 32 threads. These are kernel measurements, not full backtest throughput. See `runs/compute-benchmark.json`.

## Trading semantics and limits

Signals use completed candles and execute at a future open, including any configured additional delay. Time-based observation/cooldown/lookback settings translate into counts of trading bars; overnight/weekend periods are not fabricated. Optional higher-timeframe features become visible only after the higher-timeframe bar closes. Raw split events adjust indicator history forward causally and adjust held units at the effective open.

The original crypto long-only path still settles through TensorTrade spot wallets/orders/broker. Stocks and signed positions use a custom signed ledger inside the TensorTrade generic environment because spot wallets cannot represent short liabilities. It charges adverse slippage, quote-currency commissions, elapsed-calendar-time borrow interest, dividends paid/received, split adjustments and maintenance-margin liquidation. Exposures are capped at 100% NAV, with no intentional leverage. A gap can create negative equity; the deficit remains visible and trading halts. Dividends are credited/debited at the ex-date open; payment-date cash timing is not modeled. Short availability is an explicit simulation assumption or a per-bar boolean, not an assertion that a broker will lend the asset. Borrow recalls close at the next available open.

The simulations do not model intrabar stop fills, market impact, order-book liquidity, financing cash balances, taxes, dynamic stock-loan quotes or point-in-time universe membership. Total-return-adjusted prices are synthetic economic series, not historically executable prices. Use split-adjusted prices plus explicit cash dividends for equity execution research. No live orders or broker connections are present.

Official references: [TensorTrade source](https://github.com/tensortrade-org/tensortrade), [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [Optuna parallel optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html), [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars), [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
