# Focused SPY / QQQ experiment

This fixed experiment tests whether strategies can beat full initial allocation to the same ETF after costs. TensorTrade supplies the environment and trading components; it does not supply an automatic investment edge.

## Run and inspect

```bash
PYTHONPATH=src .venv/bin/ttlab focused --spec configs/focused_etf.yaml --output runs/focused-etf
PYTHONPATH=src .venv/bin/ttlab focused-audit --study runs/focused-etf
PYTHONPATH=src .venv/bin/ttlab focused-forward --study runs/focused-etf \
  --spy data/yahoo/SPY_1d.parquet --qqq data/yahoo/QQQ_1d.parquet
```

The dashboard's **Focused ETF study** page shows fit progress, learning curves, feature comparisons, reserved-period results, cost/delay stresses and previously seen regime diagnostics. Refresh to read updated artifacts. Training runs independently of the browser. Original `research-*` studies remain frozen.

Data and run artifacts are local and excluded from Git. To reproduce elsewhere, download the registered source window with the existing `download` command and retain metadata sidecars. Snapshot SHA-256 hashes identify the actual data; a later provider download may differ.

## Five changes

1. **Benchmark-relative selection.** Checkpoints maximize excess return over buy-and-hold subject to maximum drawdown ≤30%. Candidate selection uses mean development excess return minus half its standard deviation. Every fold/seed must respect the drawdown constraint for risk feasibility. Positive aggregate score and positive excess in at least two-thirds of evaluations are required for eligibility. If none qualifies, diagnostic finalists are retained without deployment approval.
2. **Daily ETFs.** SPY and QQQ are evaluated separately. The fixed hypotheses are 60/120-session trend, volatility-scaled trend, trend plus relative strength, two gradient-boosted tree models, and two PPO models. This is not a combined portfolio optimizer. The focused study uses long-only positions without leverage; broader searches still support long/short.
3. **Causal features.** Own-ETF context adds 20/60/120-session economic returns and trend, volatility, downside volatility and volatility regime. Cross-ETF variants add peer returns/volatility and relative strength. A peer candle is usable only after it closes. Own/cross variants share identical valid dates. Scaling fits training rows only; five-session supervised labels never cross the training boundary.
4. **Staged PPO.** Each initial PPO fit receives 100,000 steps. Promotion to 500,000 and 2,000,000 requires ≥0.5 percentage-point improvement in the best inner-validation selection score between the first and second halves of the cumulative budget in ≥50% of that variant's fits. Outer scores and reserved results never control promotion. Resumption restores the model and optimizer, and completed fits are cached. Interrupted fits restart from the latest saved checkpoint; environment/rollout state restarts, so resumed trajectories need not be bitwise identical.
5. **Separate evaluation.** Three chronological development periods precede one reserved historical period. Previously explored later regimes are diagnostics. A separate forward ledger records decisions before execution and never invents past actions.

## Timeline and fairness

The configuration and registration were saved before obtaining the 1999–2015 source window. SPY starts January 1999; QQQ starts March 1999. Indicators shorten usable history further.

| Fold | Train before | Inner checkpoint selection | Outer scoring |
|---|---|---|---|
| Financial crisis | 2006 | 2006–2007 | 2008–2009 |
| Recovery / euro crisis | 2008 | 2008–2009 | 2010–2011 |
| Expansion | 2010 | 2010–2011 | 2012–2013 |

Training drops the final five rows at its boundary. Scoring begins from the preceding completed candle so first execution falls inside the scoring period. Each fold starts with fresh capital; fold returns are not one continuously compounded portfolio. Seeds 17, 42 and 83 share the same market paths and are not independent market observations.

Only development snapshots ending before 2014 reach research workers. `selection.json` freezes candidates. Finalists refit using history before 2012 and checkpoint selection in 2012–2013, then evaluate on **2014–2015**, with seed 17 fixed in advance. This historical period was new to the application, but broad historical knowledge and retrospective provider adjustments remain; it is not prospective evidence.

Previously explored **2016–2026** data is audited only after selection, in 2016–2019, 2020–2021, 2022, 2023 onward and full-period diagnostics. The audit does not tune or replace candidates.

Strategies may allocate up to 100%, matching buy-and-hold's initial allocation. Separate drawdown/turnover reward penalties are disabled here; training rewards remain net log-equity changes. The 30% criterion is a selection constraint, not a guaranteed stop. A candidate can fail it in later data, and that failure is reported.

All historical comparators use identical periods, initial cash, 5-basis-point commission, 5-basis-point adverse slippage and end liquidation. Additional baselines are cash, 50% allocation, a 12% annualized-volatility target, and fixed exposure calibrated on inner validation. Targets use quarter increments, so exposure matches and volatility targets are approximate. Cash earns zero. Dividends are credited to cash; buy-and-hold does not automatically reinvest them. Taxes, real liquidity and variable market impact are not modeled.

Reserved-period stresses double commission/slippage, add one session of execution delay, and combine both, applying scenarios equally to the candidate and benchmark. Paired block-bootstrap intervals reuse the same sampled blocks for all seeds within each fold. They are descriptive, not corrected for strategy selection or all sources of uncertainty.

## Compute and artifacts

The local machine has 32 CPU cores and an M3 Ultra GPU. Research schedules up to 32 independent one-thread CPU fits, bounded by jobs, memory and the global core budget. Finalists and regime diagnostics also run in parallel. A direct-array TensorTrade observer speeds repeated training episodes; tests verify identical observations, settlement and rewards against the evaluation path.

Earlier benchmarks favored CPU for these small 64×64 networks. Metal remains supported and accelerated larger tensor kernels, but GPU saturation is not a useful target for this workload. PPO completes rollout batches, so actual steps may slightly exceed a stage budget.

`protocol.json` fingerprints the spec, execution/training engine, sources and metadata. Changes require a new experiment. Do not retune after opening the reserved period and keep calling it untouched. `fits/` retains all seeds, checkpoints, metrics, fills and equity; `promotion.json` records stage decisions; `finalists/` holds frozen bundles and reserved evaluations.

## Forward paper decisions

`focused-forward` freezes model/manifest/engine identities and a registration time. It consumes both ETF sources, drops unfinished candles, and records an action only if the next session has not opened. After a missed execution window it waits for a new completed session. There is no backdating or broker connection.

On later manual runs, only timely recorded actions execute. Missing decisions hold the existing position and are counted. The benchmark starts at the same time; open positions are retained across polls. Immutable raw-data prefixes detect revisions, truncation and edits. Retain and append to both snapshots; revised provider history can intentionally fail this check.

Registration calls for at least 126 observed sessions before initial assessment. Time cannot be accelerated with historical replay. No automatic polling is installed, and forward performance remains pending until future candles and timely decisions exist.

ETF bundles require both assets and the specialized feature builder. Use `focused-audit` and `focused-forward`; generic single-source paper/backtest commands reject these bundles explicitly.
