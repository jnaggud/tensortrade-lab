# Architecture

```text
CSV → strict validation → causal feature frame → chronological partitions
                                      ↓
                          training-only fitted scaler
                                      ↓
TensorTrade TradingEnv ← PPO policy → target allocation + risk checks
      ↓                              ↓
window observer                  TensorTrade Broker
      ↓                              ↓
DataFeed / Stream                BarExchange / simulated service
                                     ↓
                           Portfolio / cash + asset wallets
                                     ↓
                          net-equity reward / execution ledger
```

## Event sequence

At reset, indicators and the observation window end at the last warmup candle.
The account starts in cash. At each step:

1. The policy selects an action from the completed candle and account-state observation.
2. The event clock moves to the next candle's open.
3. Previously triggered stops and current opening-gap risk checks can override the action.
4. The target allocation is converted to a buy cash amount or sell asset quantity, solving
   for the target after fees and slippage. The simulated exchange quote reflects the
   adverse execution price. TensorTrade creates, locks, fills, and releases the order.
5. At candle close, equity is marked from settled wallet balances. The peak, drawdown,
   future risk exits, net-equity reward, and next observation are updated.
6. At the predetermined end of a backtest, all strategies liquidate at that close with
   costs. Paper replay disables that final liquidation and keeps the account open instead.

Stop checks use closes and opens. They never retrospectively fill at a stop inside
an already observed candle. A price gap can make losses exceed a configured threshold.

## Leakage controls

All indicators are causal. Feature normalization is fitted on training rows only.
Partitions are chronological and separated by a purge gap; no return crosses a
partition boundary. Validation selects a checkpoint. The held-out test partition is
used after selection. Observation history within each training-report partition is
warmup rather than traded data. Walk-forward uses independent models per fold.

The separate backtest and paper commands recover history for warmup but cannot execute
before the saved validation boundary. Repeatedly inspecting a test and then changing
configuration still creates researcher selection bias; reserve a fresh final holdout.

## Reproducibility and limits

The lockfile pins upstream source and dependencies. Manifests retain configuration,
normalizer, split bounds, source checksum, seed, versions, and model checksum. Training uses resource-budgeted CPU processes and optional Metal/CUDA acceleration. Evaluation uses deterministic policy actions. Floating-point
results can still differ across hardware and library versions. Preserve input CSVs
alongside their manifest hashes.

TensorTrade's shared global Wallet ledger is not used as a cross-environment account
history. Each environment owns independent wallets and a broker; reports are generated
from its own balance snapshots and actual broker fills. Parallel environments live in separate spawned processes to isolate the global ledger. Outer searches and walk-forward runs parallelize independent models; each worker uses one environment.

The paper runner reconstructs history rather than unpickling a mutable account. This
is deterministic and idempotent but takes linear time in history length. The account
JSON is atomically replaced and a per-account file lock prevents concurrent writers.
This is an offline paper executor, not a market-data
subscriber or live-broker reconciliation service.


## Multi-market extension

`market_data.py` supplies XNYS session grids and complete-bar aggregation. `sources.py` imports CSV/Parquet/SQLite archives and Yahoo snapshots with provenance. `margin.py` settles stocks and signed positions inside the same TensorTrade generic environment, recording borrow, dividends, splits and maintenance margin. The spot-wallet execution path remains available for long-only crypto.

`experiments.py` stores a study fingerprint including input/configuration/engine hashes, creates development-only snapshots, coordinates grid/TPE candidates in resource slots, and persists results through a single Optuna writer. Candidate scoring uses expanding date folds and inner validation. Final selection is frozen before final-period evaluation; saved policies support subsequent paper replay. `compute.py` owns thread limits, device detection, slot planning and benchmarks. See RESEARCH_GUIDE.md for exact semantics and practical limits.
