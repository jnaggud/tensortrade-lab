# Focused ETF results

Frozen historical experiment; forward evidence remains pending.

162 completed staged fits, 144 distinct candidate/asset/seed/fold fits, and 10,819,584 PPO development steps.

| ETF | Frozen candidate | Strategy return | Buy-and-hold | Excess | Strategy drawdown | Development eligible |
|---|---|---:|---:|---:|---:|---|
| SPY | trend_120 | -6.64% | 14.96% | -21.60 pp | 15.63% | No |
| QQQ | trend_60 | -8.07% | 30.43% | -38.50 pp | 15.64% | No |

All returns above cover the reserved 2014–2015 period, after 5 bps commission and 5 bps slippage per side. Strategies and benchmarks share dates and liquidation assumptions.

## Budget decisions

- ppo_own at 100,000 steps: 55.6% of fits improved enough; promotion criterion met.
- ppo_cross at 100,000 steps: 27.8% of fits improved enough; stopped at this budget.
- ppo_own at 500,000 steps: 33.3% of fits improved enough; stopped at this budget.

## Interpretation

Selection was frozen before the reserved period was opened. A diagnostic finalist does not mean a strategy passed eligibility. No deployment is approved. The 30% drawdown criterion is a selection constraint, not a guaranteed stop. Bootstrap intervals describe development resamples and are not corrected for model selection.

Own/cross feature comparisons, every seed, and all development folds are preserved in leaderboard.csv and development_results.json. Different training budgets must be considered when comparing final PPO variants.

2016–2026 checks in previously_seen.json use previously explored history and are diagnostics, not a fresh holdout. Prospective validation requires future candles and timely recorded decisions; historical replay cannot complete it.

## Verification

All 44 automated tests passed across the verification runs. Lint and formatting checks passed. An independent cash-and-shares calculation, without TensorTrade or the execution ledger, reproduced every reserved-period equity point and fill count for both strategies and both buy-and-hold benchmarks; maximum equity discrepancy was below $0.000001. Details are in `runs/focused-etf/reconciliation.json`.

The research engine hash still matches its pre-training fingerprint. Up to 32 CPU workers were used; the small networks were kept on CPU based on measured performance. The broader system retains Metal support.

Forward paper identities were registered after historical evaluation. There are currently zero prospective sessions and zero recorded decisions because the available completed candles precede an already-passed execution window. New completed data and timely polling are required; historical decisions were not backfilled.
