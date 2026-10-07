# Initial research results

Completed September 24, 2026. These are small-budget integration experiments, not evidence of a deployable strategy. 96 candidates completed across three independent studies, each with two seeds and two chronological scoring folds (384 candidate/fold/seed evaluations). PPO used only 1,024 steps per fit.

| Study | Trials | Selected candidate | Holdout return | Buy-and-hold | Candidate drawdown |
|---|---:|---|---:|---:|---:|
| daily | 32 | QQQ:1440 / mean_reversion / long_only | 16.90% | 52.43% | 10.79% |
| intraday | 48 | SPY:60 / supervised / long_only | -0.03% | -0.66% | 0.08% |
| archive | 16 | BTC-archive:15 / breakout / long_only | -3.07% | -23.14% | 3.58% |

The daily QQQ mean-reversion model lagged buy-and-hold while reducing drawdown. The recent intraday winner mainly stayed in cash: it had zero return in its scoring folds and only four fills on final holdout. The BTC archive winner lost money. None establishes an investment edge. Short Yahoo history and missing BTC archive candles limit conclusions. Returns across studies cover different dates and must not be ranked directly.

Final selections were frozen before evaluation. Each `runs/research-*/holdout.json` contains the exact dates, costs and stress tests; `leaderboard.csv` and per-trial folders retain validation results. No further tuning is allowed in these frozen study directories.

Thirty-two CPU workers were exercised. A real PPO trial completed on Metal, and a separate smoke run completed Metal PPO with two subprocess environments. All 32 automated tests pass. The kernel benchmark is in `runs/compute-benchmark.json`; kernel speed is not whole-system speed.

The daily finalist was also replayed on later QQQ data through September 23, 2026. Repeating the same replay produced zero new fills, verifying idempotency. This was historical offline replay, not forward observation of live prices.
