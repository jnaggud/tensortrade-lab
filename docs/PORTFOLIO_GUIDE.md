# Multi-asset portfolio rotation

The completed study lives in `runs/portfolio-rotation`. Open **Portfolio rotation** in the dashboard or its standalone `report.html`. All results are retrospective research. The frozen finalist did not establish an edge over SPY.

## What was built and run

1. Frozen-policy attribution of the prior SPY/QQQ strategies. Sequential zero-cost, constant-allocation and full-exposure counterfactuals reconcile exactly to their net excess return. The constant fraction is an ex post diagnostic rounded to the old action grid; the timing term includes differences from this approximate reference. These are not causal estimates.
2. A simultaneous multi-asset, long-only portfolio ledger and TensorTrade `TradingEnv` adapter. Holdings and cash are self-financing after costs; no leverage or shorting. Rules, supervised ranking and PPO share execution accounting.
3. Eight registered rotation rules: 126/252-session lookback × weekly/monthly rebalance × equal/inverse-volatility weights. Hold the three highest-scoring ETFs. Momentum skips the most recent 21 sessions: index[t−21]/index[t−lookback]−1. Inverse-volatility weights use trailing 63-session annualized volatility, floored at 1%. Ties follow the fixed universe order. After initial allocation, decisions use the last exchange session of each week/month and execute at the following open.
4. Two supervised comparisons using the same fixed gradient-boosted tree recipe and weekly/monthly decisions. Features: own 21/63/126/252-session economic returns and cross-sectional ranks, 63-session volatility, and 126-session drawdown. The rank includes SPY as an observable reference; only the 14 registered ETFs can be held. Labels are next-open entry through 21 sessions later, include entitled dividends, and subtract the investable-universe mean. Each model sees only pre-fold prices and fully matured labels. No label crosses the training boundary. Trees refit at fold boundaries; there is no hyperparameter or best-seed search.
5. Fixed benchmarks, evidence gates, family bootstrap, stress scenarios, reports and prospective paper controls. No simple candidate passed the gate, so no new market-data PPO training occurred. A separate synthetic smoke test verifies that the TensorTrade adapter can train/save/load PPO.

## Universe and chronology

Fourteen ETFs were chosen for economic coverage before evaluation: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, EFA, EEM, IEF, TLT, GLD. These are surviving funds, not a point-in-time all-ETF universe. Sector classifications and fund composition can change. SPY is a benchmark and observable reference, not a rotation holding.

Common history starts at GLD's November 2004 inception. Initial 252 sessions are feature warmup. Development windows are 2010–2014, 2015–2019 and 2020–2022, each with fresh capital. Supervised training expands through the preceding session. `selection.json` freezes the development finalist before evaluating 2023 onward. All ten fixed candidates remain visible; later results do not replace the finalist.

Several downloaded ETFs lacked September 22, 2026. The entire universe and every benchmark were truncated at September 21. No prices were filled or sessions silently skipped. Older interior gaps fail validation. Original source snapshots and quality reports remain preserved with hashes.

All SPY regimes were seen in earlier research. The recent period is a diagnostic, not an untouched holdout. New assets do not make familiar macroeconomic history fresh.

## Accounting and benchmarks

Completed daily features → next session's open. Fractional shares, 5 bps commission and 5 bps adverse slippage per side. Allocation solves for equity after costs. Strategy dividends accumulate as cash until rebalancing; passive benchmarks reinvest each holding's own distribution at that day's close, paying the same costs. Every historical comparison liquidates at the same final close. Ex-date credit approximates actual payment timing. No taxes, liquidity constraints, impact or live orders are modeled.

- **SPY:** buy once, reinvest dividends, then liquidate. Primary absolute-return benchmark.
- **Same-universe passive:** equal initial investment in all 14 funds, reinvest dividends within each holding, allow weights to drift. No periodic rebalancing.
- **Cash:** Yahoo `^IRX` bank discount yield converted using a 91-day bill assumption: annual investment yield = discount × 365 / (360 − 91 × discount). Only a strictly prior-date quote is used; ACT/365 accrual includes weekends. Quotes stale more than seven days fail validation. This is a research proxy, not a promised broker rate. Every portfolio uses it for idle cash.

CAGR uses elapsed calendar time; volatility uses 252 daily observations; Sharpe uses daily return above the cash proxy. Fold CAGR excess is averaged equally across folds; the bootstrap uses pooled daily log excess. These are distinct statistics. Cash's small daily-return volatility reflects weekend accrual, not bill price fluctuations.

## Evidence and compute gates

A simple candidate must have positive mean CAGR excess over **both** SPY and the passive universe; positive SPY excess in at least two of three folds; every fold's drawdown ≤ SPY + 2 percentage points; and volatility ≤110% of SPY. The adjusted bootstrap p-value must also be ≤0.05.

The bootstrap uses 2,000 circular resamples of 63-session blocks within each fold. All candidates receive identical date blocks. The null centers each fold/candidate and takes the maximum mean excess across ten candidates. The 95% intervals are pointwise; the maximum-statistic p-values adjust across this family only. These exploratory diagnostics do not correct earlier studies, survivorship or all researcher discretion. A high p-value is not the probability that a strategy is unprofitable.

If the gate passes, conditional PPO uses 100,000 requested steps, fixed 64×64 networks, three seeds, and the frozen finalist's rebalance frequency. It can allocate across all 14 ETFs plus cash. Scaling fits training data only. All seeds are retained; comparisons measure incremental CAGR and drawdown against the simple finalist. There is no best-seed deployment. This path was software-tested but not entered in the market study.

Independent jobs used up to **32 CPU processes, one thread each**, within the CPU/memory budget. Network downloads use four threads. Prior benchmarks favored CPU for small models; no GPU training was warranted after the evidence gate failed.

## Commands and artifacts

```bash
.venv/bin/ttlab rotation-attribution
.venv/bin/ttlab rotation --download
.venv/bin/ttlab dashboard
.venv/bin/ttlab rotation-forward
```

Completed research is immutable: configuration, engine or source changes require a new output directory and registration. Re-running unchanged completed research is idempotent. Incomplete interrupted runs repeat the fixed jobs; per-fit resumption is not implemented.

Artifacts include registration, source/engine fingerprints, all trials, models, decisions, fills, equity, bootstrap statistics, selection, cost/delay stresses and HTML/Markdown reports. `reconciliation.json` independently reconstructs cash, shares, dividends, interest, costs and daily equity from raw sources and saved fills. This does not verify Yahoo against another vendor or establish signal quality.

## Prospective paper observations

The diagnostic finalist can be observed despite failing the historical gate. It is never approved for live deployment. To acquire a new immutable snapshot and poll after a completed session, **before the next open**:

```bash
.venv/bin/ttlab rotation-forward --refresh
```

Alternatively pass a snapshot folder via `--data-directory`. Refresh downloads full history to a new directory; registered research prices are preserved. Changed historical prices, model or engine identities stop the poll rather than rewriting results.

Only timely recorded decisions trade. A missed scheduled decision holds the existing positions and is counted. Benchmarks start with the first recorded strategy execution. Positions stay open across polls, without final liquidation. The first completed session is counted immediately; equity metrics start after two sessions. At least 126 observed sessions are required before an initial review, not a claim that six months proves an edge. No scheduler or background polling is installed. Future evidence cannot be manufactured by historical replay.
