# Portfolio study results

Completed the five-part portfolio research extension. No candidate passed the registered evidence and risk gate; market-data PPO training was therefore not started.

- 14 investable ETFs, eight momentum rules and two supervised ranking variants.
- 39 development evaluations, 13 recent-period evaluations, nine cost/delay stress portfolios.
- Up to 32 single-thread CPU workers. No GPU training was needed after the gate failed.
- 59 tests passed; lint and formatting checks passed.
- Independent reconstruction passed for all 61 portfolios; maximum absolute accounting error below $0.000001.

The development-frozen diagnostic finalist was 252-session momentum, skipping the latest 21 sessions, holding three ETFs at equal weights and rebalancing weekly.

| January 3, 2023–September 21, 2026 | Total return | CAGR | Maximum drawdown |
|---|---:|---:|---:|
| Frozen rotation finalist | 88.20% | 18.55% | 12.98% |
| SPY, dividends reinvested | 110.67% | 22.20% | 18.76% |
| Same-universe passive portfolio | 66.67% | 14.74% | 12.15% |

Returns include transaction costs. The recent interval is previously seen market history, not a fresh holdout. Source gaps near the download end were handled by truncating all assets at the last complete common session. No prices were filled.

Attribution of the earlier 2014–2015 failures found that even zero-cost strategies lost 3.22% for SPY and 3.75% for QQQ. Trading costs worsened results but did not explain most of the gap. The full sequential decomposition is saved under attribution/.

The diagnostic paper strategy is registered with zero prospective observations. Manual polling with new completed data is required; no past decisions were fabricated and no live orders are enabled. Run `.venv/bin/ttlab rotation-forward --refresh` after a completed session and before the next open.

The source/engine protocol remains unchanged from the start of the completed experiment. See [the methodology and commands](PORTFOLIO_GUIDE.md), and `runs/portfolio-rotation/report.html` for every candidate, period, stress scenario and data source.
