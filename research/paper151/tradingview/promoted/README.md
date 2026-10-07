# TradingView research examples

Ten Pine scripts provide portfolio target indicators and single-chart strategies. They are experimental adaptations, not qualified trading recommendations. Copy a `.pine` file into your own Pine Editor to inspect or compile it. No private account, saved layout, or webhook is required.

## What each file does

| Rank / saved name | Plain-language rule | Technical form |
|---|---|---|
| 01 GoldDiversification | Hold 80% stocks and 20% gold; reset monthly. | SPY/GLD target-weight indicator. |
| 02 SectorLowVol | Hold the three sectors with the smallest recent swings. | Monthly equal-weight selection by trailing 63-session total-return volatility. |
| 03 SectorRotationMA | Pick the three strongest sectors; invest only in selected sectors above their own long-term average. | Monthly 252-session total-return ranking, SMA200 filter, unused thirds in cash. |
| 04 SectorMomentumSkip21 | Pick strong sectors while ignoring their most recent month. | Monthly 252-session momentum ranking skipping 21 sessions; three equal weights. |
| 05 SectorRotation | Hold the three sectors with the strongest past-year returns. | Monthly 252-session total-return ranking; three equal weights. |
| 06 SectorRotationDual | Rotate sectors when the broad stock trend is positive; otherwise hold intermediate Treasuries. | SPY/SMA200 regime filter; top-three sectors or IEF. |
| 07 MA50_200 | Hold the chart's asset when its medium-term average exceeds its long-term average. | Native strategy: SMA50 > SMA200, next-open execution. |
| 08 MA20_50_200 | Invest when short, medium and long trends are aligned. | Native strategy: SMA20 > SMA50 > SMA200, next-open execution. |
| 09 AlphaCombo | Invest when at least two of three trend signals agree. | Native strategy: majority vote of SMA200, positive 252-bar total-return momentum and Donchian55/20 state. |
| 10 IBS | Buy a close near the day's low and exit after a rebound or five sessions. | Native strategy: IBS < 0.2 entry; IBS > 0.8 or five-bar exit. |

The first six display portfolio targets. They do not produce native multiasset Strategy Tester returns. Their monthly weights persist until rebalance; actual holdings drift between rebalances. On 29 September 2026, #03 displayed one third each in XLE, XLK and XLV, matching the local 1 September target. After the dividend/share-basis correction, all six indicators match all 117 historical monthly targets and all 2,442 compared daily target vectors each.

The last four are single-chart strategies. Use standard daily candles with dividend adjustment OFF; split adjustment is permitted. Default commission is 0.10% per side as a proxy for the local US commission plus slippage; use 0.15% for BTC. The proxy, 99.8% allocation buffer, share rounding, cash interest, dividend treatment, endpoints and price feed differ from local accounting. The US signal-start default is the close of 30 December 2016. Set the date deliberately when studying crypto or other intervals.

## Performance reconciliation status

These ten Pine-compatible families were selected from the original 24-family screen and retained after the signed and factor expansions brought the total to 34 recipes. None passed its declared broad qualification rule. They remain exploratory research tools, not the final top ten from a completed whole-paper test.

The four native scripts have now been reconciled on daily TLT through 23 September 2026: 702 aggregated execution events, integer quantities and cent-rounded fill prices match an independent reconstruction using TradingView's price history. MA50/200, triple MA and AlphaCombo also match Yahoo-based events; IBS is sensitive to small OHLC differences across feeds. Reconciliation accounts for unaffordable overnight-gap orders, commission reserves and small immediate margin liquidations. Original native exports and reconciliation records are retained locally and are not distributed.

The TLT/#07 price-only return is −0.50%; the full local dividend/cash-interest study is +30.31%. Replaying native fills with the local daily-equity risk definition gives 25.60% drawdown, while TradingView's closed-trade-peak intrabar calculation is 17.72%. Both are now reproduced under their respective definitions. Exact software matching under one configuration does not establish future outperformance, real fills or parity on every market. All six portfolio histories now match the local targets; they still do not provide native multiasset account performance. The accompanying synthetic distribution-invariance tests are included in the public suite.


See the [research overview](../../../../docs/RESEARCH_RESULTS.md). Historical verification applies to the saved configuration and data at that time; recompile and validate changed scripts independently. These scripts do not connect to a broker.
