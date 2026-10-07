# Research results and evidence

TensorTrade Lab evaluates trading ideas with explicit execution costs, benchmark comparisons, chronological diagnostics, and independent accounting checks. The studies below are separate experiments with different universes and assumptions; their results should not be combined into a single performance claim.

## Published study summaries

| Study | Scope | Finding |
| --- | --- | --- |
| Focused ETF PPO | SPY/QQQ features, staged PPO budgets and multiple seeds | 10.82 million development steps did not establish the required advantage; diagnostic finalists lagged buy-and-hold on the reserved 2014–2015 period. |
| Portfolio rotation | Eight rotation rules and two supervised variants across fourteen ETFs | No candidate passed the registered development gate against SPY and risk constraints. Conditional portfolio PPO was not started. |
| Expanded daily screen | 24 rule families and 96 market/rule combinations | Some combinations met a full-sample return/drawdown screen, but no family passed its broader cross-market/block/stress screen. |
| Ensemble expansion | 636 candidate cases, including related configurations and aliases | No newly qualified strategy. Some daily blends reduced drawdowns while lagging passive returns. Cases are not independent trials. |
| Dated-data adapters | 48 filing/inflation cases | The inflation allocation beat SPY across its full historical sample but failed two subperiod return requirements. Missing earnings disclosures limited the single-stock adapter. |
| Weekly options | 58 constructions across two cost cases | All 116 account attempts halted on missing required inputs. Full-schedule returns remain unknown; completed subaccounts are not full-year results. |

The research checkpoint for the later studies is October 3, 2026. It classified 175 catalogue rows into 41 limited historical adaptations, 61 quote-episode entries, 44 synthetic-only constructions, 27 blocked entries, and two excluded noncandidates. Partial evidence is not full replication.

## Example of benchmark-specific interpretation

In the ETF portfolio confirmation period, January 3, 2023 through September 21, 2026, the frozen diagnostic rotation candidate returned **88.20%**, versus **110.67%** for SPY and **66.67%** for the passive fourteen-ETF basket. It beat the basket and lagged SPY. Its lower drawdown than SPY did not satisfy the combined return/risk requirement.

See [portfolio results](PORTFOLIO_RESULTS.md) and [focused ETF results](FOCUSED_RESULTS.md) for the original summaries and accounting assumptions.

## Evidence boundaries

- Synthetic tests validate software behavior, not market returns.
- Historical screens use modeled costs, selected universes, and sometimes previously inspected history.
- Quote episodes do not establish complete options-account performance or synchronized multi-leg execution.
- Prospective paper observations are separate from historical results and do not imply real-money fills.
- A fresh clone can run the synthetic demo and public tests. Exact historical reproduction requires matching source datasets and manifests, which are not bundled.

Research source is included under [research/paper151](../research/paper151/README.md). Publication preserves the underlying local archives rather than distributing licensed market data or operational account records.
