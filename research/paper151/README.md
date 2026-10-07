# Strategy research extensions

These modules explore strategy families discussed in Zura Kakushadze and Juan Andrés Serur's [*151 Trading Strategies*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865), alongside ensemble and execution-accounting experiments. Recipes are explicitly adapted to available instruments and data; the catalogue is not a claim of 151 profitable strategies or complete historical replication.

## Module map

| Area | Entry points |
| --- | --- |
| Daily rules and portfolio screens | `run_study.py`, `expanded_study.py`, `signed_study.py` |
| Factor and statistical adapters | `factor_study.py`, `full_catalogue_prices.py`, `full_catalogue_value.py` |
| Futures and intraday execution | `derivatives.py`, `futures_extension.py`, `intraday_common.py` |
| Options quote episodes and account continuity | `options_pilot.py`, `full_catalogue_options.py`, `options_weekly.py` |
| Causal ensembles and PPO | `ensemble_study.py`, `ensemble_intraday.py`, `ensemble_btc_intraday.py` |
| Dated fundamentals and macro releases | `filing_study.py`, `macro_study.py`, `data_expansion_fetch.py` |
| Independent checks | `full_catalogue_validate.py`, `expansion_validate.py`, `reconcile_tradingview.py` |
| Prospective recording | `prospective.py` |
| Pine examples | [TradingView guide](tradingview/promoted/README.md) |

Install the `research` extra described in the main README. The default test suite exercises these modules with synthetic data. Research scripts are run from the repository root and expect the particular inputs named in their source; they are historical experiment entry points rather than a universal one-command dataset downloader.

## Data and reproducibility

Price archives, native TradingView reports, factor downloads, external-project snapshots, model checkpoints, and frozen prospective state are deliberately excluded. Obtain inputs under the relevant provider's terms. A new download can differ from an older snapshot; preserve source checksums and do not claim an exact replication without matching them.

Cross-project adapters use optional environment variables instead of personal filesystem paths:

- `TENSORTRADE_PATTERN_PROJECT`: a separately supplied Pattern_FindR source checkout.
- `TENSORTRADE_QUANT_PROJECT`: a separately supplied quant-strategy checkout.
- `TENSORTRADE_ARCHIVE`: a local market-data archive with its matching manifest.

Defaults point to subdirectories of `external/`, which Git ignores. The corresponding original projects and extracted functions are not bundled or relicensed. Workflows that use those adapters require those inputs; the main demo and unit tests do not.

`catalogue.py` also expects a locally supplied text extraction of the source paper. The paper and its full-text extracts are not distributed. Other source adapters document their upstream URLs in code. SEC and market-data access must follow the respective provider's identification and usage requirements.

Historical registered source hashes refer to the original experiment version. Publication changes to path configuration and reporting do not retroactively change those records. Existing prospective source and local records are preserved; a new user must register a separate paper study with their own inputs.

## Interpretation

Read [research results and limitations](../../docs/RESEARCH_RESULTS.md). Historical screens, quote episodes, synthetic accounting tests, and prospective observations are separate evidence categories. A valid options payoff or a reconciled ledger is not proof that a strategy can be executed profitably.
