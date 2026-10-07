# Contributing

TensorTrade Lab welcomes reproducible bug reports, accounting fixes, and clearly scoped research extensions.

## Setup and checks

Use Python 3.12 and run commands from the repository root:

```bash
uv sync --frozen --extra dev --extra research --no-editable
uv run --frozen --no-editable pytest -q
uv run --frozen --no-editable ruff check src tests scripts
uv run --frozen --no-editable ruff format --check src tests scripts
python scripts/check_release.py
```

The default suite creates synthetic inputs and temporary files. It does not download market data, start trading, or require credentials. Four TradingView reconciliation cases are marked `external_data` and skipped by default. To run them, supply the original matching Yahoo/rotation data under `data/` and native price/report exports under `research/paper151/tradingview/reconciliation/`, then use:

```bash
uv run --frozen --no-editable pytest -q --run-data-tests
```

The fixture deliberately fails if requested inputs are missing or inconsistent. It does not silently substitute synthetic data for these historical comparisons.

The core application and tests follow Ruff formatting. Historical research scripts retain their experiment structure; the release check compiles their Python source, and research unit tests exercise their accounting and causality. Source hashes in existing experiment manifests describe the version used at the time, not later publication packaging changes.

## Development workflow

Open an issue describing a reproducible problem or propose a focused pull request. Include expected behavior, a minimal example, and relevant checks. For changes to accounting, signals, or data timing, add a regression test covering the economic invariant or future-data exclusion.

Keep raw market data, credentials, account state, checkpoints, and local exports out of commits. Use synthetic fixtures or document how authorized users can supply their own inputs. The project has no supported live-money execution interface.

Preserve frozen prospective specifications and event histories. A changed strategy requires a new registration, not an edit to an existing record. Never select a model using the test period or present inspected historical periods as fresh holdouts.

## Reporting sensitive issues

Do not include credentials, private account records, or exploitable details in a public issue. Use GitHub's private vulnerability reporting feature when available. Ordinary correctness issues can use the public issue tracker with a synthetic reproduction.
