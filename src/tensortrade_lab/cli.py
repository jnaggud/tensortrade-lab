import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Config, load_config
from .data import load_bars, synthetic_bars


def parser():
    root = argparse.ArgumentParser(description="TensorTrade Lab: train, evaluate, and replay paper trades")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a complete synthetic-data software check")
    demo.add_argument("--rows", type=int, default=2000)
    demo.add_argument("--timesteps", type=int, default=4096)
    demo.add_argument("--output", default="runs")
    sample = commands.add_parser("sample", help="Generate labeled synthetic OHLCV CSV")
    sample.add_argument("--rows", type=int, default=4000)
    sample.add_argument("--output", default="data/synthetic_hourly.csv")
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--config", default=None)
    for name in ("validate", "train", "walk-forward"):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--config", default=None)
        if name != "validate":
            command.add_argument("--output", default="runs")
            command.add_argument("--timesteps", type=int)
            command.add_argument("--synthetic", action="store_true", help="Label results as synthetic")
        if name == "walk-forward":
            command.add_argument("--folds", type=int, default=3)
    for name in ("backtest", "paper"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True, help="Trusted local checkpoint directory")
        command.add_argument(
            "--data", required=True, help="CSV including indicator warmup and future candles"
        )
        if name == "backtest":
            command.add_argument("--output", default="runs")
        else:
            command.add_argument("--state", default="runs/paper/account.json")
    search = commands.add_parser("search", help="Resume a parallel grid or adaptive TPE study")
    search.add_argument("--spec", required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--trials", type=int, help="Total trial budget, including previously finished trials")
    finalize = commands.add_parser("finalize", help="Freeze the winner and open the reserved holdout once")
    finalize.add_argument("--study", required=True)
    focused = commands.add_parser("focused", help="Run the preregistered daily SPY/QQQ staged experiment")
    focused.add_argument("--spec", default="configs/focused_etf.yaml")
    focused.add_argument("--output", default="runs/focused-etf")
    audit = commands.add_parser(
        "focused-audit", help="Evaluate frozen ETF policies on previously seen market regimes"
    )
    audit.add_argument("--study", default="runs/focused-etf")
    forward = commands.add_parser(
        "focused-forward", help="Record ETF paper decisions before future execution candles"
    )
    forward.add_argument("--study", default="runs/focused-etf")
    forward.add_argument("--spy", default="data/yahoo/SPY_1d.parquet")
    forward.add_argument("--qqq", default="data/yahoo/QQQ_1d.parquet")
    rotation = commands.add_parser("rotation", help="Run the registered multi-asset portfolio experiment")
    rotation.add_argument("--spec", default="configs/portfolio_rotation.yaml")
    rotation.add_argument("--output", default="runs/portfolio-rotation")
    rotation.add_argument("--download", action="store_true", help="Acquire missing Yahoo snapshots first")
    attribution = commands.add_parser("rotation-attribution", help="Decompose the frozen ETF strategy losses")
    attribution.add_argument("--study", default="runs/focused-etf")
    attribution.add_argument("--output", default="runs/portfolio-rotation/attribution")
    rotation_forward = commands.add_parser(
        "rotation-forward", help="Record prospective portfolio paper decisions"
    )
    rotation_forward.add_argument("--study", default="runs/portfolio-rotation")
    rotation_forward.add_argument(
        "--data-directory", help="New immutable snapshots with the same ETF symbols"
    )
    rotation_forward.add_argument(
        "--refresh", action="store_true", help="Download a new Yahoo snapshot before polling"
    )
    commands.add_parser("hardware", help="Show CPU, memory and supported GPU devices")
    bench = commands.add_parser("benchmark", help="Compare CPU and GPU training kernels")
    bench.add_argument("--output", default="runs/compute-benchmark.json")
    catalog = commands.add_parser("catalog", help="List imported datasets and coverage")
    catalog.add_argument("--directory", default="data")
    download = commands.add_parser("download", help="Snapshot yfinance OHLCV and corporate actions")
    download.add_argument("--ticker", required=True)
    download.add_argument("--interval", default="1d")
    download.add_argument("--start")
    download.add_argument("--end")
    download.add_argument("--period")
    download.add_argument("--calendar", choices=["XNYS", "24/7"], default="XNYS")
    download.add_argument("--output", required=True)
    ingest = commands.add_parser("import-data", help="Import CSV, Parquet or Pattern_FindR price_data SQLite")
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--config", required=True, help="Market calendar, interval and adjustment convention")
    ingest.add_argument("--timezone", default="UTC")
    ingest.add_argument("--ticker")
    ingest.add_argument("--interval")
    ui = commands.add_parser("dashboard")
    ui.add_argument("--runs", default="runs")
    ui.add_argument("--port", type=int, default=8501)
    return root


def main():
    args = parser().parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".venv/.matplotlib").resolve()))
    try:
        if args.command == "dashboard":
            dashboard = Path(__file__).with_name("dashboard.py")
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(dashboard),
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(args.port),
                    "--browser.gatherUsageStats",
                    "false",
                    "--",
                    "--runs",
                    args.runs,
                ]
            )
        config = load_config(getattr(args, "config", None))
        if getattr(args, "timesteps", None):
            raw = config.model_dump()
            raw["training"]["total_timesteps"] = args.timesteps
            config = Config.model_validate(raw)
        if args.command == "hardware":
            from .compute import hardware

            print(json.dumps(hardware(), indent=2))
        elif args.command == "benchmark":
            from .compute import benchmark

            print(json.dumps(benchmark(args.output), indent=2))
        elif args.command == "catalog":
            from .sources import catalog

            print(json.dumps(catalog(args.directory), indent=2))
        elif args.command == "download":
            from .sources import download_yahoo

            print(
                json.dumps(
                    download_yahoo(
                        args.ticker,
                        args.output,
                        args.interval,
                        args.start,
                        args.end,
                        args.period,
                        args.calendar,
                    ),
                    indent=2,
                )
            )
        elif args.command == "import-data":
            from .sources import import_local

            print(
                json.dumps(
                    import_local(
                        args.source, args.output, config.market, args.timezone, args.ticker, args.interval
                    ),
                    indent=2,
                )
            )
        elif args.command == "search":
            from .experiments import search

            print(search(args.spec, args.output, args.trials))
        elif args.command == "focused":
            from .focused import run

            print(run(args.spec, args.output))
        elif args.command == "rotation":
            from .rotation import run

            if args.download:
                from .rotation_data import acquire

                acquire(args.spec)
            print(run(args.spec, args.output))
        elif args.command == "rotation-attribution":
            from .rotation_attribution import attribute

            print(json.dumps(attribute(args.study, args.output), indent=2))
        elif args.command == "rotation-forward":
            from .rotation_forward import poll, refresh

            if args.refresh and args.data_directory:
                raise ValueError("Choose --refresh or --data-directory")
            directory = refresh(args.study) if args.refresh else args.data_directory
            print(json.dumps(poll(args.study, directory), indent=2))
        elif args.command == "focused-audit":
            from .etf_audit import audit

            print(json.dumps(audit(args.study), indent=2))
        elif args.command == "focused-forward":
            from .etf_forward import poll

            print(json.dumps(poll(args.study, {"SPY": args.spy, "QQQ": args.qqq}), indent=2))
        elif args.command == "finalize":
            from .experiments import finalize

            print(json.dumps(finalize(args.study), indent=2))
        elif args.command == "sample":
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            synthetic_bars(args.rows, args.seed, config.market.bar_minutes).to_csv(output, index=False)
            output.with_suffix(".metadata.json").write_text(
                json.dumps({"synthetic": True, "seed": args.seed})
            )
            print(output.resolve())
        elif args.command == "validate":
            bars = load_bars(args.data, config.market)
            print(f"Valid: {len(bars)} candles, {bars.timestamp.iloc[0]} → {bars.timestamp.iloc[-1]}")
        elif args.command in ("train", "walk-forward", "demo"):
            from .training import train, walk_forward

            if args.command == "demo":
                output = Path(args.output)
                output.mkdir(parents=True, exist_ok=True)
                source = output / "synthetic_hourly.csv"
                synthetic_bars(args.rows).to_csv(source, index=False)
                source.with_suffix(".metadata.json").write_text(json.dumps({"synthetic": True, "seed": 42}))
                raw = config.model_dump()
                raw["training"].update(n_steps=256, batch_size=64, n_epochs=4, eval_every=1024)
                config = Config.model_validate(raw)
                run = train(source, config, args.output, synthetic=True)
            else:
                sidecar = Path(args.data).with_suffix(".metadata.json")
                synthetic = args.synthetic or (
                    sidecar.exists() and json.loads(sidecar.read_text()).get("synthetic", False)
                )
                if args.command == "train":
                    run = train(args.data, config, args.output, synthetic)
                else:
                    run = walk_forward(args.data, config, args.output, args.folds, synthetic)
            print(f"\nArtifacts: {run.resolve()}")
        elif args.command == "backtest":
            from .paper import backtest

            print(backtest(args.run, args.data, args.output).resolve())
        elif args.command == "paper":
            from .paper import replay

            print(json.dumps(replay(args.run, args.data, args.state), indent=2))
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
