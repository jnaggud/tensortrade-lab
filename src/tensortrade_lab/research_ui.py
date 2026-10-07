"""Local experiment controls and inspectable research artifacts."""

import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from tensortrade_lab.report import equity_figure
from tensortrade_lab.sources import catalog


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def launch(arguments, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "console.log").open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "tensortrade_lab.cli", *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    st.session_state["research_process"] = process
    st.success(f"Started. Results will appear in {output.name}. Use Refresh to update progress.")


def research_page(root):
    st.write("Compare assets, timeframes, strategies and risk settings using chronological validation.")
    if st.button("Refresh research"):
        st.rerun()
    process = st.session_state.get("research_process")
    if process and process.poll() is None:
        st.info(f"Research process is running (PID {process.pid}).")
        if st.button("Stop running research"):
            os.killpg(process.pid, signal.SIGTERM)
            st.warning("Stopped. The study can resume; interrupted trials will be retried.")
    studies = (
        sorted(root.rglob("study.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if root.exists()
        else []
    )
    setup, results, data, compute = st.tabs(["New search", "Studies", "Data library", "Compute"])
    with setup:
        records = catalog()
        choices = {
            f"{r['market']['symbol']} · {r['market']['bar_minutes']} min · {r['source']} · {Path(r['path']).name}": r
            for r in records
        }
        with st.form("research_spec"):
            selected = st.multiselect(
                "Datasets (must overlap in time)",
                list(choices),
                default=[k for k in choices if "1440 min" in k],
            )
            frames = st.multiselect("Decision candles, minutes", [15, 60, 1440], default=[1440])
            strategies = st.multiselect(
                "Strategies",
                ["momentum", "mean_reversion", "breakout", "supervised", "ppo"],
                default=["momentum", "mean_reversion", "breakout", "supervised", "ppo"],
            )
            directions = st.multiselect(
                "Position directions", ["long_only", "long_short"], default=["long_only"]
            )
            a, b, c = st.columns(3)
            trials = a.number_input("Total trials", min_value=1, max_value=100000, value=64)
            folds = b.number_input("Validation folds", min_value=2, max_value=12, value=3)
            timesteps = c.number_input(
                "PPO steps per fit", min_value=256, max_value=10000000, value=10000, step=256
            )
            method = st.selectbox("Search method", ["tpe", "grid"])
            device = st.selectbox("Compute routing", ["auto", "cpu", "hybrid", "mps", "cuda"])
            workers = st.number_input(
                "Parallel workers (0 = all available CPU cores)", min_value=0, max_value=256, value=0
            )
            st.caption(
                "Seeds: 17, 42, 83. Final 20% of common dates stays reserved. Long/short assumes borrow availability, 3% annual borrow cost and 30% maintenance margin. Costs are fixed across trials."
            )
            commission = st.number_input(
                "Commission per fill, basis points", min_value=0.0, max_value=100.0, value=5.0
            )
            slippage = st.number_input(
                "Slippage per fill, basis points", min_value=0.0, max_value=100.0, value=5.0
            )
            submitted = st.form_submit_button("Create and run search")
        if submitted:
            if not selected or not frames or not strategies or not directions:
                st.error("Select at least one dataset, timeframe, strategy and direction.")
            else:
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
                output = root / f"research-{stamp}"
                output.mkdir(parents=True)
                spec = {
                    "datasets": [
                        {"name": f"{choices[k]['market']['symbol']}-{i}", "path": choices[k]["path"]}
                        for i, k in enumerate(selected)
                    ],
                    "timeframes": frames,
                    "strategies": strategies,
                    "directions": directions,
                    "trials": int(trials),
                    "folds": int(folds),
                    "method": method,
                    "config": {
                        "compute": {"device": device, "workers": int(workers)},
                        "training": {"total_timesteps": int(timesteps)},
                        "execution": {"commission": commission / 10000, "slippage_bps": slippage},
                        "risk": {"borrow_available": "long_short" in directions, "cooldown_minutes": 240},
                        "features": {
                            "observation_minutes": max(120, min(frames) * 8),
                            "reference_minutes": max(60, min(frames)),
                        },
                        "parameters": {
                            "policy.lookback_minutes": [
                                max(240, min(frames) * 5),
                                max(1440, min(frames) * 20),
                            ],
                            "policy.threshold": [0.001, 0.005],
                        },
                    },
                }
                path = output / "spec.yaml"
                path.write_text(yaml.safe_dump(spec, sort_keys=False))
                launch(["search", "--spec", str(path), "--output", str(output)], output)
    with results:
        if not studies:
            st.info("Completed and running searches appear here.")
        else:
            names = {str(p.parent.relative_to(root)): p.parent for p in studies}
            name = st.selectbox("Research study", list(names))
            study = names[name]
            manifest = read_json(study / "study.json")
            status = read_json(study / "status.json")
            st.caption(
                f"Common dates: {manifest['start'][:10]} to {manifest['end'][:10]} · Holdout starts {manifest['holdout_start'][:10]}"
            )
            st.json(status, expanded=False)
            board_path = study / "leaderboard.csv"
            if board_path.exists():
                board = pd.read_csv(board_path)
                st.dataframe(board, width="stretch", hide_index=True)
                st.download_button(
                    "Download leaderboard", board.to_csv(index=False), f"{name}-leaderboard.csv"
                )
            holdout = read_json(study / "holdout.json")
            if holdout:
                st.subheader("Frozen winner: final holdout")
                st.dataframe(pd.DataFrame(holdout["metrics"]).T, width="stretch")
                st.plotly_chart(equity_figure(study / "finalist/holdout"), width="stretch")
                st.subheader("Cost and delay sensitivity")
                st.dataframe(pd.DataFrame(holdout["stress_tests"]).T, width="stretch")
            elif status.get("state") == "complete":
                st.caption(
                    "Freezing the winner closes this study to further tuning and evaluates the reserved final period."
                )
                if st.button("Freeze winner and evaluate holdout"):
                    launch(["finalize", "--study", str(study)], study)
            with st.expander("Study configuration and data quality"):
                st.json(manifest)
            with st.expander("Latest process output"):
                log = study / "console.log"
                if log.exists():
                    with log.open("rb") as handle:
                        handle.seek(max(0, log.stat().st_size - 12000))
                        st.code(handle.read().decode(errors="replace"))
    with data:
        rows = [
            {
                "asset": r["market"]["symbol"],
                "minutes": r["market"]["bar_minutes"],
                "source": r["source"],
                **r["quality"],
                "path": r["path"],
            }
            for r in catalog()
        ]
        st.dataframe(
            pd.DataFrame(rows).drop(columns=["missing_examples"], errors="ignore"),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Local archives and Yahoo snapshots retain separate hashes and coverage reports. Unknown adjustment conventions must be verified before combining prices."
        )
        st.code(
            "ttlab download --ticker SPY --interval 15m --period 60d --output data/yahoo/SPY_15m_new.parquet\nttlab import-data --source /path/to/history.csv --config configs/stocks.yaml --output data/local/history.parquet",
            language="bash",
        )
    with compute:
        benchmark = read_json(root / "compute-benchmark.json")
        if benchmark:
            st.json(benchmark["hardware"])
            st.dataframe(pd.DataFrame(benchmark["results"]), hide_index=True, width="stretch")
        st.write(
            "Searches use separate processes with one thread per worker by default. Auto/hybrid reserves a GPU slot when available; rule and tree models use CPU. Standalone PPO can use parallel environment processes. Memory and CPU budgets cap concurrency."
        )
        st.caption(
            "GPU cores are scheduled by PyTorch/Metal or CUDA. Small models may run faster on CPU; occupancy is workload-dependent."
        )
