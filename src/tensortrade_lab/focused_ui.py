"""Focused ETF protocol, live fit progress and frozen comparison results."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def comparison_chart(directory):
    chart = go.Figure()
    names = {
        "agent": "Selected strategy",
        "buy_hold": "100% buy & hold",
        "fixed_50": "50% allocation",
        "volatility_12": "12% volatility target",
        "matched_exposure": "Allocation matched on validation",
        "cash": "Cash",
    }
    for key, label in names.items():
        path = directory / f"{key}_equity.csv"
        if path.exists():
            data = pd.read_csv(path)
            chart.add_trace(go.Scatter(x=data.timestamp, y=data.equity / data.equity.iloc[0] - 1, name=label))
    chart.update_layout(
        template="plotly_dark",
        yaxis_tickformat=".0%",
        yaxis_title="Cumulative return after costs",
        height=420,
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 20},
    )
    return chart


def focused_page(root):
    study = root / "focused-etf"
    st.subheader("Daily SPY / QQQ research")
    st.write("A fixed experiment to test excess returns after costs, with a 30% maximum drawdown constraint.")
    status = read_json(study / "status.json") or {"state": "not started"}
    protocol = read_json(study / "protocol.json")
    st.sidebar.button("Refresh results")
    st.sidebar.caption(f"Status: {status['state'].replace('_', ' ')}")
    if not protocol:
        st.info("Start the registered experiment with `ttlab focused`.")
        return
    spec = protocol["spec"]
    st.caption(
        "Training and selection: before 2014 · Reserved historical evaluation: 2014–2015 · Previously seen stress periods: 2016–2026"
    )
    fits = [r for path in study.glob("fits/*/*/seed-*/*/steps-*/result.json") if (r := read_json(path))]
    logs = [
        (path, value)
        for path in study.glob("fits/*/ppo_*/seed-*/*/steps-*/validation.json")
        if (value := read_json(path))
    ]
    columns = st.columns(4)
    columns[0].metric("Completed fits", len(fits))
    cumulative = {}
    for path, rows in logs:
        key = str(path.parent.parent)
        cumulative[key] = max(cumulative.get(key, 0), rows[-1]["timesteps"])
    total_steps = sum(cumulative.values())
    compact_steps = f"{total_steps / 1e6:.2f}M" if total_steps >= 1_000_000 else f"{total_steps / 1000:.0f}k"
    columns[1].metric(
        "PPO steps",
        compact_steps,
        help=f"{total_steps:,} cumulative development steps; resumed stages are not counted twice.",
    )
    columns[2].metric(
        "CPU worker budget", (read_json(study / "hardware.json") or {}).get("cpu_cores", "auto")
    )
    columns[3].metric(
        "Reserved period",
        "Done"
        if (study / "final_results.json").exists()
        else ("Frozen" if (study / "selection.json").exists() else "Sealed"),
        help="Candidate selection is frozen before evaluating the reserved period.",
    )
    if status["state"] == "training":
        completed = sum(row["budget"] == status["budget"] for row in fits)
        st.progress(
            min(1.0, completed / status["jobs"]),
            text=f"Stage {status['stage']}: {completed}/{status['jobs']} fits at {status['budget']:,} PPO steps",
        )
    summary, development, learning, stresses, setup = st.tabs(
        [
            "Reserved evaluation",
            "Development & features",
            "Learning curves",
            "Regime diagnostics",
            "Protocol & forward test",
        ]
    )
    with summary:
        final = read_json(study / "final_results.json")
        if not final:
            st.info("The reserved evaluation opens only after strategy selection is frozen.")
        else:
            st.caption(
                "One diagnostic finalist per ETF. Selection does not authorize deployment or establish an edge."
            )
            report = study / "report.html"
            if report.exists():
                st.download_button(
                    "Download complete report", report.read_bytes(), "focused-etf-report.html", "text/html"
                )
            asset = st.selectbox("ETF", spec["assets"])
            result = final["assets"][asset]
            selected = final["selection"]["assets"][asset]
            st.write(f"Frozen candidate: **{selected['candidate']['name']}**")
            metrics = result["metrics"]
            cols = st.columns(3)
            cols[0].metric("Strategy return", f"{metrics['agent']['total_return']:.2%}")
            cols[1].metric("Buy & hold return", f"{metrics['buy_hold']['total_return']:.2%}")
            cols[2].metric("Excess return", f"{result['excess_return'] * 100:+.2f} pp")
            st.plotly_chart(comparison_chart(study / "finalists" / asset / "holdout"), width="stretch")
            table = pd.DataFrame(metrics).T[
                [
                    "total_return",
                    "max_drawdown",
                    "annualized_volatility",
                    "mean_gross_exposure",
                    "fills",
                    "fees",
                    "slippage",
                ]
            ]
            st.dataframe(
                table.style.format(
                    {
                        key: "{:.2%}"
                        for key in [
                            "total_return",
                            "max_drawdown",
                            "annualized_volatility",
                            "mean_gross_exposure",
                        ]
                    }
                ),
                width="stretch",
            )
            rows = [
                {
                    "scenario": name,
                    "strategy_return": v["agent"]["total_return"],
                    "buy_hold_return": v["buy_hold"]["total_return"],
                    "excess_return": v["excess_return"],
                }
                for name, v in result["stress_tests"].items()
            ]
            st.dataframe(pd.DataFrame(rows).set_index("scenario").style.format("{:.2%}"), width="stretch")
            st.caption(
                "Costs apply equally to each strategy. Allocation benchmarks use quarter-position increments; matched allocation is calibrated on earlier validation, never on this evaluation."
            )
    with development:
        ablation = study / "feature_ablation.csv"
        if ablation.exists():
            st.write("**Feature comparisons at matching training budgets**")
            st.dataframe(pd.read_csv(ablation), hide_index=True, width="stretch")
            st.caption(
                "Paired asset, seed, fold and budget. Cross-minus-own is the change in mean excess return, in decimal units."
            )
        board = study / "leaderboard.csv"
        if board.exists():
            table = pd.read_csv(board)
            st.dataframe(table, hide_index=True, width="stretch")
            st.download_button(
                "Download development comparison", board.read_bytes(), "etf-development.csv", "text/csv"
            )
        else:
            st.info("The aggregate comparison appears after the first stage finishes.")
        st.write(
            "Own-feature and cross-ETF variants share identical dates. Cross features add peer returns, relative strength and peer volatility using only completed peer candles."
        )
        if fits:
            records = [
                {
                    key: r[key]
                    for key in (
                        "asset",
                        "candidate",
                        "seed",
                        "fold",
                        "budget",
                        "total_return",
                        "excess_return",
                        "max_drawdown",
                        "mean_gross_exposure",
                    )
                }
                for r in fits
            ]
            st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
    with learning:
        promotions = read_json(study / "promotion.json")
        if promotions:
            st.dataframe(pd.DataFrame(promotions).drop(columns="criterion"), hide_index=True, width="stretch")
        st.caption(
            "100k → 500k → 2m steps. Promotion requires ≥0.5 percentage-point improvement in best inner-validation score during the second half of a stage in ≥50% of fits. Outer evaluation returns never control promotion."
        )
        if logs:
            options = {str(path.parent.relative_to(study / "fits")): rows for path, rows in logs}
            key = st.selectbox("Learning curve", sorted(options))
            values = pd.DataFrame(options[key])
            chart = go.Figure()
            chart.add_trace(
                go.Scatter(x=values.timesteps, y=values.excess_return, name="Validation excess return")
            )
            chart.add_trace(go.Scatter(x=values.timesteps, y=values.max_drawdown, name="Maximum drawdown"))
            chart.update_layout(
                template="plotly_dark", yaxis_tickformat=".0%", xaxis_title="Cumulative training steps"
            )
            st.plotly_chart(chart, width="stretch")
    with stresses:
        audit = read_json(study / "previously_seen.json")
        st.info(
            "2016–2026 data was used in earlier experiments. These checks are diagnostics, not an untouched test."
        )
        if audit:
            table = [
                {
                    "asset": r["asset"],
                    "regime": r["regime"],
                    "strategy_return": r["metrics"]["agent"]["total_return"],
                    "buy_hold_return": r["metrics"]["buy_hold"]["total_return"],
                    "excess_return": r["excess_return"],
                    "max_drawdown": r["metrics"]["agent"]["max_drawdown"],
                }
                for r in audit["results"]
            ]
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            st.caption(audit["classification"])
        else:
            st.caption("Diagnostics become available after finalist evaluation.")
    with setup:
        prospective = read_json(study / "prospective" / "registration.json")
        st.write(
            "Forward results require decisions recorded before future execution candles. Historical replay cannot provide that evidence."
        )
        if prospective:
            st.json(prospective)
            for asset in spec["assets"]:
                state = read_json(study / "prospective" / f"{asset}.json")
                if state:
                    st.write(
                        f"**{asset}:** {state['status']}; {state['prospective_sessions']} observed sessions, {len(state['signals'])} recorded decisions."
                    )
        st.json(spec)
