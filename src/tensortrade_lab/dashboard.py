"""Read-only local dashboard for completed research artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tensortrade_lab.report import equity_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="TensorTrade Lab", page_icon="◈", layout="wide")
    st.markdown(
        """<style>
    .stApp {background:#0b1118;color:#e4edf5} [data-testid="stSidebar"]{background:#101820}
    h1{letter-spacing:-1px} [data-testid="stMetric"]{background:#121e28;padding:20px;border-radius:10px}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {color:#e4edf5}
    [data-testid="stHeader"]{background:#0b1118}
    @media (max-width: 900px) {
      [data-testid="stMetric"] {padding:12px}
      [data-testid="stMetricValue"] {font-size:1.5rem}
    }
    </style>""",
        unsafe_allow_html=True,
    )
    st.caption("TENSORTRADE / RESEARCH LAB")
    st.title("Strategy workspace")
    st.write("Crypto · stocks · ETFs · simulated execution")
    root = Path(args.runs)
    mode = st.sidebar.radio(
        "Workspace", ["Portfolio rotation", "Focused ETF study", "Research searches", "PPO training runs"]
    )
    if mode == "Portfolio rotation":
        from tensortrade_lab.rotation_report import page

        page(root)
        return
    if mode == "Focused ETF study":
        from tensortrade_lab.focused_ui import focused_page

        focused_page(root)
        return
    if mode == "Research searches":
        from tensortrade_lab.research_ui import research_page

        research_page(root)
        return
    manifests = (
        sorted(root.rglob("manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if root.exists()
        else []
    )
    if not manifests:
        st.info("Create your first experiment with `ttlab demo`, then refresh this page.")
        return
    options = {
        str(p.parent.relative_to(root)): p.parent
        for p in manifests
        if (p.parent / "test/metrics.json").exists()
    }
    if not options:
        st.info("Training is in progress. Results appear after evaluation completes.")
        return
    selection = st.sidebar.selectbox("Experiment", list(options))
    run = options[selection]
    manifest = json.loads((run / "manifest.json").read_text())
    metrics = json.loads((run / "test/metrics.json").read_text())
    if manifest["synthetic"]:
        st.warning("Synthetic-data experiment — validates the software, not market profitability.")
    else:
        st.info("Historical simulation. Evaluate across market regimes before considering deployment.")
    st.sidebar.caption(
        f"PPO · {manifest.get('actual_timesteps', 0):,} steps · seed {manifest['config']['training']['seed']}"
    )
    st.sidebar.code(str(run.resolve()), language=None)
    agent, benchmark = metrics["agent"], metrics["buy_hold"]
    cols = st.columns(4)
    cols[0].metric(
        "Net return",
        f"{agent['total_return']:.2%}",
        f"{(agent['total_return'] - benchmark['total_return']) * 100:+.2f} pp",
        help="Net return difference in percentage points versus buy-and-hold over the same dates.",
    )
    cols[1].metric("Max drawdown", f"{agent['max_drawdown']:.2%}")
    cols[2].metric("Sharpe", f"{agent['sharpe']:.2f}")
    cols[3].metric("Executed fills", f"{agent['fills']:,}")
    overview, trades, learning, setup = st.tabs(
        ["Performance", "Execution ledger", "Learning", "Configuration"]
    )
    with overview:
        st.plotly_chart(equity_figure(run / "test"), width="stretch")
        table = pd.DataFrame(metrics).T[
            ["total_return", "max_drawdown", "sharpe", "fills", "fees", "slippage", "mean_exposure"]
        ]
        st.dataframe(
            table.style.format(
                {
                    "total_return": "{:.2%}",
                    "max_drawdown": "{:.2%}",
                    "sharpe": "{:.2f}",
                    "fills": "{:.0f}",
                    "fees": "{:.2f}",
                    "slippage": "{:.2f}",
                    "mean_exposure": "{:.2%}",
                }
            ),
            width="stretch",
        )
        st.caption(
            "Buy-and-hold uses 100% allocation. Agent and momentum use configured risk limits. "
            "All strategies pay the same fees, slippage, and final liquidation costs."
        )
        st.download_button(
            "Download standalone report",
            (run / "report.html").read_bytes(),
            file_name="tensortrade-report.html",
            mime="text/html",
        )
    with trades:
        fills = pd.read_csv(run / "test/agent_fills.csv")
        st.dataframe(fills, width="stretch", hide_index=True)
        st.download_button("Download fills", fills.to_csv(index=False), "fills.csv", "text/csv")
    with learning:
        validation = pd.DataFrame(json.loads((run / "validation.json").read_text()))
        chart = go.Figure()
        chart.add_trace(
            go.Scatter(x=validation.timesteps, y=validation.total_return, name="Validation return")
        )
        chart.add_trace(go.Scatter(x=validation.timesteps, y=validation.score, name="Selection score"))
        chart.update_layout(template="plotly_dark", yaxis_tickformat=".1%", xaxis_title="Training timesteps")
        st.plotly_chart(chart, width="stretch")
        st.caption(
            "Checkpoint selection uses validation return minus maximum drawdown. Test results do not select the model."
        )
        st.dataframe(validation, hide_index=True)
    with setup:
        st.json(manifest)
        st.caption(
            "The model sees completed candles, executes at the next open, and receives rewards net of costs. "
            "Stops can gap. Liquidity, intrabar paths and market impact are not modeled."
        )


if __name__ == "__main__":
    main()
