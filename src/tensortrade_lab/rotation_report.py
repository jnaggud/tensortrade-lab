"""Readable portfolio-study report shared by the dashboard and standalone HTML."""

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def read_study(output):
    output = Path(output)
    return json.loads((output / "results.json").read_text())


def comparison_rows(results, section="confirmation"):
    return pd.DataFrame(
        [
            {"candidate": r["candidate"]["name"], "period": r["fold"]["name"], **r["metrics"]}
            for r in results[section]
        ]
    )


def equity_chart(output, results):
    output = Path(output)
    name = results["selection"]["candidate"]["name"]
    fold = results["confirmation"][0]["fold"]["name"]
    fig = go.Figure()
    for candidate, label, color in (
        (name, "Frozen rotation finalist", "#60d4bc"),
        ("spy", "SPY with dividends reinvested", "#a7aef9"),
        ("passive", "Same-universe passive portfolio", "#edae64"),
    ):
        frame = pd.read_csv(output / "confirmation" / candidate / fold / "equity.csv")
        fig.add_trace(go.Scatter(x=frame.timestamp, y=frame.equity, name=label, line={"color": color}))
    fig.update_layout(
        template="plotly_dark",
        title="Recent period • previously seen market history",
        yaxis_title="Portfolio value ($)",
        legend={"orientation": "h"},
        height=460,
    )
    return fig


def write_report(output):
    output = Path(output)
    results = read_study(output)
    protocol = json.loads((output / "protocol.json").read_text())
    selected = results["selection"]["candidate"]["name"]
    board = pd.DataFrame(results["leaderboard"])
    columns = [
        "candidate",
        "period",
        "total_return",
        "cagr",
        "max_drawdown",
        "volatility",
        "sharpe",
        "fees",
        "slippage",
        "interest",
        "fills",
    ]
    recent = comparison_rows(results)
    selected_metrics = recent[recent.candidate == selected].iloc[0]
    spy = recent[recent.candidate == "spy"].iloc[0]
    attribution_path = output / "attribution" / "attribution.json"
    attribution = json.loads(attribution_path.read_text()) if attribution_path.exists() else []
    attr_table = pd.DataFrame(
        [
            {
                "asset": r["asset"],
                **{k: 100 * v for k, v in r["components"].items()},
                "total_excess_pp": 100 * r["excess"],
            }
            for r in attribution
        ]
    )
    note = (
        "Retrospective research, not a fresh holdout or evidence of a deployable edge. "
        "The finalist was frozen using 2010–2022 development folds before recent-period evaluation. "
        "All ten fixed candidates remain visible; recent results do not reselect the finalist. "
        "The joint block-bootstrap adjustment covers this ten-candidate family only. Prior searches and "
        "the choice of surviving ETFs add uncorrected researcher discretion. Future paper evidence is pending."
    )
    decision = (
        "Simple-strategy gate passed; conditional PPO comparison completed."
        if results["selection"]["passes_rl_gate"]
        else "No candidate passed the registered evidence and risk gate. Further PPO training was not started."
    )
    markdown = f"""# Portfolio rotation research

{decision}

Frozen diagnostic candidate: `{selected}`.

Recent period ({selected_metrics["start"]} through {selected_metrics["end"]}):
- Strategy: {selected_metrics["total_return"]:.2%}; CAGR {selected_metrics["cagr"]:.2%}; maximum drawdown {selected_metrics["max_drawdown"]:.2%}.
- SPY with dividend reinvestment: {spy["total_return"]:.2%}; CAGR {spy["cagr"]:.2%}; maximum drawdown {spy["max_drawdown"]:.2%}.

{note}

Data coverage: {protocol["data"]["common_start"]} through {protocol["data"]["common_end"]}.
Eight rotation configurations and two fixed supervised variants; three development folds, one recent-period diagnostic, and three finalist stress scenarios.
All portfolios pay 5 bps commission plus 5 bps slippage per side; common terminal liquidation. Cash earns a lagged Treasury-bill proxy. Dividends credited ex-date; passive benchmarks reinvest at the close. These are execution assumptions, not broker guarantees.

## Attribution of the old strategies

"""
    for row in attribution:
        markdown += (
            f"- {row['asset']}: net excess {row['excess']:.2%}; strategy cost effect "
            f"{row['components']['strategy_cost_effect']:.2%}; timing versus "
            f"{row['constant_fraction']:.0%} constant allocation "
            f"{row['components']['timing_vs_constant_allocation']:.2%}. "
            "Sequential diagnostic decomposition, not causal attribution.\n"
        )
    verification_path = output / "reconciliation.json"
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else None
    if verification:
        markdown += (
            f"\n## Accounting verification\n\nIndependent reconstruction passed for "
            f"{verification['portfolios_checked']} portfolios. Maximum absolute discrepancy: "
            f"${verification['maximum_absolute_error']:.12f}. See reconciliation.json.\n"
        )
    (output / "report.md").write_text(markdown)
    sections = [
        f"<h1>Portfolio rotation research</h1><p class='callout'>{html.escape(decision)}</p>",
        f"<p>{html.escape(note)}</p><p>Frozen finalist: <b>{html.escape(selected)}</b></p>",
        equity_chart(output, results).to_html(full_html=False, include_plotlyjs=True),
        (
            "<h2>Development selection</h2><p>CAGR differences are annual percentage-point differences. "
            "Risk gate: every fold's drawdown ≤ SPY + 2 points and volatility ≤ 110% of SPY; "
            "positive excess in ≥2/3 folds; mean CAGR above both passive benchmarks; adjusted p ≤ 0.05.</p>"
        ),
        board.to_html(index=False, float_format=lambda v: f"{v:.5f}"),
        "<h2>Recent-period results • all fixed candidates</h2>",
        recent[columns].to_html(index=False, float_format=lambda v: f"{v:.4f}"),
        "<h2>Development regimes</h2>",
        comparison_rows(results, "development")[columns].to_html(
            index=False, float_format=lambda v: f"{v:.4f}"
        ),
        "<h2>Original SPY/QQQ losses • attribution in percentage points</h2>",
        attr_table.to_html(index=False, float_format=lambda v: f"{v:.2f}"),
        (
            "<p>The timing term is relative to an ex post, quantized constant-allocation reference. "
            "Components telescope exactly to the original net excess; they are order-dependent counterfactuals.</p>"
        ),
        "<h2>Stress tests</h2><pre>" + html.escape(json.dumps(results["stress"], indent=2)) + "</pre>",
        "<h2>RL decision</h2><pre>" + html.escape(json.dumps(results["rl"], indent=2)) + "</pre>",
        (
            "<h2>Data and assumptions</h2><p>Only common, complete sessions are used. "
            "No price forward filling. Split-adjusted OHLC with explicit dividends; Yahoo may revise history. "
            "Fixed surviving-fund universe; not a historical all-ETF universe. "
            "Daily features, next-open execution, no leverage, shorting, taxes or market-impact model.</p>"
        ),
        "<pre>" + html.escape(json.dumps(protocol["data"], indent=2)) + "</pre>",
    ]
    if verification:
        sections.append(
            f"<h2>Independent accounting verification</h2><p>Passed for "
            f"{verification['portfolios_checked']} portfolios; maximum absolute error "
            f"${verification['maximum_absolute_error']:.12f}.</p>"
        )
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Portfolio rotation research</title>"
        "<style>body{background:#0b1118;color:#e4edf5;font:16px system-ui;max-width:1500px;margin:40px auto;padding:20px}"
        "h1,h2{color:#60d4bc}.callout{background:#182b38;padding:20px}table{border-collapse:collapse;font-size:12px}"
        "td,th{padding:8px;border-bottom:1px solid #34414c}pre{white-space:pre-wrap;font-size:12px}"
        "</style>" + "\n".join(sections)
    )
    return output / "report.html"


def page(root):
    import streamlit as st

    studies = [
        p.parent
        for p in Path(root).glob("*/preregistration.json")
        if "universe" in json.loads(p.read_text()).get("spec", {})
    ]
    if not studies:
        st.info("No registered portfolio study yet.")
        return
    output = st.sidebar.selectbox("Portfolio study", studies, format_func=lambda p: p.name)
    st.header("Portfolio rotation")
    st.markdown(
        "<style>[data-testid='stMetricValue']{font-size:1.55rem!important}</style>", unsafe_allow_html=True
    )
    if not (output / "results.json").exists():
        st.info("The registered portfolio experiment is running.")
        if (output / "status.json").exists():
            st.json(json.loads((output / "status.json").read_text()))
        return
    results = read_study(output)
    selected = results["selection"]["candidate"]["name"]
    recent = comparison_rows(results)
    metrics = recent[recent.candidate == selected].iloc[0]
    spy = recent[recent.candidate == "spy"].iloc[0]
    if not results["selection"]["passes_rl_gate"]:
        st.warning(
            "No strategy passed the evidence and risk gate. This is a diagnostic finalist; no new PPO run was justified."
        )
    else:
        st.info(
            "Simple-strategy gate passed. Review the incremental PPO comparison before drawing conclusions."
        )
    st.caption("Retrospective research • previously seen market regimes • future paper evidence pending")
    cards = st.columns(4)
    cards[0].metric("Strategy return", f"{metrics.total_return:.2%}")
    cards[1].metric("SPY return", f"{spy.total_return:.2%}")
    cards[2].metric("CAGR gap", f"{(metrics.cagr - spy.cagr) * 100:+.2f} pp")
    cards[3].metric("Drawdown", f"{metrics.max_drawdown:.2%}")
    performance_tab, selection_tab, attribution_tab, protocol_tab = st.tabs(
        ["Portfolio comparison", "Development & RL", "Why the old strategies lost", "Protocol & forward"]
    )
    with performance_tab:
        st.write(f"Frozen candidate: **{selected}**")
        st.plotly_chart(equity_chart(output, results), width="stretch")
        st.dataframe(
            recent[
                ["candidate", "cagr", "total_return", "max_drawdown", "volatility", "sharpe", "fills"]
            ].style.format({k: "{:.2%}" for k in ("cagr", "total_return", "max_drawdown", "volatility")}),
            hide_index=True,
            width="stretch",
        )
        st.json(results["stress"], expanded=False)
    with selection_tab:
        st.dataframe(pd.DataFrame(results["leaderboard"]), hide_index=True, width="stretch")
        st.caption(
            "Adjustment covers these ten candidates only. It does not remove earlier researcher discretion."
        )
        st.json(results["rl"], expanded=False)
        st.dataframe(comparison_rows(results, "development"), hide_index=True)
    with attribution_tab:
        path = output / "attribution/attribution.json"
        if path.exists():
            for row in json.loads(path.read_text()):
                st.subheader(row["asset"])
                st.write(
                    f"Net excess: **{row['excess']:.2%}**. Zero-cost strategy return: **{row['gross_strategy']:.2%}**."
                )
                st.dataframe(
                    pd.DataFrame(
                        [{"component": k, "percentage_points": 100 * v} for k, v in row["components"].items()]
                    ),
                    hide_index=True,
                )
                st.caption(row["note"])
    with protocol_tab:
        verification = output / "reconciliation.json"
        if verification.exists():
            checked = json.loads(verification.read_text())
            st.success(
                f"Independent accounting reconciliation passed for {checked['portfolios_checked']} portfolios."
            )
        st.json(json.loads((output / "preregistration.json").read_text()), expanded=False)
        st.json(json.loads((output / "protocol.json").read_text()), expanded=False)
        forward = output / "prospective/status.json"
        if forward.exists():
            st.json(json.loads(forward.read_text()))
        st.write(
            "Historical simulations cannot create prospective evidence. Paper decisions must be recorded before execution. No live orders are sent."
        )
    if (output / "report.html").exists():
        st.download_button(
            "Download complete report",
            (output / "report.html").read_bytes(),
            "portfolio-rotation.html",
            "text/html",
        )
