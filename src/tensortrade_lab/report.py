from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {"agent": "#4ee4bc", "buy_hold": "#94a3ff", "momentum": "#f4bc62", "cash": "#778597"}


def equity_figure(results_dir: Path):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=("Portfolio equity", "Drawdown"),
    )
    for name, color in COLORS.items():
        path = results_dir / f"{name}_equity.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        fig.add_trace(
            go.Scatter(
                x=frame.timestamp,
                y=frame.equity,
                name=name.replace("_", " ").title(),
                line={"color": color, "width": 2},
            ),
            row=1,
            col=1,
        )
        drawdown = 1 - frame.equity / frame.equity.cummax()
        fig.add_trace(
            go.Scatter(x=frame.timestamp, y=-drawdown, showlegend=False, line={"color": color, "width": 1.5}),
            row=2,
            col=1,
        )
    fig.update_layout(
        template="plotly_dark",
        height=600,
        paper_bgcolor="#101820",
        plot_bgcolor="#101820",
        margin={"l": 30, "r": 20, "t": 50, "b": 30},
        legend={"orientation": "h"},
    )
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    return fig


def write_report(run: Path, results: dict, synthetic=False):
    directory = run / "test"
    if not directory.exists():
        directory = run
    fig = equity_figure(directory)
    rows = "".join(
        f"<tr><td>{escape(name.replace('_', ' ').title())}</td>"
        f"<td>{m['total_return']:.2%}</td><td>{m['max_drawdown']:.2%}</td>"
        f"<td>{m['sharpe']:.2f}</td><td>{m['fills']}</td><td>{m['fees']:.2f}</td></tr>"
        for name, m in results.items()
    )
    label = (
        "SYNTHETIC DATA · SOFTWARE VALIDATION ONLY" if synthetic else "HISTORICAL SIMULATION · OUT OF SAMPLE"
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TensorTrade Lab · Research report</title><style>
body{{background:#0b1118;color:#e4edf5;font:16px system-ui;margin:0;padding:40px 6%;}}
small{{color:#4ee4bc;letter-spacing:2px}}h1{{font-size:38px}}p{{color:#acb8c7;line-height:1.6}}
table{{width:100%;border-collapse:collapse;background:#101820}}td,th{{padding:16px;text-align:left;border-bottom:1px solid #25303d}}
.card{{border:1px solid #25303d;border-radius:12px;overflow:hidden;margin:28px 0}}footer{{color:#8896a7;font-size:13px}}
</style></head><body><small>TENSORTRADE / RESEARCH LAB</small><h1>Strategy evaluation</h1><p>{label}</p>
<p>Decisions use completed candles. Fills occur at the next open with fees and slippage.
Buy-and-hold invests 100%; the agent and momentum baseline use configured risk limits.</p>
<div class="card">{fig.to_html(full_html=False, include_plotlyjs=True)}</div>
<div class="card"><table><thead><tr><th>Strategy</th><th>Net return</th><th>Max drawdown</th><th>Sharpe</th>
<th>Fills</th><th>Fees</th></tr></thead><tbody>{rows}</tbody></table></div>
<footer>Sharpe uses configured periods per year and zero risk-free rate. A short backtest does not establish
profitability. Fees are included in equity; end-of-sample liquidation is included for every strategy.
Stops are evaluated at candle close and next open; intrabar paths, liquidity and market impact are not modeled.</footer>
</body></html>"""
    (run / "report.html").write_text(html)
