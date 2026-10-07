"""Standalone report from already frozen research artifacts."""

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def write_report(study):
    study = Path(study)
    final = json.loads((study / "final_results.json").read_text())
    spec = json.loads((study / "protocol.json").read_text())["spec"]
    synthetic = any(
        json.loads((study / "finalists" / asset / "manifest.json").read_text()).get("synthetic", False)
        for asset in final["assets"]
    )
    board = pd.read_csv(study / "leaderboard.csv")
    promotions = json.loads((study / "promotion.json").read_text())
    fits = [json.loads(p.read_text()) for p in study.glob("fits/*/*/seed-*/*/steps-*/result.json")]
    unique = {}
    for row in fits:
        key = (row["asset"], row["candidate"], row["seed"], row["fold"])
        unique[key] = max(unique.get(key, 0), row["actual_steps"])
    steps = sum(unique.values())
    ablations = []
    for asset in final["assets"]:
        for family in ("ppo", "tree"):
            own = {
                (r["seed"], r["fold"], r["budget"]): r
                for r in fits
                if r["asset"] == asset and r["candidate"] == f"{family}_own"
            }
            cross = {
                (r["seed"], r["fold"], r["budget"]): r
                for r in fits
                if r["asset"] == asset and r["candidate"] == f"{family}_cross"
            }
            common = own.keys() & cross.keys()
            for budget in sorted({key[2] for key in common}):
                keys = sorted(key for key in common if key[2] == budget)
                a = sum(own[key]["excess_return"] for key in keys) / len(keys)
                b = sum(cross[key]["excess_return"] for key in keys) / len(keys)
                ablations.append(
                    {
                        "asset": asset,
                        "family": family,
                        "budget": budget if family == "ppo" else 0,
                        "paired_fits": len(keys),
                        "own_mean_excess": a,
                        "cross_mean_excess": b,
                        "cross_minus_own": b - a,
                    }
                )
    ablation_table = pd.DataFrame(
        ablations,
        columns=[
            "asset",
            "family",
            "budget",
            "paired_fits",
            "own_mean_excess",
            "cross_mean_excess",
            "cross_minus_own",
        ],
    )
    ablation_table.to_csv(study / "feature_ablation.csv", index=False)
    lines = [
        "# Focused ETF results",
        "",
        "Frozen historical experiment; forward evidence remains pending.",
        "",
        f"{len(fits)} completed staged fits, {len(unique)} distinct candidate/asset/seed/fold fits, and {steps:,} PPO development steps.",
        "",
        "| ETF | Frozen candidate | Strategy return | Buy-and-hold | Excess | Strategy drawdown | Development eligible |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    if synthetic:
        lines.insert(2, "**SYNTHETIC SOFTWARE CHECK — not market performance evidence.**")
    sections = []
    names = {
        "agent": "Selected strategy",
        "buy_hold": "100% buy & hold",
        "fixed_50": "50% allocation",
        "volatility_12": "12% volatility target",
        "matched_exposure": "Validation-matched allocation",
        "cash": "Cash",
    }
    for i, (asset, result) in enumerate(final["assets"].items()):
        chosen = final["selection"]["assets"][asset]
        candidate = chosen["candidate"]["name"]
        metric = result["metrics"]
        a, b = metric["agent"], metric["buy_hold"]
        eligible = chosen["development"]["eligible"]
        lines.append(
            f"| {asset} | {candidate} | {a['total_return']:.2%} | {b['total_return']:.2%} | {result['excess_return'] * 100:+.2f} pp | {a['max_drawdown']:.2%} | {'Yes' if eligible else 'No'} |"
        )
        chart = go.Figure()
        for key, label in names.items():
            frame = pd.read_csv(study / "finalists" / asset / "holdout" / f"{key}_equity.csv")
            chart.add_trace(
                go.Scatter(x=frame.timestamp, y=frame.equity / frame.equity.iloc[0] - 1, name=label)
            )
        chart.update_layout(
            template="plotly_dark",
            yaxis_tickformat=".0%",
            margin={"t": 20},
            height=430,
            legend={"orientation": "h", "y": -0.25},
        )
        table = pd.DataFrame(metric).T[
            ["total_return", "max_drawdown", "annualized_volatility", "mean_gross_exposure", "fills"]
        ]
        formatted = table.copy()
        for col in list(formatted)[:-1]:
            formatted[col] = formatted[col].map(lambda v: f"{v:.2%}")
        stress = pd.DataFrame(
            [
                {
                    "scenario": name,
                    "strategy": f"{row['agent']['total_return']:.2%}",
                    "buy_hold": f"{row['buy_hold']['total_return']:.2%}",
                    "excess": f"{row['excess_return'] * 100:+.2f} pp",
                }
                for name, row in result["stress_tests"].items()
            ]
        )
        sections.append(
            f"<section><h2>{asset} · {html.escape(candidate)}</h2><p>Reserved period: 2014–2015. Excess return: <strong>{result['excess_return'] * 100:+.2f} percentage points</strong>. Development eligibility: {'passed' if eligible else 'failed'}.</p>{chart.to_html(full_html=False, include_plotlyjs=(i == 0))}<div class='table'>{formatted.to_html(escape=True)}</div><h3>Equal-cost stress tests</h3><div class='table'>{stress.to_html(index=False, escape=True)}</div></section>"
        )
    lines += [
        "",
        "All returns above cover the reserved 2014–2015 period, after 5 bps commission and 5 bps slippage per side. Strategies and benchmarks share dates and liquidation assumptions.",
        "",
        "## Budget decisions",
        "",
    ]
    for row in promotions:
        lines.append(
            f"- {row['candidate']} at {row['budget']:,} steps: {row['fraction_improving']:.1%} of fits improved enough; {'promotion criterion met' if row['advance'] else 'stopped at this budget'}."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Selection was frozen before the reserved period was opened. A diagnostic finalist does not mean a strategy passed eligibility. No deployment is approved. The 30% drawdown criterion is a selection constraint, not a guaranteed stop. Bootstrap intervals describe development resamples and are not corrected for model selection.",
        "",
        "Own/cross feature comparisons, every seed, and all development folds are preserved in leaderboard.csv and development_results.json. Different training budgets must be considered when comparing final PPO variants.",
        "",
        "2016–2026 checks in previously_seen.json use previously explored history and are diagnostics, not a fresh holdout. Prospective validation requires future candles and timely recorded decisions; historical replay cannot complete it.",
    ]
    (study / "report.md").write_text("\n".join(lines) + "\n")
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Focused ETF results</title><style>body{font-family:system-ui;background:#0b1118;color:#e4edf5;max-width:1180px;margin:0 auto;padding:32px}h1{font-size:36px}p{line-height:1.6}section{background:#121e28;border-radius:12px;padding:22px;margin:24px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:right;border-bottom:1px solid #30404b}th:first-child,td:first-child{text-align:left}small{color:#abc0cf}</style></head><body>"""
    document += f"<h1>SPY &amp; QQQ: frozen research results</h1><p>{steps:,} PPO development steps · {len(spec['folds'])} chronological folds · {len(spec['seeds'])} seeds · Full-allocation and comparable-risk baselines.</p><p>Reserved historical evaluation: <strong>{html.escape(spec['holdout_start'])} to {html.escape(spec['holdout_end'])} (end exclusive)</strong>. Forward evidence remains pending. TensorTrade does not guarantee an edge, and selection does not authorize deployment.</p>"
    if synthetic:
        document += "<p><strong>SYNTHETIC SOFTWARE CHECK — not market performance evidence.</strong></p>"
    document += "".join(sections)
    document += f"<section><h2>Feature comparisons at matching budgets</h2><p>Paired asset, seed, fold and training budget. Values are decimal returns. Extra training in only one variant is excluded from this comparison.</p><div class='table'>{ablation_table.to_html(index=False, escape=True, float_format=lambda v: f'{v:.4f}')}</div></section>"
    document += f"<section><h2>Development comparison</h2><div class='table'>{board.to_html(index=False, escape=True, float_format=lambda v: f'{v:.4f}')}</div><p>Return, exposure and drawdown columns are decimal fractions. Eligibility requires the registered excess-return and drawdown conditions.</p></section>"
    document += f"<section><h2>Training budget decisions</h2><div class='table'>{pd.DataFrame(promotions).drop(columns='criterion').to_html(index=False, escape=True)}</div></section>"
    audit_path = study / "previously_seen.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        rows = [
            {
                "ETF": r["asset"],
                "Period": r["regime"],
                "Strategy": f"{r['metrics']['agent']['total_return']:.2%}",
                "Buy & hold": f"{r['metrics']['buy_hold']['total_return']:.2%}",
                "Excess (pp)": f"{r['excess_return'] * 100:+.2f}",
                "Drawdown": f"{r['metrics']['agent']['max_drawdown']:.2%}",
            }
            for r in audit["results"]
        ]
        document += f"<section><h2>Previously seen regime diagnostics</h2><p>{html.escape(audit['classification'])}</p><div class='table'>{pd.DataFrame(rows).to_html(index=False, escape=True)}</div></section>"
    document += "<p><small>Cash earns zero; buy-and-hold dividends remain in cash. Volatility and exposure targets use quarter increments. Taxes, real liquidity and variable market impact are not modeled. Historical bootstrap intervals are descriptive and not adjusted for strategy selection.</small></p></body></html>"
    (study / "report.html").write_text(document)
    return study / "report.html"
