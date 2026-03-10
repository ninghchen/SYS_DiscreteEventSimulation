"""
Bygg & Bo — Monte Carlo Cost Simulation Dashboard
Run: pip install dash plotly numpy scipy pandas
Then: python bygg_bo_dashboard.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ─── Theme ───────────────────────────────────────────────────────────────────
DARK_BG     = "#f7f8fc"
PANEL_BG    = "#ffffff"
BORDER      = "#d9deea"
ACCENT      = "#4f8ef7"
ACCENT2     = "#f76f4f"
ACCENT3     = "#4fcf8e"
TEXT        = "#1f2430"
MUTED       = "#6b7280"
FONT        = "'IBM Plex Mono', monospace"

CARD_STYLE = {
    "background": PANEL_BG,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "16px",
}

# ─── Simulation ──────────────────────────────────────────────────────────────
def run_simulation(N=10_000):
    rho = 0.7

    A = np.random.uniform(40, 80, N)
    B = np.random.triangular(100, 150, 300, N)
    C_ind = np.random.normal(300, 40, N)
    D = np.random.triangular(80, 110, 160, N)
    E = np.random.uniform(90, 210, N)
    F_ind = np.random.normal(200, 30, N)
    G = np.random.triangular(70, 90, 120, N)

    R1_prob = np.random.rand(N) < 0.30
    R2_prob = np.random.rand(N) < 0.10
    R3_prob = np.random.rand(N) < 0.50
    R1_cost = np.where(R1_prob, np.random.normal(200, 50, N), 0)
    R2_cost = np.where(R2_prob, np.random.uniform(100, 500, N), 0)
    R3_cost = np.where(R3_prob, 50, 0)
    risk_cost = R1_cost + R2_cost + R3_cost

    M   = np.random.normal(0, 1, N)
    Z_C = np.random.normal(0, 1, N)
    Z_F = np.random.normal(0, 1, N)
    Corr_C = 300 + 40 * (rho * M + np.sqrt(1 - rho**2) * Z_C)
    Corr_F = 200 + 30 * (rho * M + np.sqrt(1 - rho**2) * Z_F)

    total_corr = A + B + Corr_C + D + E + Corr_F + G + risk_cost
    total_ind  = A + B + C_ind  + D + E + F_ind  + G + risk_cost

    CL   = 0.95
    VaR  = np.percentile(total_corr, (1 - CL) * 100)
    CVaR = total_corr[total_corr >= VaR].mean()

    components = {
        "A: Site Survey": A,
        "B: Demolition":  B,
        "C: Concrete":    Corr_C,
        "D: Electrical":  D,
        "E: Plumbing":    E,
        "F: HVAC":        Corr_F,
        "G: Interior":    G,
        "R1: Asbestos":   R1_cost,
        "R2: Strike":     R2_cost,
        "R3: Penalty":    R3_cost,
    }

    total_var = np.var(total_corr, ddof=0)
    var_contrib = {}
    for name, arr in components.items():
        cov = np.cov(arr, total_corr, ddof=0)[0, 1]
        var_contrib[name] = cov / total_var

    spearman_rows = []
    for name, arr in components.items():
        sp, pval = stats.spearmanr(arr, total_corr)
        spearman_rows.append({"Input": name, "rho": sp, "abs_rho": abs(sp), "pval": pval})
    sp_df = pd.DataFrame(spearman_rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)

    vc_df = pd.DataFrame([
        {"Input": k, "contribution": v} for k, v in var_contrib.items()
    ]).sort_values("contribution", ascending=False).reset_index(drop=True)

    return {
        "total_corr": total_corr,
        "total_ind":  total_ind,
        "mean":       np.mean(total_corr),
        "std_corr":   np.std(total_corr),
        "std_ind":    np.std(total_ind),
        "VaR":        VaR,
        "CVaR":       CVaR,
        "skew":       stats.skew(total_corr),
        "kurt":       stats.kurtosis(total_corr),
        "sp_df":      sp_df,
        "vc_df":      vc_df,
    }

# ─── App Layout ──────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Bygg & Bo · Sim Dashboard")
app.layout = html.Div(style={
    "background": DARK_BG, "minHeight": "100vh",
    "fontFamily": FONT, "color": TEXT, "padding": "28px 32px",
}, children=[

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div(style={"display": "flex", "alignItems": "center",
                    "justifyContent": "space-between", "marginBottom": "28px"}, children=[
        html.Div([
            html.Div("BYGG & BO", style={
                "fontSize": "11px", "letterSpacing": "4px",
                "color": MUTED, "marginBottom": "4px"
            }),
            html.H1("Project 1", style={
                "margin": 0, "fontSize": "26px", "fontWeight": "700",
                "color": TEXT,
            }),
            html.Div("N = 10,000  ·  ρ = 0.7",
                     style={"color": MUTED, "fontSize": "12px", "marginTop": "4px"}),
        ]),
        html.Button("▶  RUN SIMULATION", id="run-btn", n_clicks=0, style={
            "background": ACCENT, "color": "#fff", "border": "none",
            "borderRadius": "6px", "padding": "12px 28px",
            "fontSize": "13px", "fontFamily": FONT, "fontWeight": "700",
            "cursor": "pointer", "letterSpacing": "1px",
        }),
    ]),

    # ── KPI Row ─────────────────────────────────────────────────────────────
    html.Div(id="kpi-row", style={
        "display": "grid",
        "gridTemplateColumns": "repeat(6, 1fr)",
        "gap": "12px",
        "marginBottom": "20px",
    }),

    # ── SD Comparison ───────────────────────────────────────────────────────
    html.Div(id="sd-comparison", style={"marginBottom": "20px"}),

    # ── Histogram (full width) ───────────────────────────────────────────────
    html.Div(id="histogram-panel", style={"marginBottom": "16px"}),

    # ── Tornado + Variance (side by side) ────────────────────────────────────
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px",
                    "marginBottom": "16px"}, children=[
        html.Div(id="tornado-panel"),
        html.Div(id="variance-panel"),
    ]),
])

# ─── Callbacks ───────────────────────────────────────────────────────────────
def kpi_card(label, value, color=TEXT, sub=None):
    return html.Div(style=CARD_STYLE | {"textAlign": "center"}, children=[
        html.Div(label, style={"fontSize": "10px", "color": TEXT,
                               "letterSpacing": "2px", "marginBottom": "8px"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": "700", "color": color}),
        html.Div(sub, style={"fontSize": "10px", "color": TEXT, "marginTop": "4px"}) if sub else None,
    ])


@app.callback(
    Output("kpi-row",         "children"),
    Output("sd-comparison",   "children"),
    Output("histogram-panel", "children"),
    Output("tornado-panel",   "children"),
    Output("variance-panel",  "children"),
    Input("run-btn", "n_clicks"),
)
def update(n_clicks):
    d = run_simulation()

    # ── KPI Cards ───────────────────────────────────────────────────────────
    kpis = [
        kpi_card("MEAN",     f"{d['mean']:.0f} k",   TEXT, "kUSD"),
        kpi_card("σ CORR",   f"{d['std_corr']:.1f}", TEXT, "kUSD"),
        kpi_card("VaR 95%",  f"{d['VaR']:.0f} k",   TEXT, "kUSD"),
        kpi_card("CVaR 95%", f"{d['CVaR']:.0f} k",  TEXT, "kUSD"),
        kpi_card("SKEWNESS", f"{d['skew']:.3f}",     TEXT),
        kpi_card("KURTOSIS", f"{d['kurt']:.3f}",     TEXT),
    ]

    # ── SD Comparison ───────────────────────────────────────────────────────
    diff = d["std_corr"] - d["std_ind"]
    pct  = diff / d["std_ind"] * 100

    sd_panel = html.Div(style=CARD_STYLE, children=[
        html.Div("σ Correlated v. σ Independent", style={
            "fontSize": "10px", "letterSpacing": "3px", "color": TEXT, "marginBottom": "16px"
        }),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px",
                        "alignItems": "center"}, children=[
            html.Div([
                html.Div("σ Correlated", style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px"}),
                html.Div(f"{d['std_corr']:.2f} kUSD", style={"color": ACCENT, "fontWeight": "700",
                                                              "marginTop": "6px", "fontSize": "18px"}),
            ]),
            html.Div([
                html.Div("σ Independent", style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px"}),
                html.Div(f"{d['std_ind']:.2f} kUSD", style={"color": ACCENT2, "fontWeight": "700",
                                                             "marginTop": "6px", "fontSize": "18px"}),
            ]),
            html.Div(style={"textAlign": "center", "borderLeft": f"1px solid {BORDER}",
                            "paddingLeft": "24px"}, children=[
                html.Div("Δ (Corr − Ind)", style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px"}),
                html.Div(f"+{diff:.2f} kUSD", style={"color": TEXT, "fontWeight": "700", "fontSize": "22px"}),
            ]),
        ]),
    ])

    # ── Histogram (full width) ───────────────────────────────────────────────
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=d["total_ind"], nbinsx=90, histnorm="probability density",
        name="Independent", marker_color=ACCENT2,
        opacity=0.55, hovertemplate="%{x:.0f} kUSD<extra>Independent</extra>",
    ))
    fig_hist.add_trace(go.Histogram(
        x=d["total_corr"], nbinsx=90, histnorm="probability density",
        name="Correlated (ρ=0.7)", marker_color=ACCENT,
        opacity=0.55, hovertemplate="%{x:.0f} kUSD<extra>Correlated</extra>",
    ))
    fig_hist.add_vline(x=d["VaR"],  line_color=TEXT, line_dash="dash", line_width=2,
                       annotation_text=f"VaR={d['VaR']:.0f}", annotation_font_color=TEXT,
                       annotation_position="top right")
    fig_hist.add_vline(x=d["CVaR"], line_color=TEXT, line_dash="dot",  line_width=2,
                       annotation_text=f"CVaR={d['CVaR']:.0f}", annotation_font_color=TEXT,
                       annotation_position="top right")
    fig_hist.update_layout(
        barmode="overlay", title="Overlay Histogram — Total Cost Distribution",
        xaxis_title="Total Cost (kUSD)", yaxis_title="Density",
        legend=dict(x=1, xanchor="right", y=1, yanchor="top", font_size=11),
        **_chart_layout(height=400),
    )
    hist_panel = html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig_hist, config={"displayModeBar": False})])

    # ── Tornado Chart ────────────────────────────────────────────────────────
    sp = d["sp_df"]
    colors_tornado = [ACCENT2 if v > 0 else ACCENT for v in sp["rho"]]
    fig_torn = go.Figure(go.Bar(
        x=sp["rho"],
        y=sp["Input"],
        orientation="h",
        marker_color=colors_tornado,
        marker_line_color=BORDER,
        marker_line_width=0.5,
        text=[f"{v:+.3f}" for v in sp["rho"]],
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
        hovertemplate="%{y}<br>ρ = %{x:.3f}<extra></extra>",
    ))
    fig_torn.update_layout(
        title="Tornado Chart — Spearman Correlation with Total Cost",
        xaxis_title="Spearman ρ", yaxis_autorange="reversed",
        **_chart_layout(height=420),
    )
    fig_torn.add_vline(x=0, line_color=MUTED, line_width=1)
    torn_panel = html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig_torn, config={"displayModeBar": False})])

    # ── Variance Contribution (horizontal, matching tornado) ─────────────────
    vc = d["vc_df"]
    vc_sorted = vc.sort_values("contribution", ascending=True).reset_index(drop=True)
    #vc_colors = [ACCENT if v > 0.15 else (ACCENT2 if v > 0.05 else MUTED)
    #             for v in vc_sorted["contribution"]]
    vc_colors = MUTED
    fig_vc = go.Figure(go.Bar(
        x=vc_sorted["contribution"] * 100,
        y=vc_sorted["Input"],
        orientation="h",
        marker_color=vc_colors,
        marker_line_color=BORDER,
        marker_line_width=0.5,
        text=[f"{v*100:.1f}%" for v in vc_sorted["contribution"]],
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
        hovertemplate="%{y}<br>Contribution: %{x:.2f}%<extra></extra>",
    ))
    fig_vc.update_layout(
        title="Variance Contribution (Covariance Method)",
        xaxis_title="Variance Contribution (%)",
        yaxis_autorange="reversed",
        **_chart_layout(height=420),
    )
    fig_vc.add_vline(x=0, line_color=MUTED, line_width=1)
    vc_panel = html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig_vc, config={"displayModeBar": False})])

    return kpis, sd_panel, hist_panel, torn_panel, vc_panel


def _chart_layout(height=400):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT, size=11),
        title_font=dict(size=13, color=TEXT),
        margin=dict(l=10, r=60, t=50, b=10),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        height=height,
    )


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
