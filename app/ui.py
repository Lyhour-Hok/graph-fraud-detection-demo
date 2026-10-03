"""Shared Streamlit helpers: cached engine, charts and tables."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.engine import (GRAPHS, OP_LABEL, OPERATORS, TRACK_LABEL, Engine)

# Single-hue sequential ramp for P(fraud): light (near 0) -> dark red (near 1)
PROB_SCALE = [[0.0, "#fbeaea"], [0.25, "#f3b4b0"], [0.5, "#e66767"], [0.75, "#b83232"], [1.0, "#6e1414"]]
NODE_STYLE = {
    "customer":    {"color": "#2a78d6", "symbol": "circle",  "name": "Customer"},
    "merchant":    {"color": "#1baf7a", "symbol": "square",  "name": "Merchant"},
    "transaction": {"color": "#8a8983", "symbol": "diamond", "name": "Transaction"},
}


@st.cache_resource(show_spinner="Loading embeddings and classifiers (first visit only)...")
def get_engine():
    return Engine()


def prob_color(p):
    """Interpolate PROB_SCALE at p in [0, 1]."""
    if p is None or np.isnan(p):
        return "#c9c8c2"
    stops = [s for s, _ in PROB_SCALE]
    cols = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for _, c in PROB_SCALE]
    j = min(np.searchsorted(stops, p, side="right"), len(stops) - 1)
    lo, hi = stops[j - 1], stops[j]
    t = 0 if hi == lo else (p - lo) / (hi - lo)
    rgb = [round(a + (b - a) * t) for a, b in zip(cols[j - 1], cols[j])]
    return "#%02x%02x%02x" % tuple(rgb)


def label_at(p, threshold):
    if p is None or np.isnan(p):
        return "n/a"
    return "⚑ Fraud" if p >= threshold else "Genuine"


def prob_heatmap(engine, scores, track, threshold, title):
    """2 graphs x 4 operators heatmap of P(fraud) for one transaction and one track."""
    z = [[scores[g].probs[track][op] for op in OPERATORS] for g in GRAPHS]
    text, hover = [], []
    for g, row in zip(GRAPHS, z):
        text.append([("n/a" if np.isnan(p) else f"<b>{p:.2f}</b><br>{label_at(p, threshold)}") for p in row])
        hrow = []
        for op, p in zip(OPERATORS, row):
            prec, rec = engine.precision_recall_at(g, track, op, threshold)
            hrow.append(f"{g.title()} · {OP_LABEL[op]}<br>P(fraud) = {'n/a' if np.isnan(p) else f'{p:.3f}'}"
                        f"<br>At threshold {threshold:.2f} on the test set:<br>"
                        f"precision {prec:.3f} · recall {rec:.3f}")
        hover.append(hrow)
    fig = go.Figure(go.Heatmap(
        z=z, x=[OP_LABEL[o] for o in OPERATORS], y=[g.title() for g in GRAPHS],
        zmin=0, zmax=1, colorscale=PROB_SCALE, xgap=2, ygap=2,
        text=text, texttemplate="%{text}", hovertext=hover, hoverinfo="text",
        colorbar=dict(title="P(fraud)", thickness=10, len=0.9),
    ))
    fig.update_traces(textfont=dict(size=13))     # plotly picks dark/light ink per cell
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=250,
                      margin=dict(l=90, r=10, t=40, b=35),
                      xaxis=dict(type="category", tickfont=dict(size=13)),
                      yaxis=dict(type="category", autorange="reversed", tickfont=dict(size=13)))
    return fig


def pr_table(engine, tracks, threshold):
    """Test-set precision/recall each classifier would have had at `threshold`."""
    rows = []
    for g in GRAPHS:
        for op in OPERATORS:
            row = {"Graph": g.title(), "Operator": OP_LABEL[op]}
            for t in tracks:
                p, r = engine.precision_recall_at(g, t, op, threshold)
                short = "Thesis" if t == "thesis" else "Enhanced"
                row[f"{short} precision"] = p
                row[f"{short} recall"] = r
            rows.append(row)
    return pd.DataFrame(rows)


def _arc(center, n, start_deg, end_deg, radius):
    if n == 0:
        return []
    angles = np.linspace(np.radians(start_deg), np.radians(end_deg), n) if n > 1 else [np.radians((start_deg + end_deg) / 2)]
    return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]


def neighborhood_figure(engine, graph, scored, track, op, cc_num, merchant):
    gm = engine.graphs[graph]
    edges = gm.neighborhood(scored, track, op)
    focal = scored.edges
    cust = focal[0][0]
    merch = focal[-1][1]
    tx_focal = focal[0][1] if graph == "tripartite" else None

    pos = {cust: (-1.0, 0.0), merch: (1.0, 0.0)}
    if tx_focal:
        pos[tx_focal] = (0.0, 0.0)
    around_c = [e["v"] for e in edges if e["kind"] == "neighbor" and e["u"] == cust]
    around_m = [e["v"] for e in edges if e["kind"] == "neighbor" and e["u"] == merch]
    for k, xy in zip(around_c, _arc(pos[cust], len(around_c), 105, 255, 0.8)):
        pos[k] = xy
    for k, xy in zip(around_m, _arc(pos[merch], len(around_m), -75, 75, 0.8)):
        pos[k] = xy

    def kind_of(key):
        if key.startswith("new-"):
            return key[4:]
        if key in gm.customer_key.values():
            return "customer"
        if key in gm.merchant_key.values():
            return "merchant"
        return "transaction"

    def describe(key):
        kind = kind_of(key)
        if key == "new-transaction":
            return "This transaction (new node)"
        if key.startswith("new-"):
            return f"Unseen {kind} (not in the graph)"
        if kind == "transaction":
            r = gm.tx.iloc[gm.tx_row[key]]
            return f"Transaction {r.trans_num[:8]}…<br>amt ${r.amt:,.2f} · {'fraud' if r.is_fraud else 'genuine'}"
        name = gm.key_name[key]
        return f"{kind.title()}: {name.replace('fraud_', '') if kind == 'merchant' else name}"

    fig = go.Figure()
    mid_x, mid_y, mid_text = [], [], []
    for e in sorted(edges, key=lambda e: e["kind"] == "focal"):      # focal drawn on top
        (x0, y0), (x1, y1) = pos[e["u"]], pos[e["v"]]
        focal_e = e["kind"] == "focal"
        new_e = focal_e and (scored.confidence != "In graph")
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=prob_color(e["prob"]), width=5 if focal_e else 2,
                      dash="dash" if new_e else "solid")))
        mid_x.append((x0 + x1) / 2); mid_y.append((y0 + y1) / 2)
        known = "" if e["label"] is None else f"<br>Known label: {'fraud' if e['label'] else 'genuine'}"
        mid_text.append(f"{'This transaction' if focal_e else 'Existing edge'}<br>"
                        f"Model P(fraud) = {e['prob']:.3f}{known}")
    fig.add_trace(go.Scatter(x=mid_x, y=mid_y, mode="markers", hovertext=mid_text, hoverinfo="text",
                             marker=dict(size=14, color="rgba(0,0,0,0)"), showlegend=False))

    for kind, style in NODE_STYLE.items():
        keys = [k for k in pos if kind_of(k) == kind]
        if not keys:
            continue
        is_focal = [k in (cust, merch, tx_focal) for k in keys]
        fig.add_trace(go.Scatter(
            x=[pos[k][0] for k in keys], y=[pos[k][1] for k in keys], mode="markers+text",
            name=style["name"], hovertext=[describe(k) for k in keys], hoverinfo="text",
            text=[("<b>" + ("new " if k.startswith("new-") else "") + style["name"].lower() + "</b>") if f else ""
                  for k, f in zip(keys, is_focal)],
            textposition="top center",
            marker=dict(symbol=style["symbol"], color=[("#ffffff" if k.startswith("new-") else style["color"]) for k in keys],
                        size=[24 if f else 11 for f in is_focal],
                        line=dict(color=style["color"], width=[3 if f else 1 for f in is_focal]))))

    # colorbar for edge probability
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=PROB_SCALE, cmin=0, cmax=1, color=[0],
                                         colorbar=dict(title="Edge P(fraud)", thickness=10, len=0.8))))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False, range=[-2.0, 2.0]), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.02), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def metrics_frame(rows, extra_cols=()):
    df = pd.DataFrame(rows)
    df["graph"] = df["graph"].str.title()
    df["operator"] = df["operator"].map(OP_LABEL)
    cols = ["graph", "operator", "precision", "recall", "f1", "roc_auc", "pr_auc", *extra_cols]
    df = df[[c for c in cols if c in df.columns]]
    return df.rename(columns={"graph": "Graph", "operator": "Operator", "precision": "Precision",
                              "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
                              "cv_pr_auc": "CV PR-AUC", "best_params": "best_params_"})
