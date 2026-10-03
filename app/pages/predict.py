import io

import numpy as np
import pandas as pd
import streamlit as st

from app.engine import (GRAPHS, IN_GRAPH, KNOWN, APPROX, LOW, UNAVAILABLE, MAX_ROWS, OP_LABEL,
                        OPERATORS, TRACK_LABEL, ARTIFACTS_DIR, InputError, validate_frame)
from app.ui import get_engine, label_at, metrics_frame, neighborhood_figure, pr_table, prob_heatmap

engine = get_engine()
enhanced_ready = engine.has_track("enhanced")

CONF_BADGE = {IN_GRAPH: "🟢 In graph", KNOWN: "🟢 Known entities", APPROX: "🟡 Approximated",
              LOW: "🟠 Low confidence", UNAVAILABLE: "⚪ Unavailable"}

# ------------------------------------------------------------------ sidebar: model + threshold
with st.sidebar:
    st.subheader("Model")
    views = ["Thesis Model", "Enhanced Model (tuned)", "Side by side"]
    view = st.radio("Classifier", views if enhanced_ready else views[:1], label_visibility="collapsed",
                    help="Both use the same embeddings; only the Random Forest differs.")
    if not enhanced_ready:
        st.caption("Enhanced Model artifacts not found yet. Run `python -m pipeline.phase1b_enhanced`.")
    tracks = {"Thesis Model": ["thesis"], "Enhanced Model (tuned)": ["enhanced"],
              "Side by side": ["thesis", "enhanced"]}[view]

    st.subheader("Decision threshold")
    threshold = st.slider("Flag as fraud when P(fraud) ≥", 0.0, 1.0, 0.5, 0.01,
                          help="Relabels the already-computed probabilities. The thesis used 0.5.")
    st.caption("Precision/recall at this threshold come from each model's precomputed test-set "
               "precision-recall curve; nothing is retrained.")

st.title("Credit card fraud: graph-based scoring")
st.caption("Each transaction is scored by 8 Random Forests (2 graph types × 4 edge operators) on top of "
           "Node2Vec embeddings. See **Model Card** for measured performance before trusting any score.")


# ------------------------------------------------------------------ input
def load(df, source):
    try:
        clean, problems = validate_frame(df)
    except InputError as e:
        st.session_state.pop("batch", None)
        st.error(f"Could not use this input. {e}")
        return
    if clean.empty:
        st.session_state.pop("batch", None)
        st.error("No valid rows to score.\n\n" + "\n".join(f"- {p}" for p in problems[:10]))
        return
    st.session_state.update(batch=clean, scores=engine.score_frame(clean, tracks), problems=problems,
                            source=source, selected=0)


c1, c2 = st.columns([1, 3], vertical_alignment="center")
if c1.button("▶ Try with sample data", type="primary", use_container_width=True):
    load(pd.read_csv(ARTIFACTS_DIR / "sample_transactions.csv", dtype={"cc_num": str}), "sample")
c2.caption("10 real transactions from `fraudTest.csv`, which the models never saw "
           "(including 2 from customers absent from the training graph).")

tab_one, tab_csv = st.tabs(["Single transaction", "Upload CSV"])
with tab_one:
    with st.form("single"):
        a, b, c = st.columns([2, 3, 1.3])
        cc = a.text_input("Card number (cc_num)", value=engine.customers[0],
                          help="Any card number; known ones have embeddings.")
        shown = [m.removeprefix("fraud_") for m in engine.merchants]
        m_pick = b.selectbox("Merchant", shown, index=0)
        amt = c.number_input("Amount ($)", min_value=0.0, value=50.0, step=1.0)
        m_new = b.text_input("…or type a merchant name not in the list", "")
        with st.expander("Optional: trans_num"):
            tn = st.text_input("trans_num", "", help="If this exact transaction is in fraudTrain.csv, "
                               "the tripartite model can use its real transaction node.")
        st.caption("The amount is recorded but is not a classifier input: it only weighted the random "
                   "walks when the embeddings were trained.")
        if st.form_submit_button("Score transaction"):
            row = {"cc_num": cc, "merchant": m_new.strip() or m_pick, "amt": amt}
            if tn.strip():
                row["trans_num"] = tn.strip()
            load(pd.DataFrame([row]), "manual")

with tab_csv:
    up = st.file_uploader(f"CSV with at least cc_num, merchant, amt (max {MAX_ROWS} rows)", type="csv")
    st.caption("Same schema as Kaggle's fraudTrain.csv; extra columns are ignored. If `trans_num` or "
               "`is_fraud` are present they are used (exact node lookup / showing the actual label).")
    if up is not None and st.button("Score file"):
        try:
            df = pd.read_csv(io.BytesIO(up.getvalue()), dtype={"cc_num": str, "trans_num": str})
        except Exception as e:  # noqa: BLE001 - any parse failure is a user-facing input error
            st.session_state.pop("batch", None)
            st.error(f"Could not read this file as CSV ({type(e).__name__}: {e}).")
        else:
            load(df, "upload")

if "batch" not in st.session_state:
    st.info("Enter a transaction, upload a CSV, or try the sample data.")
    st.stop()

batch = st.session_state.batch
# only the model(s) on screen are scored; the other track is scored the first time it is shown
scores = engine.add_tracks(batch, st.session_state.scores, tracks)
if st.session_state.problems:
    with st.expander(f"⚠ {len(st.session_state.problems)} row(s) skipped"):
        st.write("\n".join(f"- {p}" for p in st.session_state.problems))

st.divider()
st.caption("ℹ Node2Vec only has embeddings for customers and merchants present in the training graph, so "
           "transactions involving unseen entities are approximated and flagged, or reported as unavailable, "
           "rather than guessed.")


# ------------------------------------------------------------------ results table
def results_frame(track_list):
    out = batch[["cc_num", "merchant", "amt"]].copy()
    out["merchant"] = out["merchant"].str.removeprefix("fraud_")
    if "is_fraud" in batch.columns:
        out["actual"] = batch["is_fraud"].map({1: "fraud", 0: "genuine", "1": "fraud", "0": "genuine"})
    for g in GRAPHS:
        out[f"{g} confidence"] = [CONF_BADGE[s.confidence] for s in scores[g]]
    for t in track_list:
        short = "" if len(track_list) == 1 else ("T·" if t == "thesis" else "E·")
        for g in GRAPHS:
            for op in OPERATORS:
                out[f"{short}{g[:3]}·{OP_LABEL[op]}"] = [s.probs[t][op] for s in scores[g]]
    flags = []
    for i in range(len(batch)):
        ps = [scores[g][i].probs[t][op] for t in track_list for g in GRAPHS for op in OPERATORS]
        ps = [p for p in ps if not np.isnan(p)]
        flags.append(f"{sum(p >= threshold for p in ps)}/{len(ps)}" if ps else "n/a")
    out["flagged"] = flags
    return out


def download_frame():
    """Every scored track, with labels at the current threshold."""
    out = batch.copy()
    for g in GRAPHS:
        out[f"{g}_confidence"] = [s.confidence for s in scores[g]]
        out[f"{g}_note"] = [s.note for s in scores[g]]
        out[f"{g}_in_training_set"] = [s.in_sample for s in scores[g]]
        for t in scores[g][0].probs:
            for op in OPERATORS:
                p = [s.probs[t][op] for s in scores[g]]
                out[f"{t}_{g}_{op}_prob"] = np.round(p, 4)
                out[f"{t}_{g}_{op}_label"] = [label_at(x, threshold).replace("⚑ ", "") for x in p]
    out["threshold"] = threshold
    return out


if len(batch) > 1:
    st.subheader(f"Results · {len(batch)} transactions")
    tbl = results_frame(tracks)
    prob_cols = [c for c in tbl.columns if "·" in c]
    event = st.dataframe(
        tbl, hide_index=False, use_container_width=True, on_select="rerun", selection_mode="single-row",
        column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.2f")
                       for c in prob_cols} | {
            "amt": st.column_config.NumberColumn("amt", format="$%.2f"),
            "flagged": st.column_config.TextColumn("flagged", help=f"Classifiers with P(fraud) ≥ {threshold:.2f}")})
    st.download_button("Download results as CSV", download_frame().to_csv(index=False).encode(),
                       "fraud_scores.csv", "text/csv")
    sel = event.selection.rows
    if sel:
        st.session_state.selected = sel[0]
    i = st.session_state.get("selected", 0)
    st.caption("Select a row to see its details below.")
else:
    i = 0
    st.download_button("Download result as CSV", download_frame().to_csv(index=False).encode(),
                       "fraud_scores.csv", "text/csv")

# ------------------------------------------------------------------ detail for one transaction
r = batch.iloc[i]
one = {g: scores[g][i] for g in GRAPHS}
st.subheader(f"Transaction {i + 1}: card …{str(r.cc_num)[-4:]} → {r.merchant.removeprefix('fraud_')} · ${r.amt:,.2f}")
if "is_fraud" in batch.columns:
    actual = "fraud" if str(r.is_fraud) == "1" else "genuine"
    st.markdown(f"Actual label (held out, not shown to the model): **{actual}**")

cols = st.columns(2)
for col, g in zip(cols, GRAPHS):
    col.markdown(f"**{g.title()} graph:** {CONF_BADGE[one[g].confidence]}")
    col.caption(one[g].note + (" Its score is in-sample, so likely optimistic." if one[g].in_sample else ""))

heat_cols = st.columns(len(tracks))
for col, t in zip(heat_cols, tracks):
    col.plotly_chart(prob_heatmap(engine, one, t, threshold, TRACK_LABEL[t]), use_container_width=True,
                     config={"displayModeBar": False})

st.markdown(f"**Test-set precision / recall at threshold {threshold:.2f}**")
st.dataframe(pr_table(engine, tracks, threshold), hide_index=True, use_container_width=True,
             column_config={c: st.column_config.NumberColumn(format="%.3f")
                            for c in pr_table(engine, tracks, 0.5).columns if "precision" in c or "recall" in c})

# ------------------------------------------------------------------ local subgraph
st.markdown("**Local subgraph**")
v1, v2, v3 = st.columns(3)
vg = v1.radio("Graph", GRAPHS, format_func=str.title, horizontal=True)
vo = v2.radio("Operator", OPERATORS, format_func=OP_LABEL.get, horizontal=True)
vt = v3.radio("Colored by", tracks, format_func=TRACK_LABEL.get, horizontal=True) if len(tracks) > 1 else tracks[0]
if one[vg].confidence == UNAVAILABLE:
    st.info("Neither entity is in the graph, so there is no neighborhood to show.")
else:
    st.plotly_chart(neighborhood_figure(engine, vg, one[vg], vt, vo, r.cc_num, r.merchant),
                    use_container_width=True, config={"displayModeBar": False})
    st.caption("This transaction's edge(s) are thick (dashed when new) and a sample of up to 10 existing "
               "edges around the customer and the merchant are thin; every edge is colored by the model's "
               "P(fraud). Hover an edge for its known label.")

# ------------------------------------------------------------------ honest baseline next to the prediction
st.markdown("**Measured test-set performance of the model(s) above** (threshold 0.5)")
pcols = st.columns(len(tracks))
for col, t in zip(pcols, tracks):
    rows = engine.manifest[t]["rerun"]
    extra = ("cv_pr_auc", "best_params") if t == "enhanced" else ()
    df = metrics_frame(rows, extra)
    if "best_params_" in df.columns:
        df["best_params_"] = df["best_params_"].map(lambda d: ", ".join(f"{k}={v}" for k, v in d.items()))
    col.caption(TRACK_LABEL[t])
    col.dataframe(df, hide_index=True, use_container_width=True)
