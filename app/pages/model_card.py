import pandas as pd
import streamlit as st

from app.engine import GRAPHS, OP_LABEL
from app.ui import get_engine, metrics_frame

engine = get_engine()
m = engine.manifest

st.title("Model Card")
st.caption("Measured test-set performance, before trusting any prediction. Positive class = fraud; "
           "threshold 0.5; test sets keep the natural fraud rate (~3%).")

ds, graphs = m.get("dataset", {}), m.get("graphs", {})
a, b, c, d = st.columns(4)
a.metric("Training transactions (Option B)", f"{ds.get('sample_rows', 0):,}")
b.metric("Fraud rate", f"{100 * ds.get('sample_fraud_rate', 0):.2f}%")
c.metric("Bipartite nodes / edges", f"{graphs.get('bipartite', {}).get('nodes', 0):,} / "
                                    f"{graphs.get('bipartite', {}).get('edges', 0):,}")
d.metric("Tripartite nodes / edges", f"{graphs.get('tripartite', {}).get('nodes', 0):,} / "
                                     f"{graphs.get('tripartite', {}).get('edges', 0):,}")

st.subheader("Thesis Model")
st.markdown("Random Forest, 100 trees, `random_state=42`, trained on a 1:1 undersampled training set; "
            "evaluated on the untouched 30% test split.")
t1, t2 = st.tabs(["Published in the thesis (Table 4.1)", "This deployment's re-run"])
with t1:
    st.dataframe(metrics_frame(m["thesis"]["published_table_4_1"], ("TP", "FP", "FN", "TN")),
                 hide_index=True, use_container_width=True)
    base = m["thesis"]["published_baseline_pr_auc"]
    st.caption(f"PR-AUC of a random classifier (= test fraud rate): bipartite {base['bipartite']}, "
               f"tripartite {base['tripartite']}.")
with t2:
    st.dataframe(metrics_frame(m["thesis"]["rerun"], ("TP", "FP", "FN", "TN")),
                 hide_index=True, use_container_width=True)
    st.caption("The live predictions come from these re-run models. The same code was run again locally "
               "(Node2Vec walks are unseeded, so embeddings and metrics differ slightly from the thesis), "
               "because the original run did not save the node-to-entity mapping the website needs.")

st.subheader("Enhanced Model (tuned)")
if "enhanced" not in m:
    st.info("Not trained yet. Run `python -m pipeline.phase1b_enhanced`.")
else:
    e = m["enhanced"]
    st.markdown("Same embeddings, same split, same 1:1 training set as the Thesis Model; only the Random "
                "Forest hyperparameters were tuned with `GridSearchCV` (5-fold stratified CV, "
                "`scoring='average_precision'`). The best estimator was evaluated once on the same test set.")
    df = metrics_frame(e["rerun"], ("cv_pr_auc", "best_params"))
    df["best_params_"] = df["best_params_"].map(lambda p: ", ".join(f"{k}={v}" for k, v in p.items()))
    st.dataframe(df, hide_index=True, use_container_width=True)

    thesis = {(r["graph"], r["operator"]): r for r in m["thesis"]["rerun"]}
    cmp = pd.DataFrame([{
        "Graph": r["graph"].title(), "Operator": OP_LABEL[r["operator"]],
        "PR-AUC thesis": thesis[(r["graph"], r["operator"])]["pr_auc"], "PR-AUC enhanced": r["pr_auc"],
        "Δ PR-AUC": round(r["pr_auc"] - thesis[(r["graph"], r["operator"])]["pr_auc"], 3),
        "Recall thesis": thesis[(r["graph"], r["operator"])]["recall"], "Recall enhanced": r["recall"],
    } for r in e["rerun"]])
    st.markdown("**Thesis vs Enhanced on the same test set**")
    st.dataframe(cmp, hide_index=True, use_container_width=True)
    with st.expander("Search space"):
        st.json(e["param_grid"])

st.subheader("Limitations")
st.markdown(
    "- **Precision is low (about 0.03–0.09 at threshold 0.5)**: most transactions the models flag are genuine. "
    "Use as a triage signal for human review, not an automatic decision.\n"
    "- **Transductive embeddings**: only customers and merchants in the training graph have embeddings. "
    "New transactions in the tripartite graph are approximated from related transaction nodes.\n"
    "- **Embeddings saw the whole graph**: Node2Vec was trained on all edges before the train/test split "
    "(labels were never used), as in the thesis.\n"
    "- **Amount is not a feature**: it only weighted the random walks.\n"
    "- **Synthetic data**: the Kaggle dataset is simulated (Sparkov), so these numbers say nothing about "
    "real card networks.")

with st.expander("Other thesis results: K-Means (Table 4.3) and transaction-grouped split (Table 4.4)"):
    st.markdown("**K-Means, k = 2 (published)**")
    km = pd.DataFrame(m["thesis"]["published_kmeans_table_4_3"])
    km["graph"] = km["graph"].str.title(); km["operator"] = km["operator"].map(OP_LABEL)
    st.dataframe(km, hide_index=True, use_container_width=True)
    st.markdown("**Random Forest, tripartite, transaction-grouped split (published)**")
    gr = pd.DataFrame(m["thesis"]["published_grouped_table_4_4"])
    gr["operator"] = gr["operator"].map(OP_LABEL)
    st.dataframe(gr, hide_index=True, use_container_width=True)
