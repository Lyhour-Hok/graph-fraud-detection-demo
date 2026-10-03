"""Phase 1 offline pipeline: the "Thesis Model".

Sections A.1-A.7 of the thesis appendix ("Full Source Code"), reused as-is.
The only changes are the ones needed to run outside Google Colab:
  * no drive.mount(); OUT_DIR is the local artifacts/ folder
  * fraudTrain.csv is read from a local path instead of being fetched by kagglehub
  * the code is split into stages so that the bipartite and tripartite embeddings
    run in separate processes, as they ran in separate Colab sessions
Everything added for the live demo is marked "# [demo]".

Usage, from the project root (each stage is a separate process):
  python -m pipeline.phase1_thesis embed --graph bipartite  --data <fraudTrain.csv>
  python -m pipeline.phase1_thesis embed --graph tripartite --data <fraudTrain.csv>
  python -m pipeline.phase1_thesis classify
  python -m pipeline.phase1_thesis extras
  python -m pipeline.phase1_thesis samples --test-data <fraudTest.csv or .zip>
"""
import argparse
import os, gc, time
import numpy as np, pandas as pd, networkx as nx
import psutil
import joblib                                                     # [demo]

from pipeline.common import (ARTIFACTS_DIR, BUILD_DIR, GRAPHS, check,   # [demo]
                             default_train_csv, update_manifest)
from pipeline import thesis_tables                                # [demo]

# ---------------------------------------------------------------- A.1 Setup and Parameters
OUT_DIR = str(ARTIFACTS_DIR)        # was '/content/drive/MyDrive/thesis_fraud_optionB'
os.makedirs(OUT_DIR, exist_ok=True)

NUM_WALKS  = 10     # walks per node (Section 1.2.4)
SAMPLE_GEN = 0.20   # Option B: keep 20% of genuine transactions

def ram():
    return f"{psutil.virtual_memory().percent}%"


# ---------------------------------------------------------------- A.2 Data Sampling (Option B)
def load_option_b(data_path):
    cache = BUILD_DIR / "df_b.parquet"                            # [demo] deterministic, so cache it
    if cache.exists():                                            # [demo]
        df_b = pd.read_parquet(cache)                             # [demo]
        print("Option B (cached):", df_b.shape, df_b.is_fraud.value_counts().to_dict())
        return df_b

    df = pd.read_csv(data_path)                                   # was: kagglehub.dataset_download(...)

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    full_rows, full_fraud = len(df), int(df.is_fraud.sum())       # [demo]
    fraud   = df[df.is_fraud == 1]
    genuine = df[df.is_fraud == 0].sample(frac=SAMPLE_GEN, random_state=42)
    df_b = pd.concat([fraud, genuine]).reset_index(drop=True)

    print("Option B:", df_b.shape, df_b.is_fraud.value_counts().to_dict())
    del df, fraud, genuine
    gc.collect()

    df_b.to_parquet(cache)                                        # [demo]
    update_manifest(["dataset"], {                                # [demo]
        "source": "Kaggle kartik2112/fraud-detection, fraudTrain.csv",
        "full_rows": full_rows, "full_fraud": full_fraud,
        "sampling": "Option B: all fraud + 20% of genuine (random_state=42)",
        "sample_rows": len(df_b), "sample_fraud": int(df_b.is_fraud.sum()),
        "sample_fraud_rate": round(float(df_b.is_fraud.mean()), 4),
    })
    return df_b


# ---------------------------------------------------------------- A.3 Graph Construction
def build_graph_bipartite(df_input, graph_type=nx.Graph):
    df = df_input.copy()
    mapping = {x: i for i, x in enumerate(
        set(df["cc_num"].values.tolist() + df["merchant"].values.tolist()))}
    df["from"] = df["cc_num"].map(mapping)
    df["to"]   = df["merchant"].map(mapping)
    df = df[["from", "to", "amt", "is_fraud"]].groupby(
        ["from", "to"]).agg({"is_fraud": "sum", "amt": "sum"}).reset_index()
    df["is_fraud"] = (df["is_fraud"] > 0).astype(int)
    return nx.from_edgelist(
        [(r["from"], r["to"], {"weight": r["amt"], "is_fraud": r["is_fraud"]})
         for _, r in df.iterrows()],
        create_using=graph_type())

def build_graph_tripartite(df_input, graph_type=nx.Graph):
    df = df_input.copy()
    df["idx"] = df.index
    mapping = {x: i for i, x in enumerate(
        set(df["cc_num"].values.tolist() + df["merchant"].values.tolist()
            + df["idx"].values.tolist()))}
    df["from"] = df["cc_num"].map(mapping)
    df["to"]   = df["merchant"].map(mapping)
    df["transaction"] = df["idx"].map(mapping)
    G = graph_type()
    for _, r in df.iterrows():
        G.add_edge(r["from"], r["transaction"],
                   weight=r["amt"] / 2, is_fraud=r["is_fraud"])
        G.add_edge(r["transaction"], r["to"],
                   weight=r["amt"] / 2, is_fraud=r["is_fraud"])
    return G


# ---------------------------------------------------------------- A.4 Node Embedding (Node2Vec)
from node2vec import Node2Vec

def embed_and_save(G, name):
    t0 = time.time()
    n2v = Node2Vec(G, dimensions=64, walk_length=10,
                    num_walks=NUM_WALKS, workers=1)
    model = n2v.fit(window=10, min_count=1, batch_words=10000, workers=2)
    model.wv.save(f"{OUT_DIR}/{name}_wv.kv")
    u, v, y = zip(*[(a, b, int(d["is_fraud"]))
                     for a, b, d in G.edges(data=True)])
    np.savez_compressed(f"{OUT_DIR}/{name}_edges.npz",
                         u=np.array(u), v=np.array(v), y=np.array(y))
    print(f"[{name}] nodes={G.number_of_nodes():,} "
          f"edges={G.number_of_edges():,} "
          f"fraud_edges={sum(y):,} ({100*sum(y)/len(y):.2f}%) "
          f"time={(time.time()-t0)/60:.1f} min RAM={ram()}")


# ---------------------------------------------------------------- [demo] node lookup tables
# The thesis code builds node ids with enumerate(set(...)), whose order changes between
# Python processes, and does not save the mapping. Re-evaluating the same expression on the
# same data in the same process gives the same ids; the asserts below prove it against G.
# Keys are stored as the Node2Vec/KeyedVectors key strings, i.e. str(node object in G).

def entity_keys(df_b, G, with_transactions):
    items = df_b["cc_num"].values.tolist() + df_b["merchant"].values.tolist()
    if with_transactions:
        items += df_b.index.values.tolist()                       # df["idx"] = df.index
    mapping = {x: i for i, x in enumerate(set(items))}
    node_obj = {n: n for n in G.nodes}                            # bipartite ids are floats (iterrows upcast)
    keys = {x: str(node_obj[i]) for x, i in mapping.items()}
    if with_transactions:
        for cc, m, t in zip(df_b["cc_num"], df_b["merchant"], df_b.index):
            assert G.has_edge(mapping[cc], mapping[t]) and G.has_edge(mapping[t], mapping[m]), \
                "node mapping does not match the tripartite graph"
        assert G.number_of_edges() == 2 * len(df_b)
    else:
        pairs = df_b[["cc_num", "merchant"]].drop_duplicates()
        for cc, m in pairs.itertuples(index=False):
            assert G.has_edge(mapping[cc], mapping[m]), "node mapping does not match the bipartite graph"
        assert G.number_of_edges() == len(pairs)
    print("  node mapping verified against the graph")
    return keys


def save_lookup_tables(name, df_b, keys):
    d = np.load(f"{OUT_DIR}/{name}_edges.npz")
    su, sv = d["u"].astype(str), d["v"].astype(str)
    edge_of = {}
    for i, (a, b) in enumerate(zip(su, sv)):
        edge_of[(a, b)] = i
        edge_of[(b, a)] = i

    customers = pd.Series(df_b["cc_num"].unique())
    merchants = pd.Series(df_b["merchant"].unique())
    nodes = pd.concat([
        pd.DataFrame({"key": customers.map(keys), "kind": "customer", "name": customers.astype(str)}),
        pd.DataFrame({"key": merchants.map(keys), "kind": "merchant", "name": merchants}),
    ], ignore_index=True)
    nodes.to_parquet(f"{OUT_DIR}/{name}_nodes.parquet", index=False)

    if name == "bipartite":
        pairs = (df_b.groupby(["cc_num", "merchant"])
                 .agg(n_tx=("amt", "size"), amt_sum=("amt", "sum"), n_fraud=("is_fraud", "sum"))
                 .reset_index())
        pairs["cust_key"] = pairs["cc_num"].map(keys)
        pairs["merch_key"] = pairs["merchant"].map(keys)
        pairs["edge"] = [edge_of[(c, m)] for c, m in zip(pairs["cust_key"], pairs["merch_key"])]
        pairs.to_parquet(f"{OUT_DIR}/bipartite_pairs.parquet", index=False)
        print(f"  saved {len(nodes):,} entity nodes and {len(pairs):,} customer-merchant pairs")
    else:
        tx = pd.DataFrame({
            "tx_key": pd.Series(df_b.index).map(keys).values,
            "cust_key": df_b["cc_num"].map(keys).values,
            "merch_key": df_b["merchant"].map(keys).values,
            "cc_num": df_b["cc_num"].values,
            "merchant": df_b["merchant"].values,
            "trans_num": df_b["trans_num"].values,
            "amt": df_b["amt"].values,
            "is_fraud": df_b["is_fraud"].values,
        })
        tx["edge_ct"] = [edge_of[(c, t)] for c, t in zip(tx["cust_key"], tx["tx_key"])]
        tx["edge_tm"] = [edge_of[(t, m)] for t, m in zip(tx["tx_key"], tx["merch_key"])]
        tx.to_parquet(f"{OUT_DIR}/tripartite_transactions.parquet", index=False)
        print(f"  saved {len(nodes):,} entity nodes and {len(tx):,} transaction nodes")


def stage_embed(graph, data_path):
    df_b = load_option_b(data_path)
    if graph == "bipartite":
        print("Building Bipartite...")
        G = build_graph_bipartite(df_b)
    else:
        print("Building Tripartite...")
        G = build_graph_tripartite(df_b)
    print(f"Nodes={G.number_of_nodes():,} Edges={G.number_of_edges():,} RAM={ram()}")
    keys = entity_keys(df_b, G, with_transactions=(graph == "tripartite"))    # [demo]

    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()                # [demo]
    n_fraud = sum(int(d["is_fraud"]) for _, _, d in G.edges(data=True))       # [demo]
    embed_and_save(G, graph)
    del G; gc.collect()

    save_lookup_tables(graph, df_b, keys)                                      # [demo]
    exp = thesis_tables.GRAPH_STATS[graph]
    print(f"Check against thesis Table 3.7 ({graph}):")
    check("nodes", n_nodes, exp["nodes"])
    check("edges", n_edges, exp["edges"])
    check("fraud edges", n_fraud, exp["fraud_edges"])
    update_manifest(["graphs", graph], {"nodes": n_nodes, "edges": n_edges, "fraud_edges": n_fraud,
                                        "embedded": time.strftime("%Y-%m-%d %H:%M")})


# ---------------------------------------------------------------- A.5 Supervised Classification (Random Forest)
from gensim.models import KeyedVectors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score)
from sklearn.metrics import precision_recall_curve                # [demo]

def edge_features(A, B, method):
    if method == "hadamard": return A * B
    if method == "average":  return (A + B) / 2
    if method == "l1":       return np.abs(A - B)
    if method == "l2":       return (A - B) ** 2


def stage_classify():
    rows = []
    for name in ["bipartite", "tripartite"]:
        kv = KeyedVectors.load(f"{OUT_DIR}/{name}_wv.kv")
        d = np.load(f"{OUT_DIR}/{name}_edges.npz")
        key2idx = {k: i for i, k in enumerate(kv.index_to_key)}
        iu = np.array([key2idx[str(x)] for x in d["u"]])
        iv = np.array([key2idx[str(x)] for x in d["v"]])
        A, B, y = kv.vectors[iu], kv.vectors[iv], d["y"]

        # [demo] the split depends only on len(y), y and random_state, so it is identical for
        # all four operators; record which edges were trained on (0=test, 1=trained, 2=train but
        # dropped by undersampling) so the website can flag in-sample lookups.
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)
        split = None

        for method in ["hadamard", "average", "l1", "l2"]:
            t0 = time.time()                                        # [demo]
            X = edge_features(A, B, method)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)

            # Undersample the TRAINING set only, to a 1:1 ratio
            rng = np.random.RandomState(42)
            pos = np.where(ytr == 1)[0]
            neg = rng.choice(np.where(ytr == 0)[0],
                              size=len(pos), replace=False)
            sel = np.concatenate([pos, neg])

            clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                          n_jobs=-1).fit(Xtr[sel], ytr[sel])
            pred = clf.predict(Xte)
            prob = clf.predict_proba(Xte)[:, 1]

            tn, fp, fn, tp = confusion_matrix(
                yte, pred, labels=[0, 1]).ravel()
            p, r, f1, _ = precision_recall_fscore_support(
                yte, pred, pos_label=1, average="binary", zero_division=0)

            rows.append({"graph": name, "operator": method,
                         "TP": int(tp), "FP": int(fp),
                         "FN": int(fn), "TN": int(tn),
                         "precision": round(p, 3), "recall": round(r, 3),
                         "f1": round(f1, 3),
                         "roc_auc": round(roc_auc_score(yte, prob), 3),
                         "pr_auc": round(average_precision_score(yte, prob), 3),
                         "baseline_pr_auc": round(yte.mean(), 3)})

            # [demo] save the classifier, its test-set PR curve and the split
            if split is None:
                assert np.array_equal(Xtr, X[itr]) and np.array_equal(Xte, X[ite])
                split = np.zeros(len(y), dtype=np.int8)
                split[itr] = 2
                split[itr[sel]] = 1
                np.save(f"{OUT_DIR}/{name}_edge_split.npy", split)
                update_manifest(["graphs", name, "split"], {
                    "train": len(itr), "train_fraud": int(ytr.sum()), "train_used_1to1": len(sel),
                    "test": len(ite), "test_fraud": int(yte.sum())})
            joblib.dump(clf, f"{OUT_DIR}/{name}_rf_{method}_thesis.joblib", compress=3)
            prec, rec, thr = precision_recall_curve(yte, prob)
            np.savez_compressed(f"{OUT_DIR}/{name}_rf_{method}_thesis_pr.npz",
                                precision=prec, recall=rec, thresholds=thr)
            print(f"done {name}/{method}: R={r:.3f} P={p:.3f} PR-AUC={rows[-1]['pr_auc']:.3f} "
                  f"time={time.time()-t0:.0f}s")
            del clf, X, Xtr, Xte; gc.collect()

    res = pd.DataFrame(rows)
    res.to_csv(f"{BUILD_DIR}/final_results_with_confmat.csv", index=False)
    print(res.to_string(index=False))

    update_manifest(["thesis"], {                                    # [demo]
        "label": "Thesis Model",
        "classifier": "RandomForestClassifier(n_estimators=100, random_state=42), "
                      "trained on the 1:1 undersampled training set",
        "published_table_4_1": thesis_tables.RF_RESULTS,
        "published_baseline_pr_auc": thesis_tables.PR_AUC_BASELINE,
        "published_kmeans_table_4_3": thesis_tables.KMEANS_RESULTS,
        "published_grouped_table_4_4": thesis_tables.GROUPED_RESULTS,
        "rerun": rows,
        "trained": time.strftime("%Y-%m-%d %H:%M"),
    })


# ---------------------------------------------------------------- A.6 Unsupervised Clustering (K-Means)
# ---------------------------------------------------------------- A.7 Robustness Check (Transaction-Grouped Split)
def stage_extras():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    K = 2
    rows = []
    for name in ["bipartite", "tripartite"]:
        kv = KeyedVectors.load(f"{OUT_DIR}/{name}_wv.kv")
        d = np.load(f"{OUT_DIR}/{name}_edges.npz")
        key2idx = {k: i for i, k in enumerate(kv.index_to_key)}
        iu = np.array([key2idx[str(x)] for x in d["u"]])
        iv = np.array([key2idx[str(x)] for x in d["v"]])
        A, B, y = kv.vectors[iu], kv.vectors[iv], d["y"]
        base = y.mean()

        for method in ["hadamard", "average", "l1", "l2"]:
            X = edge_features(A, B, method)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)

            km = KMeans(n_clusters=K, n_init=10,
                         random_state=42).fit(Xtr)
            fr = [ytr[km.labels_ == c].mean() for c in range(K)]
            fraud_cluster = int(np.argmax(fr))

            clu = km.predict(Xte)
            pred = (clu == fraud_cluster).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                yte, pred, pos_label=1, average="binary", zero_division=0)

            rows.append({"graph": name, "operator": method,
                         "precision": p, "recall": r, "f1": f1,
                         "lift": p / base,
                         "ARI": adjusted_rand_score(yte, clu),
                         "flagged_pct": 100 * pred.mean()})
            print(f"done k-means {name}/{method}")

    res = pd.DataFrame(rows).round(3)
    print(res.to_string(index=False))
    update_manifest(["thesis", "rerun_kmeans"], res.to_dict("records"))      # [demo]

    from sklearn.model_selection import GroupShuffleSplit

    kv = KeyedVectors.load(f"{OUT_DIR}/tripartite_wv.kv")
    d  = np.load(f"{OUT_DIR}/tripartite_edges.npz")
    u, v, y = d["u"], d["v"], d["y"]

    # Identify the transaction node of each edge (degree exactly 2)
    nodes, counts = np.unique(np.concatenate([u, v]), return_counts=True)
    deg_u = counts[np.searchsorted(nodes, u)]
    tx = np.where(deg_u == 2, u, v)

    key2idx = {k: i for i, k in enumerate(kv.index_to_key)}
    A = kv.vectors[np.array([key2idx[str(x)] for x in u])]
    B = kv.vectors[np.array([key2idx[str(x)] for x in v])]

    rows = []
    for method in ["hadamard", "average", "l1", "l2"]:
        X = edge_features(A, B, method)
        tr, te = next(GroupShuffleSplit(
            n_splits=1, test_size=0.3, random_state=42).split(X, y, groups=tx))

        rng = np.random.RandomState(42)
        pos = tr[y[tr] == 1]
        neg = rng.choice(tr[y[tr] == 0], size=len(pos), replace=False)
        sel = np.concatenate([pos, neg])

        clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                      n_jobs=-1).fit(X[sel], y[sel])
        prob = clf.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        p, r, f1, _ = precision_recall_fscore_support(
            y[te], pred, pos_label=1, average="binary", zero_division=0)

        rows.append({"operator": method, "TP": int(tp), "FP": int(fp),
                     "FN": int(fn), "TN": int(tn),
                     "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(f1, 3),
                     "roc_auc": round(roc_auc_score(y[te], prob), 3),
                     "pr_auc": round(average_precision_score(y[te], prob), 3)})
        print(f"done grouped {method}")

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    update_manifest(["thesis", "rerun_grouped"], rows)                         # [demo]


# ---------------------------------------------------------------- [demo] held-out sample transactions
def stage_samples(test_path, n_per_class=4):
    """Pick real transactions from fraudTest.csv (never part of the training graph)."""
    te = pd.read_csv(test_path)
    if "Unnamed: 0" in te.columns:
        te = te.drop("Unnamed: 0", axis=1)
    known = set(pd.read_parquet(f"{OUT_DIR}/tripartite_nodes.parquet").query("kind == 'customer'")["name"])
    te["known_customer"] = te["cc_num"].astype(str).isin(known)

    rng = 42
    parts = [
        te[te.known_customer & (te.is_fraud == 1)].sample(n_per_class, random_state=rng),
        te[te.known_customer & (te.is_fraud == 0)].sample(n_per_class, random_state=rng),
        # every fraudTest transaction from a customer unseen in training is fraud (163 rows, 16 cards)
        te[~te.known_customer].drop_duplicates("cc_num").sample(2, random_state=rng),
    ]
    cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "trans_num", "is_fraud"]
    sample = pd.concat(parts)[cols].sample(frac=1, random_state=rng).reset_index(drop=True)
    sample.to_csv(f"{OUT_DIR}/sample_transactions.csv", index=False)
    print(sample.to_string(index=False))
    update_manifest(["samples"], {
        "source": "fraudTest.csv (held out: no trans_num overlaps fraudTrain.csv)",
        "rows": len(sample), "fraud": int(sample.is_fraud.sum()),
        "unseen_customers": int((~sample.cc_num.astype(str).isin(known)).sum())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["embed", "classify", "extras", "samples"])
    ap.add_argument("--graph", choices=GRAPHS)
    ap.add_argument("--data", default=default_train_csv(), help="path to fraudTrain.csv")
    ap.add_argument("--test-data", help="path to fraudTest.csv (or .zip)")
    args = ap.parse_args()

    if args.stage == "embed":
        if not args.graph:
            ap.error("embed needs --graph bipartite|tripartite")
        stage_embed(args.graph, args.data)
    elif args.stage == "classify":
        stage_classify()
    elif args.stage == "extras":
        stage_extras()
    else:
        if not args.test_data:
            ap.error("samples needs --test-data")
        stage_samples(args.test_data)
