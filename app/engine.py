"""Phase 2 prediction engine: loads the Phase 1 artifacts once and scores transactions.

No training happens here. A transaction is scored by looking up the Node2Vec embeddings of its
customer and merchant (and, for the tripartite graph, its transaction node), combining them with
the four edge operators, and running the saved Random Forests.

Node2Vec is transductive: only nodes present in the training graph have embeddings. How each case
is handled is recorded per transaction in `confidence` and `note` (see `GraphModel.score`).
"""
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
GRAPHS = ["bipartite", "tripartite"]
OPERATORS = ["hadamard", "average", "l1", "l2"]
TRACKS = ["thesis", "enhanced"]
TRACK_LABEL = {"thesis": "Thesis Model", "enhanced": "Enhanced Model (tuned)"}
OP_LABEL = {"hadamard": "Hadamard", "average": "Average", "l1": "L1", "l2": "L2"}

# Confidence levels, best to worst
IN_GRAPH, KNOWN, APPROX, LOW, UNAVAILABLE = "In graph", "Known entities", "Approximated", "Low confidence", "Unavailable"

MAX_ROWS = 500
REQUIRED_COLUMNS = ["cc_num", "merchant", "amt"]

# Classifiers are the bulk of the memory (16 forests, ~0.9 GB loaded). They are loaded on first use
# and the least recently used ones are dropped once their total exceeds this budget.
MODEL_BUDGET_MB = float(os.environ.get("MODEL_BUDGET_MB", 600))


def forest_mb(clf):
    """In-memory size of a fitted RandomForest's tree arrays, in MB."""
    total = 0
    for est in clf.estimators_:
        state = est.tree_.__getstate__()
        total += state["nodes"].nbytes + state["values"].nbytes
    return total / 1e6


class ModelCache:
    """Lazily loaded classifiers, shared by every session, bounded by MODEL_BUDGET_MB (LRU)."""

    def __init__(self, budget_mb=MODEL_BUDGET_MB):
        self.budget_mb = budget_mb
        self._models = OrderedDict()          # path -> (clf, mb)
        self._lock = threading.Lock()

    def get(self, path):
        with self._lock:
            if path in self._models:
                self._models.move_to_end(path)
                return self._models[path][0]
            clf = joblib.load(path)
            clf.n_jobs = 1                    # tiny batches: threading overhead dominates
            self._models[path] = (clf, forest_mb(clf))
            while len(self._models) > 1 and self.loaded_mb() > self.budget_mb:
                self._models.popitem(last=False)
            return clf

    def loaded_mb(self):
        return sum(mb for _, mb in self._models.values())


def edge_features(A, B, method):
    """Identical to the thesis pipeline (Appendix A.5)."""
    if method == "hadamard": return A * B
    if method == "average":  return (A + B) / 2
    if method == "l1":       return np.abs(A - B)
    if method == "l2":       return (A - B) ** 2


@dataclass
class Scored:
    probs: dict          # probs[track][operator] -> float (nan when unavailable); only scored tracks
    confidence: str
    note: str
    in_sample: bool      # the exact edge(s) were in the classifier's training set
    edges: list          # [(key_u, key_v, vec_u, vec_v)] used for the visualization


class GraphModel:
    """Embeddings + lookup tables + classifiers for one graph type."""

    def __init__(self, name, cache):
        self.name = name
        self.cache = cache
        self.kv = KeyedVectors.load(str(ARTIFACTS_DIR / f"{name}_wv.kv"), mmap="r")
        self.vectors = np.asarray(self.kv.vectors)
        self.key2idx = {k: i for i, k in enumerate(self.kv.index_to_key)}

        nodes = pd.read_parquet(ARTIFACTS_DIR / f"{name}_nodes.parquet")
        self.customer_key = dict(zip(nodes.loc[nodes.kind == "customer", "name"],
                                     nodes.loc[nodes.kind == "customer", "key"]))
        self.merchant_key = dict(zip(nodes.loc[nodes.kind == "merchant", "name"],
                                     nodes.loc[nodes.kind == "merchant", "key"]))
        self.key_name = dict(zip(nodes["key"], nodes["name"]))
        # Stand-in for an unseen entity: the centroid of all entities of that kind
        self.customer_centroid = self.vec(list(self.customer_key.values())).mean(axis=0)
        self.merchant_centroid = self.vec(list(self.merchant_key.values())).mean(axis=0)

        edges = np.load(ARTIFACTS_DIR / f"{name}_edges.npz")
        self.edge_y = edges["y"]
        self.edge_split = np.load(ARTIFACTS_DIR / f"{name}_edge_split.npy")   # 0 test, 1 trained, 2 dropped

        if name == "bipartite":
            pairs = pd.read_parquet(ARTIFACTS_DIR / "bipartite_pairs.parquet")
            self.pairs = pairs
            self.pair_edge = dict(zip(zip(pairs.cust_key, pairs.merch_key), pairs.edge))
            self.pair_label = (pairs.n_fraud.values > 0).astype(int)
            # neighbor lookups for the plot: entity key -> row positions in `pairs`
            self.cust_pairs = pairs.groupby("cust_key").indices
            self.merch_pairs = pairs.groupby("merch_key").indices
        else:
            tx = pd.read_parquet(ARTIFACTS_DIR / "tripartite_transactions.parquet",
                                 columns=["tx_key", "cust_key", "merch_key", "trans_num", "amt",
                                          "is_fraud", "edge_ct", "edge_tm"])
            self.tx = tx
            self.trans_num_row = dict(zip(tx.trans_num, range(len(tx))))
            self.tx_row = dict(zip(tx.tx_key, range(len(tx))))
            tx_vec = self.vec(tx.tx_key.tolist())
            self.tx_vec = tx_vec
            self.pair_rows = tx.groupby(["cust_key", "merch_key"]).indices
            self.cust_rows = tx.groupby("cust_key").indices
            self.merch_rows = tx.groupby("merch_key").indices
            self.cust_tx_mean = {k: tx_vec[r].mean(axis=0) for k, r in self.cust_rows.items()}
            self.merch_tx_mean = {k: tx_vec[r].mean(axis=0) for k, r in self.merch_rows.items()}
            self.tx_centroid = tx_vec.mean(axis=0)

        # Classifiers load lazily through the shared cache; PR curves are small, so load them now.
        self.model_path, self.pr = {}, {}
        for track in TRACKS:
            for op in OPERATORS:
                stem = ARTIFACTS_DIR / f"{name}_rf_{op}_{track}"
                if Path(f"{stem}.joblib").exists():
                    self.model_path[(track, op)] = str(f"{stem}.joblib")
                    self.pr[(track, op)] = dict(np.load(f"{stem}_pr.npz"))

    def vec(self, keys):
        return self.vectors[[self.key2idx[k] for k in keys]]

    def predict(self, track, op, A, B):
        path = self.model_path.get((track, op))
        if path is None or len(A) == 0:
            return np.full(len(A), np.nan)
        return self.cache.get(path).predict_proba(edge_features(A, B, op))[:, 1]

    # ------------------------------------------------------------------ resolving one transaction
    def resolve(self, cc_num, merchant, trans_num=None):
        """Return (confidence, note, in_sample, edge_list) for one transaction.

        edge_list holds the (u_key, v_key, u_vec, v_vec) of every edge to score; the transaction's
        probability is the mean over those edges (tripartite: customer-transaction and
        transaction-merchant).
        """
        ck, mk = self.customer_key.get(cc_num), self.merchant_key.get(merchant)
        if ck is None and mk is None:
            return UNAVAILABLE, "Customer and merchant both unseen: no embedding exists for either.", False, []

        if self.name == "bipartite":
            cu = self.vec([ck])[0] if ck else self.customer_centroid
            mv = self.vec([mk])[0] if mk else self.merchant_centroid
            edge = [(ck or "new-customer", mk or "new-merchant", cu, mv)]
            if ck and mk:
                e = self.pair_edge.get((ck, mk))
                if e is not None:
                    trained = self.edge_split[e] == 1
                    return (IN_GRAPH, "This customer-merchant pair is an existing edge of the graph"
                            + (" that the classifier was trained on." if trained else "."), bool(trained), edge)
                return KNOWN, "Both entities are in the graph; this pair is a new edge.", False, edge
            side = "customer" if ck is None else "merchant"
            return LOW, f"Unseen {side}: replaced by the average {side} embedding.", False, edge

        # tripartite: the transaction itself is a node
        row = self.trans_num_row.get(trans_num) if trans_num else None
        if row is not None:
            r = self.tx.iloc[row]
            tv = self.tx_vec[row]
            trained = bool((self.edge_split[[r.edge_ct, r.edge_tm]] == 1).any())
            return (IN_GRAPH, "This exact transaction (trans_num) is a node of the training graph"
                    + (" and was in the classifier's training set." if trained else "."), trained,
                    [(r.cust_key, r.tx_key, self.vec([r.cust_key])[0], tv),
                     (r.tx_key, r.merch_key, tv, self.vec([r.merch_key])[0])])

        if ck and mk:
            rows = self.pair_rows.get((ck, mk))
            if rows is not None:
                tv = self.tx_vec[rows].mean(axis=0)
                conf = APPROX
                note = (f"New transaction node, approximated by the mean embedding of this pair's "
                        f"{len(rows)} earlier transaction(s).")
            else:
                tv = (self.cust_tx_mean[ck] + self.merch_tx_mean[mk]) / 2
                conf = APPROX
                note = "New transaction node and new pair: approximated from the customer's and merchant's other transactions."
            cu, mv = self.vec([ck])[0], self.vec([mk])[0]
        else:
            side = "customer" if ck is None else "merchant"
            known_mean = self.cust_tx_mean[ck] if ck else self.merch_tx_mean[mk]
            tv = known_mean
            cu = self.vec([ck])[0] if ck else self.customer_centroid
            mv = self.vec([mk])[0] if mk else self.merchant_centroid
            conf = LOW
            note = f"Unseen {side}: replaced by the average {side} embedding; transaction node approximated from the known side."
        return conf, note, False, [(ck or "new-customer", "new-transaction", cu, tv),
                                   ("new-transaction", mk or "new-merchant", tv, mv)]

    def score(self, txs, tracks=TRACKS):
        """Score a list of (cc_num, merchant, trans_num) with the classifiers of `tracks`."""
        resolved = [self.resolve(*t) for t in txs]
        # batch every edge of every transaction per classifier: 16 predict_proba calls in total
        owner, A, B = [], [], []
        for i, (_, _, _, edges) in enumerate(resolved):
            for (_, _, a, b) in edges:
                owner.append(i); A.append(a); B.append(b)
        owner = np.array(owner, dtype=int)
        A = np.array(A).reshape(-1, self.vectors.shape[1])
        B = np.array(B).reshape(-1, self.vectors.shape[1])

        probs = {track: {} for track in tracks}
        for track in tracks:
            for op in OPERATORS:
                p = self.predict(track, op, A, B)
                sums = np.bincount(owner, weights=p, minlength=len(txs)) if len(p) else np.zeros(len(txs))
                counts = np.bincount(owner, minlength=len(txs))
                with np.errstate(invalid="ignore", divide="ignore"):
                    probs[track][op] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

        return [Scored(probs={t: {op: float(probs[t][op][i]) for op in OPERATORS} for t in tracks},
                       confidence=c, note=n, in_sample=s, edges=e)
                for i, (c, n, s, e) in enumerate(resolved)]

    # ------------------------------------------------------------------ neighborhood for the plot
    def neighborhood(self, scored, track, op, k=10, seed=0):
        """Focal edge(s) plus up to k existing edges around each focal entity, each scored.

        Returns a list of dicts {u, v, kind ('focal'|'neighbor'), prob, label (0/1, None if new)}.
        """
        rng = np.random.default_rng(seed)
        out = [{"u": u, "v": v, "kind": "focal", "label": None,
                "prob": float(self.predict(track, op, a[None], b[None])[0])}
               for (u, v, a, b) in scored.edges]
        focal = {x for e in scored.edges for x in e[:2]}

        # (entity key -> row positions, key at the other end of each row), precomputed at load time
        if self.name == "bipartite":
            p = self.pairs
            groups = [(self.cust_pairs, p.merch_key.values), (self.merch_pairs, p.cust_key.values)]
            labels = self.pair_label
        else:
            t = self.tx
            groups = [(self.cust_rows, t.tx_key.values), (self.merch_rows, t.tx_key.values)]
            labels = t.is_fraud.values

        for index, other in groups:
            for node in focal:
                rows = index.get(node)
                if rows is None:
                    continue
                rows = rows[~np.isin(other[rows], list(focal))]
                if len(rows) > k:     # keep some fraud history visible when it exists
                    fraud, rest = rows[labels[rows] == 1], rows[labels[rows] == 0]
                    n_f = min(len(fraud), k // 3)
                    rows = np.concatenate([rng.choice(fraud, n_f, replace=False),
                                           rng.choice(rest, min(len(rest), k - n_f), replace=False)])
                for r in rows:
                    out.append({"u": node, "v": other[r], "kind": "neighbor", "label": int(labels[r])})

        nb = [e for e in out if e["kind"] == "neighbor"]
        if nb:
            probs = self.predict(track, op, self.vec([e["u"] for e in nb]), self.vec([e["v"] for e in nb]))
            for e, pi in zip(nb, probs):
                e["prob"] = float(pi)
        return out


class Engine:
    def __init__(self):
        self.manifest = json.loads((ARTIFACTS_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.cache = ModelCache()
        self.graphs = {g: GraphModel(g, self.cache) for g in GRAPHS}
        self.merchants = sorted(self.graphs["tripartite"].merchant_key)
        self.customers = sorted(self.graphs["tripartite"].customer_key, key=int)

    def has_track(self, track):
        return all((track, op) in self.graphs[g].model_path for g in GRAPHS for op in OPERATORS)

    def precision_recall_at(self, graph, track, op, threshold):
        """Test-set precision/recall of a classifier at `threshold`, from the saved PR curve."""
        pr = self.graphs[graph].pr.get((track, op))
        if pr is None:
            return np.nan, np.nan
        i = np.searchsorted(pr["thresholds"], threshold, side="left")   # predict fraud iff prob >= t
        return float(pr["precision"][i]), float(pr["recall"][i])

    def score_frame(self, df, tracks=TRACKS):
        """Score a validated frame. Returns (scores_by_graph, normalized merchant names)."""
        merchants = [self.normalize_merchant(m) for m in df["merchant"]]
        trans = df["trans_num"].tolist() if "trans_num" in df.columns else [None] * len(df)
        txs = list(zip(df["cc_num"].tolist(), merchants, trans))
        return {g: self.graphs[g].score(txs, tracks) for g in GRAPHS}

    def add_tracks(self, df, scores, tracks):
        """Score `df` with any of `tracks` not yet in `scores` (in place), e.g. after a model toggle."""
        missing = [t for t in tracks if t not in scores[GRAPHS[0]][0].probs]
        if missing:
            extra = self.score_frame(df, missing)
            for g in GRAPHS:
                for old, new in zip(scores[g], extra[g]):
                    old.probs.update(new.probs)
        return scores

    def normalize_merchant(self, name):
        """The Kaggle data prefixes every merchant with 'fraud_'; accept names without it."""
        name = str(name).strip()
        if name in self.graphs["tripartite"].merchant_key:
            return name
        if f"fraud_{name}" in self.graphs["tripartite"].merchant_key:
            return f"fraud_{name}"
        return name


# ---------------------------------------------------------------------- input validation
class InputError(ValueError):
    pass


def validate_frame(df):
    """Check columns and rows of an uploaded/entered frame.

    Returns (clean_df, problems) where problems is a list of human-readable row errors; raises
    InputError when the file as a whole is unusable.
    """
    if df is None or df.empty:
        raise InputError("The file has no rows.")
    df = df.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InputError(f"Missing required column(s): {', '.join(missing)}. "
                         f"Expected at least: {', '.join(REQUIRED_COLUMNS)} (fraudTrain.csv schema). "
                         f"Found: {', '.join(map(str, df.columns[:12]))}.")
    if len(df) > MAX_ROWS:
        raise InputError(f"The file has {len(df):,} rows; the demo accepts at most {MAX_ROWS}. "
                         f"Upload a smaller sample.")

    problems, keep = [], []
    for i, r in df.reset_index(drop=True).iterrows():
        cc = re.sub(r"\.0$", "", str(r["cc_num"]).strip())
        errs = []
        if not re.fullmatch(r"\d{8,19}", cc):
            errs.append("cc_num must be 8-19 digits")
        if not str(r["merchant"]).strip() or pd.isna(r["merchant"]):
            errs.append("merchant is empty")
        amt = pd.to_numeric(r["amt"], errors="coerce")
        if pd.isna(amt) or amt < 0:
            errs.append("amt must be a non-negative number")
        if errs:
            problems.append(f"Row {i + 2}: " + "; ".join(errs))   # +2: header line, 1-based
        else:
            keep.append(i)
    clean = df.reset_index(drop=True).loc[keep].copy()
    clean["cc_num"] = clean["cc_num"].map(lambda x: re.sub(r"\.0$", "", str(x).strip()))
    clean["merchant"] = clean["merchant"].astype(str).str.strip()
    clean["amt"] = pd.to_numeric(clean["amt"])
    if "trans_num" in clean.columns:
        clean["trans_num"] = clean["trans_num"].astype(str).str.strip()
    return clean.reset_index(drop=True), problems
