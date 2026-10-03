# Credit Card Fraud Detection: Graph ML Live Demo

Live demo for the thesis comparing **bipartite** and **tripartite** transaction graphs for fraud
detection (Node2Vec embeddings → edge operators → Random Forest).

Two phases:

| Phase | What | Where | Time |
|---|---|---|---|
| 1 (offline) | Thesis code (Appendix A) → graphs, embeddings, 8 Random Forests ("Thesis Model") | `pipeline/phase1_thesis.py` | ~1 h |
| 1b (offline) | GridSearchCV-tuned Random Forests ("Enhanced Model") on the same embeddings/split | `pipeline/phase1b_enhanced.py` | ~1–3 h |
| 2 (website) | Streamlit app that only loads `artifacts/` and scores transactions in well under a second | `streamlit_app.py`, `app/` | – |

## 1. Offline pipeline

Uses the Anaconda Python that already has `node2vec`, `gensim`, `scikit-learn` and `networkx`
(or `pip install -r requirements-pipeline.txt`). Run from the project root; each stage is a separate
process (the tripartite embedding needs ~5–6 GB RAM, as in the thesis' Colab runs).

```bash
python -m pipeline.phase1_thesis embed --graph bipartite  --data path/to/fraudTrain.csv
python -m pipeline.phase1_thesis embed --graph tripartite --data path/to/fraudTrain.csv
python -m pipeline.phase1_thesis classify          # Thesis Model: 8 RFs + PR curves + manifest
python -m pipeline.phase1_thesis extras            # optional: K-Means (A.6) + grouped split (A.7)
python -m pipeline.phase1_thesis samples --test-data path/to/fraudTest.csv.zip
python -m pipeline.phase1b_enhanced                # Enhanced Model (resumable; prints per combination)
```

`phase1_thesis.py` contains Appendix A.1–A.7 unchanged apart from Colab-specific lines
(`drive.mount`, `kagglehub`); every addition for the demo is marked `# [demo]`. The main addition:
the thesis code numbers nodes with `enumerate(set(...))`, whose order differs between Python
processes and was never saved, so the pipeline re-derives the mapping in the same process,
**asserts it against the built graph**, and saves it (`*_nodes.parquet`, `bipartite_pairs.parquet`,
`tripartite_transactions.parquet`). Without it the website could not find a customer's embedding.

Because Node2Vec walks are unseeded (as in the thesis), a re-run gives slightly different
embeddings and metrics than Table 4.1. The Model Card shows both the published table and the
re-run metrics of the models actually being served.

### `artifacts/` contents
- `{graph}_wv.kv` (+ `.vectors.npy`), `{graph}_edges.npz`: embeddings and labelled edges (thesis code)
- `{graph}_rf_{operator}_{thesis|enhanced}.joblib` + `_pr.npz`: classifiers and test-set PR curves
- `{graph}_edge_split.npy`: per edge, 0 = test, 1 = trained on, 2 = train but dropped by undersampling
- lookup tables (above), `sample_transactions.csv` (held-out rows from `fraudTest.csv`), `manifest.json`

## 2. Website

```bash
streamlit run streamlit_app.py
```

Pages: **Predict** (single entry form, CSV upload up to 500 rows, sample data, 2×4 probability
heatmap, local subgraph, threshold slider, Thesis/Enhanced/side-by-side toggle, CSV download),
**Model Card**, **About / Methodology**.

### How unseen entities are handled
| Case | Bipartite | Tripartite |
|---|---|---|
| `trans_num` is in the training graph | – | real transaction node (**In graph**, in-sample) |
| customer & merchant known, pair exists | existing edge (**In graph**) | new tx node ≈ mean of that pair's tx nodes (**Approximated**) |
| customer & merchant known, new pair | real embeddings, new edge (**Known entities**) | new tx node ≈ mean of customer's and merchant's tx means (**Approximated**) |
| one side unseen | unseen side → centroid of its kind (**Low confidence**) | same, tx node ≈ known side's tx mean (**Low confidence**) |
| both unseen | **Unavailable** | **Unavailable** |

The tripartite score is the mean P(fraud) of the transaction's two edges. The amount is not a
classifier input (it only weighted the random walks).

### Deploying to Streamlit Community Cloud
Push the repo **including `artifacts/`** (`.gitignore` already excludes `build/`, raw CSVs and
secrets). In *New app → Advanced settings* pick **Python 3.12** (the pins below have no wheels for
3.13) and `streamlit_app.py` as the entry point. `requirements.txt` holds runtime dependencies only,
pinned to the training environment: scikit-learn must match the pickled forests, gensim the `.kv`
files. Every file is under GitHub's 100 MB limit (largest: `tripartite_wv.kv.vectors.npy`, 68 MB),
so no Git LFS is needed.

**Memory.** Classifiers load on first use, only for the model being viewed, through a shared LRU
cache capped at `MODEL_BUDGET_MB` (default 600; set it as a root-level key in the app's *Secrets*
to change it). Measured RSS: ~570 MB after startup, ~890 MB once the Thesis Model has scored,
~1.23 GB peak with the Enhanced Model or side by side. A lower budget lowers the peak but makes
Enhanced/side-by-side scoring reload evicted forests (~0.5 s each).
